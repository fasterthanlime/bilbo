# bilbo

A Cargo workspace for recovering a value's type **at runtime by reading
the program's own DWARF debug info**, guided by the stack frame.

Who hangs out with elves and dwarves all day? That's right — Bilbo the
hobbit. So:

- **`bilbo`** — the ELF/DWARF/unwinding support layer: capture a frame,
  unwind it, find which local a pointer aliases, and classify its type
  into a DWARF-free `plan::Ty`. Format-agnostic.
- **`bilbo-json`** — a JSON deserializer built on `bilbo` that
  **JIT-compiles a specialized parser with cranelift** — and beats
  `serde_json` on the nativejson-benchmark trio. (Room for a
  `bilbo-postcard` etc. later — they'd share `bilbo`'s cold path.)

It started as a stupid idea:

```rust
let mut e: MaybeUninit<Endpoint> = MaybeUninit::uninit();
bilbo_json::from_json(r#"{ "host": "rustweek.org", "port": 443 }"#,
                      &mut e as *mut _ as *mut u8);
let e = unsafe { e.assume_init() };
```

`from_json` is handed a `*mut u8` and a string. It has *no generic
parameter*, no `Deserialize` impl, nothing. It figures out that the
pointer aliases an `Endpoint { host: String, port: u16, … }` — and how
that type is laid out in memory — by unwinding its own stack and parsing
its own debug info. Then it builds the value by poking bytes.

This is not a serious library. It is, however, faster than serde_json.

## Numbers

`&str` → owned Rust value, release, vs **default** `serde_json` (divan
medians on Apple M-series; also runs on x86_64 Linux, not yet timed there):

| input | serde_json | bilbo-json | |
|---|---|---|---|
| `citm_catalog.json` (1.7 MB) | ~1.06 ms | **~845 µs** | ~1.25× |
| `canada.json` (2.3 MB) | ~2.23 ms | **~1.34 ms** | ~1.7× |
| `twitter.json` (632 KB) | ~397 µs | **~322 µs** | ~1.2× |

Output is byte-for-byte identical to `serde_json` (citm, twitter —
including twitter's recursive `retweeted_status`). On `canada` we're
actually *more* correct: serde_json's default float
parser is best-effort (~1 ULP off on some coordinates); ours
(`fast-float2`) is correctly-rounded, so the gate is "structure exact +
coords within serde's own error."

## How it works

Two phases. The cursed part happens once; the hot path is boring and
fast. The cold phase is all `bilbo`; the hot phase is `bilbo-json`.

**Cold — `bilbo` (once per type, cached):**

1. `platform` — capture registers and unwind exactly one frame with
   [`framehop`](https://crates.io/crates/framehop) (real CFI: macOS
   compact-unwind / `.eh_frame`; Linux `.eh_frame` / `.eh_frame_hdr`).
   This gives the caller's de-ASLR'd PC, SP and FP — correct even under
   `-O3`, where a hand-rolled frame-pointer walk breaks.
2. `dwarf.rs` — load our own DWARF once into a process-global `Store`
   (macOS: the `.dSYM`; Linux: the `.debug_*` embedded in
   `current_exe()`). Map that PC to its `DW_TAG_subprogram`, then
   evaluate every local's DWARF location expression against the caller's
   frame to find *which local the pointer aliases* (it's the
   `MaybeUninit<T>` one). Recover its type DIE.
3. `classify` turns the type DIE into a self-contained `plan::Ty`:
   field offsets, primitive sizes, the real `ptr`/`cap`/`len` offsets
   inside `String`/`Vec` (Rust does not promise their order — we learn
   it from DWARF), niche *and* tagged `Option`, `()`, tuples, `Box<T>`,
   `BTreeMap`, and recursive types (a cycle in the DIE graph becomes a
   back-edge in `Ty`, tied off with `Arc<OnceLock<Ty>>`).
4. `resolve.rs` — a two-level cache: call-site PC → type → `Resolved`
   (the `Ty` plus a generic per-type `ext` slot). The deserializer is a
   property of the *type*, not the call site, so two sites filling the
   same type share one classify — and one compile, because the consumer
   stashes its compiled artifact in `Resolved::ext`, keyed by that same
   per-type cache.

**Hot — `bilbo-json` (every call, no DWARF, no file I/O):**

- `jit.rs` — cranelift compiles a function specialized to the `Ty`:
  field-name bytes and offsets baked in as constants, a `memchr`/
  hybrid-SIMD scanner, the whole parse+bind in one pass, no
  intermediate `Json` tree. Object keys dispatch through a linear
  word-compare chain for narrow structs, but a wide struct (twitter's
  `User` is 41 fields) instead dispatches in O(1) on the key *length*
  via a `br_table`, then a tiny per-length chain — the difference
  between losing and winning on twitter. Recursive types compile one
  function per cycle and call into it. Or `interp.rs`, a plain
  interpreter over the same `Ty`, kept as a baseline. The compiled
  parser lands in `bilbo::Resolved::ext` (one compile per type).
- `jitdump.rs` — emits `/tmp/jit-<pid>.dump` (perf jitdump) so
  profilers (e.g. [stax](https://github.com/bearcove/stax)) can name
  and disassemble the JIT'd code instead of showing `<unresolved>`.

The one honest caveat: a `BTreeMap` has no DWARF-discoverable layout we
can poke our way into (B-tree nodes, unstable). So for maps we call the
*real* `std::collections::BTreeMap` through thin `#[inline(never)]`
trampolines (`bilbo-json`'s `tramp.rs`), monomorphized once per value
type, whose addresses `bilbo` resolves — from DWARF, like everything
else.

## Running it

```sh
cargo run -p bilbo-json                 # demo: Endpoint + tagged-Option + Box + recursive, narrated
cargo bench -p bilbo-json --bench de    # small struct vs serde_json / facet-json
cargo bench -p bilbo-json --bench citm  # citm_catalog.json
cargo bench -p bilbo-json --bench canada
cargo bench -p bilbo-json --bench twitter  # full fidelity: enums, Box, recursion
```

`BILBO_JSON_PROFILE=1 cargo run -p bilbo-json --release` prints the
JIT'd parser's address and hammers it forever, for `stax` to sample.

Each bench gates against `serde_json` before timing: byte-for-byte for
`citm`/`twitter`/`de`, and structure-exact + correctly-rounded floats
for `canada` (where serde's default parser is the less accurate one).

## Supported types

Structs, tuples (positional arrays), `Vec<T>`, `String`, `&str`,
`bool`, `char`, `u8..u64`/`i8..i64`, `f32`/`f64`, `()`, `Option<T>`
(both niche — `Option<String>`, `Option<Box<_>>` — and tagged —
`Option<u64>`, `Option<bool>`), an absent struct key meaning `None`
(serde's rule), `Box<T>`, `BTreeMap<String, V>`, and **recursive
types** (`Status::retweeted_status: Option<Box<Status>>` ties the knot
via `Arc<OnceLock<Ty>>`; the JIT emits one function per cycle). This is
exactly what the full-fidelity `twitter.json` benchmark exercises.

## Caveats

- **Two targets only: macOS/aarch64 and Linux/x86_64.** macOS uses
  dyld, a Mach-O `.dSYM`, NEON, and framehop's aarch64 unwinder; Linux
  uses `dl_iterate_phdr`, the ELF's embedded `.debug_*`, and framehop's
  x86_64 unwinder. Each backend has its own arch inline asm. Nothing
  else is supported.
- Needs debug info, configured in `.cargo/config.toml` / `[profile.*]`:
  macOS keeps a packed `.dSYM`; Linux forces `split-debuginfo=off` so
  the full `.debug_*` stays embedded in the ELF. Both force frame
  pointers.
- Wildly `unsafe` by construction. `from_json(s, ptr)` writes a fully
  reconstructed value through a raw pointer based on what it *believes*
  the type is. This is a toy / proof of cursedness, not a crate to
  depend on.
- Optimized builds reuse stack slots; we disambiguate by preferring the
  `MaybeUninit<T>` local at the matched address (which is exactly the
  API contract).

## Module map

`crates/bilbo` — the ELF/DWARF support layer:

| file | role |
|---|---|
| `platform/mod.rs` | platform-agnostic surface; cfg-selects one backend |
| `platform/darwin.rs` | aarch64 + Mach-O + dyld + `.dSYM` |
| `platform/linux.rs` | x86_64 + ELF + `dl_iterate_phdr` + embedded `.debug_*` |
| `frame.rs` | thin re-export of the active backend's regs/unwind |
| `dwarf.rs` | DWARF store, PC→subprogram, local-by-pointer, classify |
| `plan.rs` | `Ty` — the cached, DWARF-free layout artifact |
| `resolve.rs` | two-level cache: callsite → type → `Resolved` (`Ty` + generic `ext`) |

`crates/bilbo-json` — the JSON consumer:

| file | role |
|---|---|
| `jit.rs` | cranelift backend (specialized parser) + runtime shims |
| `interp.rs` | plain interpreter backend (baseline) |
| `json.rs` | tiny lenient JSON parser (interp/baseline only) |
| `jitdump.rs` | perf jitdump emitter for profilers |
| `tramp.rs` | real-`BTreeMap` trampolines, resolved via DWARF |

## License

Licensed under either of

- Apache License, Version 2.0 ([LICENSE-APACHE](LICENSE-APACHE) or
  <http://www.apache.org/licenses/LICENSE-2.0>)
- MIT license ([LICENSE-MIT](LICENSE-MIT) or
  <http://opensource.org/licenses/MIT>)

at your option. It's a stupid idea; do whatever you want with it.

Unless you explicitly state otherwise, any contribution intentionally
submitted for inclusion in the work by you, as defined in the Apache-2.0
license, shall be dual licensed as above, without any additional terms or
conditions.
