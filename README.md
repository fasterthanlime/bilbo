# dwarf-json

A JSON deserializer that figures out the destination type **at runtime by
reading the program's own DWARF debug info**, then **JIT-compiles a
specialized parser with cranelift** — and beats `serde_json` on the
nativejson-benchmark trio.

It started as a stupid idea:

```rust
let mut e: MaybeUninit<Endpoint> = MaybeUninit::uninit();
from_json(r#"{ "host": "rustweek.org", "port": 443 }"#,
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

`&str` → owned Rust value, Apple M-series, release, vs **default**
`serde_json` (divan medians):

| input | serde_json | dwarf-json | |
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
fast.

**Cold (once per type, cached):**

1. `frame.rs` — capture registers and unwind exactly one frame with
   [`framehop`](https://crates.io/crates/framehop) (real CFI: macOS
   compact-unwind / `.eh_frame`). This gives the caller's de-ASLR'd PC,
   SP and FP — correct even under `-O3`, where a hand-rolled
   frame-pointer walk breaks.
2. `dwarf.rs` — load our own `.dSYM` once into a process-global
   `Store`. Map that PC to its `DW_TAG_subprogram`, then evaluate every
   local's DWARF location expression against the caller's frame to find
   *which local the pointer aliases* (it's the `MaybeUninit<T>` one).
   Recover its type DIE.
3. `classify` turns the type DIE into a self-contained [`plan::Ty`]:
   field offsets, primitive sizes, the real `ptr`/`cap`/`len` offsets
   inside `String`/`Vec` (Rust does not promise their order — we learn
   it from DWARF), niche `Option`, `()`, tuples, `BTreeMap`.
4. `resolve.rs` — a two-level cache: call-site PC → type → resolved
   plan + JIT'd function. The deserializer is a property of the *type*,
   not the call site, so two sites filling the same type share one
   compile.

**Hot (every call, no DWARF, no file I/O):**

- `jit.rs` — cranelift compiles a function specialized to the `Ty`:
  field-name bytes and offsets baked in as constants, a `memchr`/
  hybrid-SIMD scanner, the whole parse+bind in one pass, no
  intermediate `Json` tree. Or `interp.rs`, a plain interpreter over
  the same `Ty`, kept as a baseline.
- `jitdump.rs` — emits `/tmp/jit-<pid>.dump` (perf jitdump) so
  profilers (e.g. [stax](https://github.com/bearcove/stax)) can name
  and disassemble the JIT'd code instead of showing `<unresolved>`.

The one honest caveat: a `BTreeMap` has no DWARF-discoverable layout we
can poke our way into (B-tree nodes, unstable). So for maps we call the
*real* `std::collections::BTreeMap` through thin `#[inline(never)]`
trampolines (`tramp.rs`), monomorphized once per value type, whose
addresses we resolve — from DWARF, like everything else.

## Running it

```sh
cargo run                 # the demo: reconstructs an Endpoint, narrates
cargo bench --bench de    # small struct vs serde_json / facet-json
cargo bench --bench citm  # citm_catalog.json
cargo bench --bench canada
cargo bench --bench twitter  # full fidelity: enums, Box, recursion
```

Each bench asserts byte-for-byte equality with `serde_json` before
timing.

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

- **macOS / AArch64 only.** Uses dyld, AArch64 inline asm, NEON, and
  framehop's AArch64 unwinder.
- Needs debug info: every profile sets `split-debuginfo = "packed"`
  and `.cargo/config.toml` forces frame pointers.
- Wildly `unsafe` by construction. `from_json(s, ptr)` writes a fully
  reconstructed value through a raw pointer based on what it *believes*
  the type is. This is a toy / proof of cursedness, not a crate to
  depend on.
- Optimized builds reuse stack slots; we disambiguate by preferring the
  `MaybeUninit<T>` local at the matched address (which is exactly the
  API contract).

## Module map

| file | role |
|---|---|
| `frame.rs` | capture regs, framehop one-frame unwind, ASLR slide |
| `dwarf.rs` | `.dSYM` store, PC→subprogram, local-by-pointer, classify |
| `plan.rs` | `Ty` — the cached, DWARF-free layout artifact |
| `resolve.rs` | two-level cache: callsite → type → resolved |
| `jit.rs` | cranelift backend (specialized parser) + runtime shims |
| `interp.rs` | plain interpreter backend (baseline) |
| `json.rs` | tiny lenient JSON parser (interp/baseline only) |
| `jitdump.rs` | perf jitdump emitter for profilers |
| `tramp.rs` | real-`BTreeMap` trampolines, resolved via DWARF |

## License

Do whatever you want; it's a stupid idea.
