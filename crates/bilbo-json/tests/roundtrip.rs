//! End-to-end tests for the whole cold path: capture the frame (`raw`),
//! find the caller (`caller_pc` / CFI `unwind`), recover the destination
//! type from this test binary's own DWARF, then bind JSON into it with
//! both backends (interpreter and cranelift JIT).
//!
//! These are the cross-platform acid test: they exercise the entire
//! platform layer (inline asm, framehop arch, object format, ASLR slide,
//! DWARF source + opcodes) on whichever OS runs them. Running this suite on
//! macOS validates the `platform::darwin` backend the same way it validates
//! `platform::linux` here.
//!
//! Each `#[test]` is itself the caller frame `from_json` unwinds to, so the
//! `MaybeUninit<T>` local must live directly in the test function — exactly
//! the documented API contract.

use std::mem::MaybeUninit;

use serde::Deserialize;

#[derive(Deserialize, PartialEq, Debug)]
struct Endpoint {
    host: String,
    port: u16,
    tags: Vec<String>,
}

#[derive(Deserialize, PartialEq, Debug)]
struct Inner {
    x: u32,
    label: String,
}

#[derive(Deserialize, PartialEq, Debug)]
struct Config {
    name: String,
    retries: i32,
    // Tagged `Option` (present and absent) + a nested struct + a `Box`.
    timeout: Option<u64>,
    fallback: Option<String>,
    inner: Inner,
    boxed: Box<Inner>,
    list: Vec<u32>,
}

// Recursive, heap-linked: the A2/A3 capability (`Box` + a type cycle).
#[derive(Deserialize, PartialEq, Debug)]
struct Node {
    val: u64,
    next: Option<Box<Node>>,
}

#[test]
fn interp_endpoint_fields() {
    let json = r#"{ "host": "rustweek.org", "port": 443,
                    "tags": ["conf", "rust", "🦀"] }"#;
    let mut e: MaybeUninit<Endpoint> = MaybeUninit::uninit();
    // Safety: `e` is exactly the local `ptr` will be matched to.
    unsafe {
        bilbo_json::from_json(json, &mut e as *mut _ as *mut u8);
    }
    let e = unsafe { e.assume_init() };
    assert_eq!(e.host, "rustweek.org");
    assert_eq!(e.port, 443);
    assert_eq!(e.tags, ["conf", "rust", "🦀"]);
}

#[test]
fn jit_endpoint_fields() {
    let json = r#"{ "host": "jit.example", "port": 9001,
                    "tags": ["a", "b"] }"#;
    let mut e: MaybeUninit<Endpoint> = MaybeUninit::uninit();
    // Safety: same contract as above.
    unsafe {
        bilbo_json::from_json_jit_parse(json, &mut e as *mut _ as *mut u8);
    }
    let e = unsafe { e.assume_init() };
    assert_eq!(e.host, "jit.example");
    assert_eq!(e.port, 9001);
    assert_eq!(e.tags, ["a", "b"]);
}

const CONFIG_JSON: &str = r#"
{
  "name": "svc",
  "retries": -3,
  "timeout": 5000,
  "fallback": null,
  "inner": { "x": 7, "label": "deep" },
  "boxed": { "x": 42, "label": "heap" },
  "list": [1, 2, 3, 4]
}
"#;

#[test]
fn interp_matches_serde_json() {
    let want: Config = serde_json::from_str(CONFIG_JSON).unwrap();
    let mut got: MaybeUninit<Config> = MaybeUninit::uninit();
    // Safety: `got` is the local being filled.
    let got = unsafe {
        bilbo_json::from_json(CONFIG_JSON, &mut got as *mut _ as *mut u8);
        got.assume_init()
    };
    assert_eq!(got, want);
}

#[test]
fn jit_matches_serde_json() {
    let want: Config = serde_json::from_str(CONFIG_JSON).unwrap();
    let mut got: MaybeUninit<Config> = MaybeUninit::uninit();
    // Safety: `got` is the local being filled.
    let got = unsafe {
        bilbo_json::from_json_jit_parse(
            CONFIG_JSON,
            &mut got as *mut _ as *mut u8,
        );
        got.assume_init()
    };
    assert_eq!(got, want);
}

#[test]
fn jit_recursive_boxed_matches_serde_json() {
    let json = r#"{ "val": 1,
                    "next": { "val": 2,
                              "next": { "val": 3, "next": null } } }"#;
    let want: Node = serde_json::from_str(json).unwrap();
    let mut got: MaybeUninit<Node> = MaybeUninit::uninit();
    // Safety: `got` is the local being filled.
    let got = unsafe {
        bilbo_json::from_json_jit_parse(json, &mut got as *mut _ as *mut u8);
        got.assume_init()
    };
    assert_eq!(got, want);
    assert_eq!(got.next.unwrap().next.unwrap().val, 3);
}
