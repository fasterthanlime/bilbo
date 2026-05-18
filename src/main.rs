//! Demo. `from_json` figures out `Endpoint`'s layout from this binary's own
//! DWARF, guided by the stack frame, then binds the JSON into raw memory.

use tracing::info;

fn main() {
    tracing_subscriber::fmt()
        .with_max_level(tracing::Level::TRACE)
        .with_target(false)
        .without_time()
        .init();

    #[derive(Debug)]
    struct Endpoint {
        host: String,
        port: u16,
        tags: Vec<String>,
    }

    let mut e: std::mem::MaybeUninit<Endpoint> = std::mem::MaybeUninit::uninit();
    // Safety: `e` is exactly the local `ptr` will be matched to.
    unsafe {
        dwarf_json::from_json(
            r#"
            {
              "host": "rustweek.org",
              "port": 443,
              "tags": ["conf", "rust", "🦀"],
            }
        "#,
            &mut e as *mut _ as *mut u8,
        );
    }
    // Safety: yolo
    let e = unsafe { e.assume_init() };
    info!(
        "🎉 interp: host={:?} port={} tags={:?}",
        e.host, e.port, e.tags
    );

    // Same thing, but bound by cranelift-compiled code.
    let mut j: std::mem::MaybeUninit<Endpoint> = std::mem::MaybeUninit::uninit();
    // Safety: `j` is exactly the local `ptr` will be matched to.
    unsafe {
        dwarf_json::from_json_jit(
            r#"{ "host": "jit.dev", "port": 8443, "tags": ["fast","🦞"], }"#,
            &mut j as *mut _ as *mut u8,
        );
    }
    let j = unsafe { j.assume_init() };
    info!(
        "🎉 jit:    host={:?} port={} tags={:?}",
        j.host, j.port, j.tags
    );

    // And the cranelift parser that eats raw bytes (no Json tree).
    let mut k: std::mem::MaybeUninit<Endpoint> = std::mem::MaybeUninit::uninit();
    // Safety: `k` is exactly the local `ptr` will be matched to.
    unsafe {
        dwarf_json::from_json_jit_parse(
            r#"{ "host": "parse.jit", "port": 9001, "tags": ["raw","bytes","🥖"], }"#,
            &mut k as *mut _ as *mut u8,
        );
    }
    let k = unsafe { k.assume_init() };
    info!(
        "🎉 jitp:   host={:?} port={} tags={:?}",
        k.host, k.port, k.tags
    );
}
