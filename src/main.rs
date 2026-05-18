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
    // Safety: yolo
    let e = unsafe { e.assume_init() };
    info!(
        "🎉 reconstructed: host={:?} port={} tags={:?}",
        e.host, e.port, e.tags
    );
    info!("   debug view: {e:#?}");
}
