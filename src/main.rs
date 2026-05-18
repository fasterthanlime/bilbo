//! Deserialize JSON into raw memory by reading our *own* DWARF debug info at
//! runtime — recovering the destination type from the stack frame, not from
//! any generic parameter. This is a very silly idea, implemented earnestly.

mod dwarf;
mod frame;
mod json;
mod poke;

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
    from_json(
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

/// Deserialize `s` into `*ptr`. We never learn the type from a generic — we
/// read the frame pointer, find our caller in our own DWARF, and identify
/// exactly which of its locals `ptr` aliases. Panics, loudly, on any snag.
#[inline(never)]
fn from_json(s: &str, ptr: *mut u8) {
    info!("from_json: ptr={:#x}; consulting our own debug info", ptr as u64);

    let caller = frame::caller();

    dwarf::with(|d, units| {
        let (ui, sp) = dwarf::subprogram_at(d, units, caller.static_pc)
            .expect("no subprogram for caller PC");
        let unit = &units[ui];

        let (name, ty_off) = dwarf::local_at_address(
            d,
            unit,
            sp,
            caller.caller_fp,
            caller.caller_sp,
            caller.static_pc,
            ptr as u64,
        )
        .expect("no local matched the pointer we were handed");
        let ty = dwarf::classify(d, unit, ty_off);
        info!("destination is local `{name}` : {ty:#?}");

        let value = json::parse(s);
        unsafe { poke::poke(ptr, &ty, &value) };
        info!("done; returning to a hopeful caller");
    });
}
