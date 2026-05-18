//! The cold path, memoized. The first call from a given call site walks the
//! DWARF; every later call from that site is a hash lookup. The key is the
//! caller's return address: a fixed call site always passes the same local,
//! so the resolved [`Ty`] is stable.

use std::collections::HashMap;
use std::sync::{Arc, Mutex, OnceLock};

use tracing::info;

use crate::plan::Ty;
use crate::{dwarf, frame};

static CACHE: OnceLock<Mutex<HashMap<u64, Arc<Ty>>>> = OnceLock::new();

pub fn plan_for(caller: &frame::Caller, ptr: u64) -> Arc<Ty> {
    let cache = CACHE.get_or_init(|| Mutex::new(HashMap::new()));
    if let Some(hit) = cache.lock().unwrap().get(&caller.static_pc) {
        return hit.clone();
    }

    let store = dwarf::store();
    let (ui, sp) =
        dwarf::subprogram_at(&store.dwarf, &store.units, caller.static_pc)
            .expect("no subprogram for caller PC");
    let unit = &store.units[ui];
    let (name, ty_off) = dwarf::local_at_address(
        &store.dwarf,
        unit,
        sp,
        caller.caller_fp,
        caller.caller_sp,
        caller.static_pc,
        ptr,
    )
    .expect("no local matched the pointer we were handed");
    let ty = dwarf::classify(&store.dwarf, unit, ty_off);
    info!(
        "resolved local `{name}` at pc {:#x} -> {ty:?}",
        caller.static_pc
    );

    let arc = Arc::new(ty);
    cache
        .lock()
        .unwrap()
        .insert(caller.static_pc, arc.clone());
    arc
}
