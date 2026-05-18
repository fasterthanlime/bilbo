//! Two-level memoization, because the deserializer is a property of the
//! *type*, not the call site:
//!
//! * **L1** `caller_pc -> TypeKey` — every call does this (cheap: read the
//!   return address from our frame record, hash-lookup). `framehop` and the
//!   DWARF local-finding only run on an L1 miss.
//! * **L2** `TypeKey -> Resolved` — the [`Ty`] and the JIT'd function, built
//!   once *per type* and shared by every call site that uses that type.
//!
//! `TypeKey` is the type's DWARF DIE `(unit, offset)`: two different call
//! sites that both fill an `Endpoint` resolve to the very same type DIE, so
//! they share one classify and one cranelift compile.

use std::collections::HashMap;
use std::sync::{Arc, Mutex, OnceLock};

use tracing::info;

use crate::jit::Parser;
use crate::plan::Ty;
use crate::{dwarf, frame};

type TypeKey = (usize, usize); // (unit index, DIE offset)

/// Everything the hot path needs for a given type, built once.
pub struct Resolved {
    pub ty: Ty,
    /// The cranelift raw-bytes parser, compiled lazily on first use.
    pub jit_parser: OnceLock<Parser>,
}

static L1: OnceLock<Mutex<HashMap<u64, TypeKey>>> = OnceLock::new();
static L2: OnceLock<Mutex<HashMap<TypeKey, Arc<Resolved>>>> = OnceLock::new();

pub fn resolved(raw: &frame::Raw, ptr: u64) -> Arc<Resolved> {
    let l1 = L1.get_or_init(|| Mutex::new(HashMap::new()));
    let l2 = L2.get_or_init(|| Mutex::new(HashMap::new()));

    let pc = frame::caller_pc(*raw);
    if let Some(&tk) = l1.lock().unwrap().get(&pc) {
        // Hot path: callsite known -> type known -> deserializer cached.
        return l2.lock().unwrap().get(&tk).expect("L2 present").clone();
    }

    // L1 miss: now (and only now) pay for a real CFI unwind to find which
    // local `ptr` aliases, and what its type DIE is.
    let caller = frame::unwind(*raw);
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
    let tk: TypeKey = (ui, ty_off.0);

    // L2: classify this type exactly once, shared across call sites.
    let arc = {
        let mut g = l2.lock().unwrap();
        g.entry(tk)
            .or_insert_with(|| {
                let ty = dwarf::classify(&store.dwarf, unit, ty_off);
                info!("classified type {tk:?} (`{name}`) -> {ty:?}");
                Arc::new(Resolved {
                    ty,
                    jit_parser: OnceLock::new(),
                })
            })
            .clone()
    };
    l1.lock().unwrap().insert(pc, tk);
    info!("callsite {pc:#x} -> type {tk:?} (`{name}`)");
    arc
}
