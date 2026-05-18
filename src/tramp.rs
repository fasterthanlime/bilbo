//! The one honest caveat: a `BTreeMap` has no DWARF-discoverable layout we
//! can poke our way into (B-tree nodes, unstable). So for maps we call the
//! *real* `std::collections::BTreeMap` — through these thin, ABI-pinned,
//! `#[inline(never)]` trampolines, monomorphized once per value type `V`
//! (key is always `String`). The JIT resolves their addresses from DWARF by
//! matching the `V` template parameter, like everything else.

use std::collections::BTreeMap;

/// Placement-construct an empty `BTreeMap<String, V>` at `dst`.
///
/// # Safety
/// `dst` must point to uninitialized storage of the right size/align.
#[inline(never)]
pub unsafe extern "C" fn map_new_at<V>(dst: *mut BTreeMap<String, V>) {
    unsafe { dst.write(BTreeMap::new()) };
}

/// Move `*k` / `*v` into the map at `dst`.
///
/// # Safety
/// `dst` must be an initialized map; `k`/`v` initialized and not used after
/// (their ownership is moved into the map).
#[inline(never)]
pub unsafe extern "C" fn map_insert<V>(
    dst: *mut BTreeMap<String, V>,
    k: *mut String,
    v: *mut V,
) {
    unsafe {
        let m = &mut *dst;
        let _ = m.insert(k.read(), v.read());
    }
}

/// Force the `map_new_at::<V>` / `map_insert::<V>` monomorphizations to be
/// emitted (out-of-line, so they have an address) for value type `V`. Call
/// once per `V` the schema uses, before deserializing.
pub fn force<V>() {
    let a = map_new_at::<V> as *const ();
    let b = map_insert::<V> as *const ();
    std::hint::black_box((a, b));
}
