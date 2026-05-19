//! Deserialize JSON into raw memory. The destination type is recovered from
//! the program's own DWARF by [`bilbo`] (the cold path); the hot path here
//! binds parsed [`json`] into memory using only that [`bilbo::plan::Ty`].
//!
//! Two backends: the [`interp`]reter and the cranelift [`jit`]. The compiled
//! parser is cached *per type* in [`bilbo::Resolved::ext`].

pub mod interp;
pub mod jit;
pub mod json;
pub mod tramp;

mod jitdump;

// Re-export the cold-path surface so callers have one import root.
pub use bilbo::{Raw, Resolved, raw, resolve, resolved};

/// Opt-in tracing for benches: WARN only, so failure diagnostics show but
/// cranelift's verbose `log::info!("defining function …")` (the whole CLIF)
/// doesn't flood benchmark output. (The demo sets its own TRACE subscriber.)
pub fn debug_init() {
    let _ = tracing_subscriber::fmt()
        .with_max_level(tracing::Level::WARN)
        .with_target(false)
        .without_time()
        .try_init();
}

/// The whole pipeline for the demo: resolve the type from our own DWARF,
/// parse the JSON, and bind it with the interpreter.
///
/// `#[inline(never)]` so the frame capture sees the real caller (this is the
/// one-frame-up boundary; [`bilbo::raw`] inlines into it).
///
/// # Safety
/// `ptr` must point to uninitialized storage of exactly the type of the
/// local it aliases in the caller's frame.
#[inline(never)]
pub unsafe fn from_json(s: &str, ptr: *mut u8) {
    let raw = bilbo::raw();
    let r = bilbo::resolved(&raw, ptr as u64);
    let value = json::parse(s);
    unsafe { interp::run(ptr, &r.ty, &value) };
}

/// A cranelift-compiled parser specialized to the type that walks the raw
/// JSON bytes straight into the struct — no `Json` tree, no shims beyond the
/// allocator. The compiled function is cached per type.
///
/// # Safety
/// Same contract as [`from_json`].
#[inline(never)]
pub unsafe fn from_json_jit_parse(s: &str, ptr: *mut u8) {
    let raw = bilbo::raw();
    let r = bilbo::resolved(&raw, ptr as u64);
    let f: jit::Parser = *r.ext(jit::compile_parser);
    unsafe { f(ptr, s.as_ptr(), s.len()) };
}
