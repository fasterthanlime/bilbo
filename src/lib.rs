//! Deserialize JSON into raw memory by reading our *own* DWARF debug info at
//! runtime to recover the destination type from the stack frame.
//!
//! The work splits in two:
//!
//! * **cold** — [`resolve`] walks the frame and our DWARF on the first call
//!   from a site, producing a [`Resolved`] (`Ty` + JIT slot) cached *per
//!   type* and reached via a cheap per-call-site -> type lookup.
//! * **hot** — a backend ([`interp`] or the cranelift [`jit`]) binds parsed
//!   [`json`] into memory using only that `Ty`. No file I/O, no DWARF.

pub mod interp;
pub mod jit;
pub mod json;
pub mod plan;

mod dwarf;
mod frame;
mod resolve;

use std::sync::Arc;

pub use resolve::Resolved;

/// Opt-in tracing for diagnostics (used by the demo and benches).
pub fn debug_init() {
    let _ = tracing_subscriber::fmt()
        .with_max_level(tracing::Level::INFO)
        .with_target(false)
        .without_time()
        .try_init();
}

/// Resolve (and cache, two levels) whatever local `ptr` aliases in the
/// *caller's* frame: call site -> type -> [`Resolved`] (`Ty` + JIT slot).
///
/// `#[inline(never)]` so the caller is exactly one frame up.
///
/// # Safety
/// `ptr` must point to the local you intend to fill.
#[inline(never)]
pub unsafe fn resolve(ptr: *mut u8) -> Arc<Resolved> {
    let raw = frame::raw();
    resolve::resolved(&raw, ptr as u64)
}

/// The whole pipeline for the demo: resolve the type from our own DWARF,
/// parse the JSON, and bind it with the interpreter.
///
/// `#[inline(never)]` so the frame capture sees the real caller.
///
/// # Safety
/// `ptr` must point to uninitialized storage of exactly the type of the
/// local it aliases in the caller's frame.
#[inline(never)]
pub unsafe fn from_json(s: &str, ptr: *mut u8) {
    let raw = frame::raw();
    let r = resolve::resolved(&raw, ptr as u64);
    let value = json::parse(s);
    unsafe { interp::run(ptr, &r.ty, &value) };
}

/// Same pipeline, but the bind step is a cranelift-compiled function
/// specialized to the *type* (compiled once per type, shared across sites).
///
/// # Safety
/// Same contract as [`from_json`].
#[inline(never)]
pub unsafe fn from_json_jit(s: &str, ptr: *mut u8) {
    let raw = frame::raw();
    let r = resolve::resolved(&raw, ptr as u64);
    let f = *r.jit.get_or_init(|| jit::compile(&r.ty));
    let value = json::parse(s);
    unsafe { f(ptr, &value as *const json::Json) };
}

/// The fastest path: a cranelift-compiled parser specialized to the type
/// that walks the raw JSON bytes straight into the struct — no `Json` tree.
///
/// # Safety
/// Same contract as [`from_json`].
#[inline(never)]
pub unsafe fn from_json_jit_parse(s: &str, ptr: *mut u8) {
    let raw = frame::raw();
    let r = resolve::resolved(&raw, ptr as u64);
    let f = *r.jit_parser.get_or_init(|| jit::compile_parser(&r.ty));
    unsafe { f(ptr, s.as_ptr(), s.len()) };
}
