//! Deserialize JSON into raw memory by reading our *own* DWARF debug info at
//! runtime to recover the destination type from the stack frame.
//!
//! The work splits in two:
//!
//! * **cold** — [`resolve`] walks the frame and our DWARF (once per call
//!   site, then cached) to produce a [`plan::Ty`]: a self-contained layout.
//! * **hot** — a backend ([`interp`], later a cranelift JIT) binds parsed
//!   [`json`] into memory using only that `Ty`. No file I/O, no DWARF.

pub mod interp;
pub mod json;
pub mod plan;

mod dwarf;
mod frame;
mod resolve;

use std::sync::Arc;

use plan::Ty;

/// Opt-in tracing for diagnostics (used by the demo and benches).
pub fn debug_init() {
    let _ = tracing_subscriber::fmt()
        .with_max_level(tracing::Level::INFO)
        .with_target(false)
        .without_time()
        .try_init();
}

/// Resolve (and cache) the layout of whatever local `ptr` aliases in the
/// *caller's* frame. Call this once per site; reuse the returned `Ty`.
///
/// `#[inline(never)]` so the caller is exactly one frame up.
/// # Safety
/// `ptr` must point to the local you intend to fill; the returned `Ty`
/// describes that local's type.
#[inline(never)]
pub unsafe fn resolve(ptr: *mut u8) -> Arc<Ty> {
    let caller = frame::caller();
    resolve::plan_for(&caller, ptr as u64)
}

/// The whole pipeline for the demo: resolve the type from our own DWARF,
/// parse the JSON, and bind it with the interpreter.
///
/// `#[inline(never)]` so the frame walk sees the real caller.
///
/// # Safety
/// `ptr` must point to uninitialized storage of exactly the type of the
/// local it aliases in the caller's frame.
#[inline(never)]
pub unsafe fn from_json(s: &str, ptr: *mut u8) {
    let caller = frame::caller();
    let ty = resolve::plan_for(&caller, ptr as u64);
    let value = json::parse(s);
    unsafe { interp::run(ptr, &ty, &value) };
}
