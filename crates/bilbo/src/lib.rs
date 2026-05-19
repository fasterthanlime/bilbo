//! Recover a value's destination type at runtime by reading our *own* DWARF
//! debug info, guided by the caller's stack frame. Format-agnostic: this is
//! the ELF/DWARF/unwinding support layer. A consumer (e.g. `bilbo-json`)
//! does the actual binding.
//!
//! [`resolved`] walks the frame and our DWARF on the first call from a site,
//! producing a [`Resolved`] (`Ty` + a generic per-type extension slot)
//! cached *per type* and reached via a cheap per-call-site -> type lookup.
//! Downstream crates consume only the [`plan::Ty`] and stash their own
//! compiled artifact in [`Resolved::ext`].

pub mod plan;

mod dwarf;
mod frame;
mod resolve;

use std::sync::Arc;

pub use frame::{Raw, raw};
pub use resolve::{Resolved, resolved};

/// Resolve (and cache, two levels) whatever local `ptr` aliases in the
/// *caller's* frame: call site -> type -> [`Resolved`].
///
/// This is the convenience entry for code that calls it *directly* (the
/// caller is then exactly one frame up). A wrapper that wants to stay the
/// one-frame-up boundary itself — like `bilbo-json`'s `from_json` — should
/// instead call [`raw`] (it inlines) followed by [`resolved`].
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
