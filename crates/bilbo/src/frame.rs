//! The frame layer's public face. The platform-specific bodies — register
//! capture, the CFI unwind, and the ASLR slide — live in [`crate::platform`]
//! (one of `darwin` / `linux` is compiled in). This module just re-exports
//! them so the rest of the crate keeps using stable `crate::frame::*` paths
//! and never has to know which platform it's on.

pub use crate::platform::{Raw, caller_pc, image_slide, raw, unwind};
