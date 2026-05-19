//! Everything that differs between the platforms we run on: capturing our
//! own registers, the CFI unwind, the ASLR slide, where our own DWARF
//! lives, and the handful of DWARF opcodes whose register numbers are
//! architecture-specific.
//!
//! Exactly one backend is compiled in:
//!
//! * [`darwin`] — aarch64 + Mach-O + dyld (the platform bilbo was born on).
//! * [`linux`]  — x86_64 + ELF + `dl_iterate_phdr`.
//!
//! Both expose the same surface, re-exported here so the rest of the crate
//! is platform-agnostic: `Raw`, [`raw`], [`caller_pc`], [`unwind`],
//! [`image_slide`], [`dwarf_bytes`], and the `dwarf_regs` opcode constants.

/// What we recovered about whoever called us. The shape is identical on
/// every platform; only how each field is *obtained* (which register, which
/// unwinder, which object format) differs per backend.
#[derive(Debug, Clone, Copy)]
pub struct Caller {
    /// Return address into the caller, de-ASLR'd to match the link-time
    /// addresses DWARF stores in `DW_AT_low_pc`.
    pub static_pc: u64,
    /// The caller's frame-pointer register. A `DW_OP_breg<fp>` and an
    /// `fbreg` whose `DW_AT_frame_base` is `DW_OP_reg<fp>` resolve against
    /// it.
    pub caller_fp: u64,
    /// The caller's stack pointer at the call site (== our CFA). rustc
    /// emits stack locals relative to this.
    pub caller_sp: u64,
}

#[cfg(target_os = "macos")]
mod darwin;
#[cfg(target_os = "macos")]
pub use darwin::{
    Raw, caller_pc, dwarf_bytes, dwarf_regs, image_slide, raw, unwind,
};

#[cfg(target_os = "linux")]
mod linux;
#[cfg(target_os = "linux")]
pub use linux::{
    Raw, caller_pc, dwarf_bytes, dwarf_regs, image_slide, raw, unwind,
};
