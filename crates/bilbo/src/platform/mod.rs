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

#[cfg(test)]
mod tests {
    use super::{caller_pc, dwarf_regs, image_slide, raw, unwind};

    /// The arch-specific DWARF register opcodes are the heart of the port;
    /// pin them so a wrong constant fails loudly (and documents intent on
    /// whichever arch runs the suite).
    #[test]
    fn dwarf_reg_opcodes_match_arch() {
        #[cfg(target_arch = "x86_64")]
        {
            // DWARF reg 6 = rbp, 7 = rsp.
            assert_eq!(dwarf_regs::OP_REG_FP, 0x56); // DW_OP_reg6
            assert_eq!(dwarf_regs::OP_BREG_FP, 0x76); // DW_OP_breg6
            assert_eq!(dwarf_regs::OP_BREG_SP, 0x77); // DW_OP_breg7
        }
        #[cfg(target_arch = "aarch64")]
        {
            // x29 = frame pointer, x31 = SP.
            assert_eq!(dwarf_regs::OP_REG_FP, 0x6d); // DW_OP_reg29
            assert_eq!(dwarf_regs::OP_BREG_FP, 0x8d); // DW_OP_breg29
            assert_eq!(dwarf_regs::OP_BREG_SP, 0x8f); // DW_OP_breg31
        }
        // The breg<fp>/breg<sp>/reg<fp> opcodes must stay distinct, and
        // distinct from the arch-neutral DW_OP_fbreg (0x91) /
        // DW_OP_call_frame_cfa (0x9c) the evaluator also matches.
        let ops = [
            dwarf_regs::OP_REG_FP,
            dwarf_regs::OP_BREG_FP,
            dwarf_regs::OP_BREG_SP,
            0x91,
            0x9c,
        ];
        for (i, a) in ops.iter().enumerate() {
            for b in &ops[i + 1..] {
                assert_ne!(a, b, "DWARF opcode constants must be distinct");
            }
        }
    }

    /// `image_slide()` is memoized for the process; it must be stable.
    #[test]
    fn image_slide_is_stable() {
        assert_eq!(image_slide(), image_slide());
    }

    /// The boundary `raw()` inlines into. `#[inline(never)]` so the frame it
    /// captures is *this* function and the recovered caller is the `#[test]`
    /// that called us — exactly the `from_json`/`resolve` contract.
    #[inline(never)]
    fn fast_and_slow() -> (u64, u64) {
        let r = raw();
        // Fast path: the L1 cache key, read straight from the frame record.
        let fast = caller_pc(r);
        // Cold path: a real CFI unwind of one frame.
        let slow = unwind(r).static_pc;
        std::hint::black_box((fast, slow))
    }

    /// The load-bearing cross-platform invariant: the cheap frame-record
    /// cache key must equal the CFI-unwound caller PC. If the platform
    /// backend (asm, framehop arch, object format, or ASLR slide) is wrong,
    /// these diverge — this is the single best smoke test for the port, and
    /// the one to watch when validating the macOS backend.
    #[test]
    fn fast_path_pc_matches_cfi_unwind() {
        let (fast, slow) = fast_and_slow();
        assert_ne!(fast, 0, "recovered caller PC should be non-zero");
        assert_eq!(
            fast, slow,
            "L1 frame-record PC ({fast:#x}) must equal the CFI-unwound \
             caller PC ({slow:#x})"
        );
    }
}
