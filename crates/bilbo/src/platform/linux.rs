//! x86_64 + ELF + `dl_iterate_phdr`. The Linux counterpart of
//! [`super::darwin`], same surface, different machinery:
//!
//! * registers via x86_64 inline asm (no link register — the return
//!   address lives on the stack);
//! * CFI unwinding through `.eh_frame` / `.eh_frame_hdr` with
//!   `framehop::x86_64`;
//! * the ASLR slide is the program's ELF load bias, from
//!   `dl_iterate_phdr`;
//! * our own DWARF is embedded in the executable (the build forces
//!   `split-debuginfo=off` for Linux), so we read it straight back out of
//!   `current_exe()` — no separate debug file.

use std::sync::OnceLock;

use framehop::Unwinder;
use framehop::x86_64::{CacheX86_64, UnwindRegsX86_64, UnwinderX86_64};
use framehop::{ExplicitModuleSectionInfo, FrameAddress, Module};
use object::{Object, ObjectSection};
use tracing::{debug, info};

use super::Caller;

/// The register numbers `rustc` uses for the DWARF location opcodes we
/// understand, on x86_64: DWARF register 6 is `rbp` (frame pointer), 7 is
/// `rsp` (stack pointer).
pub mod dwarf_regs {
    /// `DW_OP_reg6` — the value *is* rbp (used as a `DW_AT_frame_base`).
    pub const OP_REG_FP: u8 = 0x56;
    /// `DW_OP_breg6` — rbp + SLEB128 offset.
    pub const OP_BREG_FP: u8 = 0x76;
    /// `DW_OP_breg7` — rsp + SLEB128 offset.
    pub const OP_BREG_SP: u8 = 0x77;
}

type Unw = UnwinderX86_64<&'static [u8]>;

/// The registers of the public function the user called, captured cheaply.
/// `framehop` is *not* run here — that only happens on a cache miss. Unlike
/// AArch64 there is no link register; the return address lives in memory
/// (`*(rbp + 8)` under the System V frame record), so `Raw` carries none.
#[derive(Debug, Clone, Copy)]
pub struct Raw {
    pub pc: u64,
    pub sp: u64,
    pub fp: u64,
}

/// Capture this frame's registers. `#[inline(always)]` so they belong to the
/// *public* function the user actually called (`from_json` / `resolve`),
/// which has a real frame record (forced frame pointers + `#[inline(never)]`).
#[inline(always)]
pub fn raw() -> Raw {
    let (pc, sp, fp): (u64, u64, u64);
    // Safety: reading our own rip/rsp/rbp. `lea {}, [rip]` materializes an
    // address inside this (inlined-into-the-public) function — all framehop
    // needs to find the right FDE for the current frame.
    unsafe {
        core::arch::asm!(
            "lea {pc}, [rip]",
            "mov {sp}, rsp",
            "mov {fp}, rbp",
            pc = out(reg) pc,
            sp = out(reg) sp,
            fp = out(reg) fp,
            options(nomem, nostack),
        );
    }
    Raw { pc, sp, fp }
}

/// The cache key: the caller's return address, de-ASLR'd. Under the System
/// V frame record the saved rbp is at `*rbp` and the return address at
/// `*(rbp + 8)` — two loads, no unwinding. Same value `framehop` would
/// return as the caller PC.
#[inline(always)]
pub fn caller_pc(r: Raw) -> u64 {
    let ret = unsafe { *((r.fp + 8) as *const u64) };
    ret - load_bias()
}

/// The cold path: actually unwind one frame (CFI-correct via `.eh_frame`,
/// works even where the FP chain wouldn't) to recover the caller's
/// `sp`/`fp` so we can locate the aliased local.
pub fn unwind(r: Raw) -> Caller {
    let unwinder = unwinder();
    let mut cache = CacheX86_64::<_>::new();
    // x86_64 takes (ip, sp, bp) — there is no link register.
    let mut regs = UnwindRegsX86_64::new(r.pc, r.sp, r.fp);
    let mut read_stack = |addr: u64| {
        // We're unwinding our own live stack; these addresses are valid.
        Ok(unsafe { (addr as *const u64).read() })
    };
    let ret = unwinder
        .unwind_frame(
            FrameAddress::from_instruction_pointer(r.pc),
            &mut regs,
            &mut cache,
            &mut read_stack,
        )
        .expect("unwind_frame")
        .expect("caller frame (we are not the root)");

    let slide = load_bias();
    let c = Caller {
        static_pc: ret - slide,
        caller_fp: regs.bp(),
        // After one frame, framehop's recovered sp is the caller's CFA
        // (== rsp at the call site), which is what `DW_OP_call_frame_cfa`
        // and the `fbreg`-relative locals resolve against.
        caller_sp: regs.sp(),
    };
    debug!(
        "unwound: ret={ret:#x} slide={slide:#x} static_pc={:#x} \
         caller_fp={:#x} caller_sp={:#x}",
        c.static_pc, c.caller_fp, c.caller_sp
    );
    c
}

/// The unwinder for our own executable, built once. The mapped binary is
/// leaked to `'static` (process-lifetime; nothing to free).
fn unwinder() -> &'static Unw {
    static UNW: OnceLock<Unw> = OnceLock::new();
    UNW.get_or_init(|| {
        let exe = std::env::current_exe().expect("current_exe");
        let file = std::fs::File::open(&exe).expect("open exe");
        let mmap =
            unsafe { memmap2::Mmap::map(&file) }.expect("mmap exe");
        let bytes: &'static [u8] = Box::leak(Box::new(mmap));
        let obj = object::File::parse(bytes).expect("parse exe ELF");

        let sec_range_data = |name: &str| {
            obj.section_by_name(name)
                .map(|s| (s.address()..s.address() + s.size(), s.data().ok()))
        };

        // For ELF, `base_svma` is zero (framehop converts SVMA<->relative
        // against it); the load bias is the AVMA base.
        let slide = load_bias();
        let (text_svma, text_data) =
            sec_range_data(".text").expect(".text section");
        let avma_range =
            (text_svma.start + slide)..(text_svma.end + slide);

        let mut info = ExplicitModuleSectionInfo {
            base_svma: 0,
            text_svma: Some(text_svma),
            text: text_data,
            ..Default::default()
        };
        if let Some((r, d)) = sec_range_data(".eh_frame") {
            info.eh_frame_svma = Some(r);
            info.eh_frame = d;
        }
        if let Some((r, d)) = sec_range_data(".eh_frame_hdr") {
            info.eh_frame_hdr_svma = Some(r);
            info.eh_frame_hdr = d;
        }

        let module = Module::new(
            exe.to_string_lossy().into_owned(),
            avma_range,
            slide, // base_avma
            info,
        );
        let mut u = UnwinderX86_64::new();
        u.add_module(module);
        u
    })
}

// The prefix of glibc's `struct dl_phdr_info`. We only ever read
// `dlpi_addr` (offset 0), but spelling the next few fields keeps the
// layout honest.
#[repr(C)]
struct DlPhdrInfo {
    dlpi_addr: u64,
    dlpi_name: *const core::ffi::c_char,
    dlpi_phdr: *const core::ffi::c_void,
    dlpi_phnum: u16,
}

unsafe extern "C" {
    fn dl_iterate_phdr(
        callback: unsafe extern "C" fn(
            *mut DlPhdrInfo,
            usize,
            *mut core::ffi::c_void,
        ) -> core::ffi::c_int,
        data: *mut core::ffi::c_void,
    ) -> core::ffi::c_int;
}

/// The ASLR slide: the ELF load bias of the main program, so a DWARF
/// `DW_AT_low_pc` (link-time) can be turned into the runtime address we can
/// actually `call` from JIT'd code. For a PIE this is nonzero; for a
/// non-PIE `ET_EXEC` it is zero. The first object `dl_iterate_phdr` reports
/// is always the main executable, so we grab its `dlpi_addr` and stop.
pub fn image_slide() -> u64 {
    load_bias()
}

fn load_bias() -> u64 {
    static SLIDE: OnceLock<u64> = OnceLock::new();
    *SLIDE.get_or_init(|| {
        unsafe extern "C" fn cb(
            info: *mut DlPhdrInfo,
            _size: usize,
            data: *mut core::ffi::c_void,
        ) -> core::ffi::c_int {
            // SAFETY: `data` is the `&mut Option<u64>` we passed in; `info`
            // points at a valid `dl_phdr_info` for the duration of the call.
            unsafe {
                let out = &mut *(data as *mut Option<u64>);
                if out.is_none() {
                    *out = Some((*info).dlpi_addr);
                }
            }
            1 // stop after the first object (the main program)
        }
        let mut out: Option<u64> = None;
        unsafe {
            dl_iterate_phdr(cb, &mut out as *mut _ as *mut core::ffi::c_void);
        }
        out.unwrap_or(0)
    })
}

/// The raw bytes of the object that holds our DWARF. On Linux that's the
/// executable itself — the build forces `split-debuginfo=off`, so the full
/// `.debug_*` sections are embedded. Mapped once and leaked to `'static`;
/// `dwarf::store()` parses it.
pub fn dwarf_bytes() -> &'static [u8] {
    static BYTES: OnceLock<&'static [u8]> = OnceLock::new();
    *BYTES.get_or_init(|| {
        let exe = std::env::current_exe().expect("current_exe");
        info!("reading our own DWARF from {} (once)", exe.display());
        let file = std::fs::File::open(&exe).expect("open exe");
        let mmap =
            unsafe { memmap2::Mmap::map(&file) }.expect("mmap exe");
        Box::leak(Box::new(mmap))
    })
}
