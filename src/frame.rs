//! Recover the caller's frame the correct way: real CFI-based unwinding
//! (`__unwind_info` / `__eh_frame`) via `framehop`, not a hand-rolled
//! frame-pointer chain. Under `-O3` the FP chain assumptions break; the
//! unwinder gets the caller's pc / sp / fp right regardless.

use std::sync::OnceLock;

use framehop::Unwinder;
use framehop::aarch64::{CacheAarch64, UnwindRegsAarch64, UnwinderAarch64};
use framehop::{ExplicitModuleSectionInfo, FrameAddress, Module};
use object::{Object, ObjectSection, ObjectSegment};
use tracing::debug;

/// What we recovered about whoever called us.
#[derive(Debug, Clone, Copy)]
pub struct Caller {
    /// Return address into the caller, de-ASLR'd to match the link-time
    /// addresses DWARF stores in `DW_AT_low_pc`.
    pub static_pc: u64,
    /// The caller's frame-pointer register (`x29`). `DW_OP_breg29` and an
    /// `fbreg` whose `DW_AT_frame_base` is `DW_OP_reg29` resolve against it.
    pub caller_fp: u64,
    /// The caller's stack pointer at the call site (== our CFA). rustc emits
    /// stack locals as `DW_OP_breg31 (SP) + N`, so they live here.
    pub caller_sp: u64,
}

type Unw = UnwinderAarch64<&'static [u8]>;

/// The registers of the public function the user called, captured cheaply.
/// `framehop` is *not* run here — that only happens on a cache miss.
#[derive(Debug, Clone, Copy)]
pub struct Raw {
    pub pc: u64,
    pub sp: u64,
    pub fp: u64,
    pub lr: u64,
}

/// Capture this frame's registers. `#[inline(always)]` so they belong to the
/// *public* function the user actually called (`from_json` / `resolve`),
/// which has a real frame record (forced frame pointers + `#[inline(never)]`).
#[inline(always)]
pub fn raw() -> Raw {
    let (pc, sp, fp, lr): (u64, u64, u64, u64);
    // Safety: reading our own pc/sp/x29/x30.
    unsafe {
        core::arch::asm!(
            "adr {pc}, .",
            "mov {sp}, sp",
            "mov {fp}, x29",
            "mov {lr}, x30",
            pc = out(reg) pc,
            sp = out(reg) sp,
            fp = out(reg) fp,
            lr = out(reg) lr,
            options(nomem, nostack),
        );
    }
    Raw { pc, sp, fp, lr }
}

/// The cache key: the caller's return address, de-ASLR'd. This is just the
/// `lr` slot of our own AAPCS64 frame record (`*(fp + 8)`) — two loads, no
/// unwinding. Same value `framehop` would return as the caller PC.
#[inline(always)]
pub fn caller_pc(r: Raw) -> u64 {
    let ret = unsafe { *((r.fp + 8) as *const u64) };
    ret - main_image_slide()
}

/// The cold path: actually unwind one frame (CFI-correct, works under `-O3`)
/// to recover the caller's `sp`/`fp` so we can locate the aliased local.
pub fn unwind(r: Raw) -> Caller {
    unwind_one(r.pc, r.sp, r.fp, r.lr)
}

fn unwind_one(pc: u64, sp: u64, fp: u64, lr: u64) -> Caller {
    let unwinder = unwinder();
    let mut cache = CacheAarch64::<_>::new();
    let mut regs = UnwindRegsAarch64::new(lr, sp, fp);
    let mut read_stack = |addr: u64| {
        // We're unwinding our own live stack; these addresses are valid.
        Ok(unsafe { (addr as *const u64).read() })
    };
    let ret = unwinder
        .unwind_frame(
            FrameAddress::from_instruction_pointer(pc),
            &mut regs,
            &mut cache,
            &mut read_stack,
        )
        .expect("unwind_frame")
        .expect("caller frame (we are not the root)");

    let slide = main_image_slide();
    let c = Caller {
        static_pc: ret - slide,
        caller_fp: regs.fp(),
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
        let obj = object::File::parse(bytes).expect("parse exe Mach-O");

        let seg = |name: &str| {
            obj.segments().find(|s| s.name() == Ok(Some(name)))
        };
        let sec_range_data = |name: &str| {
            obj.section_by_name(name)
                .map(|s| (s.address()..s.address() + s.size(), s.data().ok()))
        };

        let text_seg = seg("__TEXT").expect("__TEXT segment");
        let base_svma = text_seg.address();
        let text_seg_svma =
            text_seg.address()..text_seg.address() + text_seg.size();
        let slide = main_image_slide();
        let base_avma = base_svma + slide;
        let avma_range =
            (text_seg_svma.start + slide)..(text_seg_svma.end + slide);

        let mut info = ExplicitModuleSectionInfo {
            base_svma,
            text_segment_svma: Some(text_seg_svma.clone()),
            text_segment: text_seg.data().ok(),
            ..Default::default()
        };
        if let Some((r, d)) = sec_range_data("__text") {
            info.text_svma = Some(r);
            info.text = d;
        }
        if let Some((r, _)) = sec_range_data("__stubs") {
            info.stubs_svma = Some(r);
        }
        if let Some((r, _)) = sec_range_data("__stub_helper") {
            info.stub_helper_svma = Some(r);
        }
        if let Some((r, _)) = sec_range_data("__got") {
            info.got_svma = Some(r);
        }
        if let Some((_, d)) = sec_range_data("__unwind_info") {
            info.unwind_info = d;
        }
        if let Some((r, d)) = sec_range_data("__eh_frame") {
            info.eh_frame_svma = Some(r);
            info.eh_frame = d;
        }

        let module = Module::new(
            exe.to_string_lossy().into_owned(),
            avma_range,
            base_avma,
            info,
        );
        let mut u = UnwinderAarch64::new();
        u.add_module(module);
        u
    })
}

// dyld gives us the ASLR slide of any loaded image.
unsafe extern "C" {
    fn _dyld_image_count() -> u32;
    fn _dyld_get_image_name(image_index: u32) -> *const core::ffi::c_char;
    fn _dyld_get_image_vmaddr_slide(image_index: u32) -> isize;
}

/// The ASLR slide, so a DWARF `DW_AT_low_pc` (link-time) can be turned into
/// the runtime address we can actually `bl` to from JIT'd code.
pub(crate) fn image_slide() -> u64 {
    main_image_slide()
}

fn main_image_slide() -> u64 {
    static SLIDE: OnceLock<u64> = OnceLock::new();
    *SLIDE.get_or_init(|| {
        let exe = std::env::current_exe().expect("current_exe");
        let exe = std::fs::canonicalize(&exe).unwrap_or(exe);
        let count = unsafe { _dyld_image_count() };
        for i in 0..count {
            let name = unsafe { _dyld_get_image_name(i) };
            if name.is_null() {
                continue;
            }
            let cstr = unsafe { std::ffi::CStr::from_ptr(name) };
            let path = std::path::Path::new(cstr.to_str().unwrap_or(""));
            let canon =
                std::fs::canonicalize(path).unwrap_or_else(|_| path.into());
            if canon == exe {
                return unsafe { _dyld_get_image_vmaddr_slide(i) } as u64;
            }
        }
        unsafe { _dyld_get_image_vmaddr_slide(0) as u64 }
    })
}
