//! No symbol names, no `backtrace` crate. We read the AArch64 frame pointer
//! and walk exactly one frame record to learn (a) where in our own code we
//! were called from and (b) the caller's CFA, which is the anchor every
//! DWARF local location is expressed relative to.

use tracing::info;

/// What we recovered about whoever called `from_json`.
#[derive(Debug, Clone, Copy)]
pub struct Caller {
    /// Return address into the caller, already de-ASLR'd so it matches the
    /// link-time addresses DWARF stores in `DW_AT_low_pc`.
    pub static_pc: u64,
    /// The caller's frame-pointer register (`x29`) value — the AAPCS64 frame
    /// record at `[from_json_fp]`. `DW_OP_fbreg` / `DW_OP_breg29` resolve
    /// against this (the subprogram's `DW_AT_frame_base` is `DW_OP_reg29`).
    pub caller_fp: u64,
    /// The caller's stack pointer at the call site. By definition this is
    /// `from_json`'s CFA, which on AAPCS64 is `from_json_fp + 16`. rustc
    /// emits stack locals as `DW_OP_breg31 (SP) + N`, so they live here.
    pub caller_sp: u64,
}

/// Read the current frame pointer (x29). Must be called from a function with
/// a real frame (`#[inline(never)]` + forced frame pointers).
#[inline(always)]
fn frame_pointer() -> u64 {
    let fp: u64;
    // Safety: just reading a register.
    unsafe { core::arch::asm!("mov {}, x29", out(reg) fp, options(nomem, nostack)) };
    fp
}

/// Given `from_json`'s own frame pointer, recover its caller.
///
/// An AArch64 frame record is two words at `[fp]`: `[fp] = caller_fp`,
/// `[fp + 8] = return address`. So one hop up gets us the caller's frame
/// pointer and the return address sitting inside the caller.
pub fn caller_of(my_fp: u64) -> Caller {
    let caller_fp = unsafe { *(my_fp as *const u64) };
    let return_addr = unsafe { *((my_fp + 8) as *const u64) };
    // from_json's CFA == the caller's SP at the `bl` (a `bl` doesn't touch
    // SP), and on AAPCS64 the CFA is fp + 16.
    let caller_sp = my_fp + 16;

    let slide = main_image_slide();
    let static_pc = return_addr - slide;
    info!(
        "frame walk: my_fp={my_fp:#x} caller_fp={caller_fp:#x} \
         caller_sp={caller_sp:#x} ret={return_addr:#x} slide={slide:#x} \
         static_pc={static_pc:#x}"
    );
    Caller {
        static_pc,
        caller_fp,
        caller_sp,
    }
}

/// Convenience: capture our frame pointer and resolve the caller in one go.
#[inline(always)]
pub fn caller() -> Caller {
    caller_of(frame_pointer())
}

// dyld lets us ask for the ASLR slide of any loaded image. Image 0 is the
// main executable; we still match by path to be honest about it.
unsafe extern "C" {
    fn _dyld_image_count() -> u32;
    fn _dyld_get_image_name(image_index: u32) -> *const core::ffi::c_char;
    fn _dyld_get_image_vmaddr_slide(image_index: u32) -> isize;
}

fn main_image_slide() -> u64 {
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
        let canon = std::fs::canonicalize(path).unwrap_or_else(|_| path.into());
        if canon == exe {
            return unsafe { _dyld_get_image_vmaddr_slide(i) } as u64;
        }
    }
    // Fall back to image 0, which is by convention the main executable.
    unsafe { _dyld_get_image_vmaddr_slide(0) as u64 }
}
