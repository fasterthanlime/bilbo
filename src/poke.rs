//! The recursive writer. Given a [`Ty`] recovered from DWARF, a [`Json`]
//! value, and a destination pointer, it pokes correctly-laid-out bytes into
//! raw memory — recursing into structs and rebuilding `String`/`Vec`/`&str`
//! honestly (real allocations, three words written at the DWARF-discovered
//! offsets, since Rust does not promise their order).

use std::alloc::{Layout, alloc};

use tracing::info;

use crate::dwarf::{SeqLayout, Ty};
use crate::json::Json;

/// # Safety
/// `dst` must point to uninitialized memory laid out exactly as `ty`
/// describes (it does: `ty` came from the DWARF for *this* destination).
pub unsafe fn poke(dst: *mut u8, ty: &Ty, val: &Json) {
    match ty {
        Ty::Bool => unsafe { *dst = matches!(val, Json::Bool(true)) as u8 },
        Ty::Char => {
            let c = val.as_str().chars().next().unwrap_or('\0') as u32;
            unsafe { write_bytes(dst, &c.to_le_bytes()) };
        }
        Ty::U(n) => unsafe {
            write_bytes(dst, &val.as_u128().to_le_bytes()[..*n as usize])
        },
        Ty::I(n) => unsafe {
            write_bytes(dst, &val.as_i128().to_le_bytes()[..*n as usize])
        },
        Ty::F32 => unsafe {
            write_bytes(dst, &(val.as_f64() as f32).to_le_bytes())
        },
        Ty::F64 => unsafe { write_bytes(dst, &val.as_f64().to_le_bytes()) },

        Ty::Struct { name, fields, .. } => {
            info!("poking struct `{name}` with {} field(s)", fields.len());
            for f in fields {
                let fv = val.get(&f.name).unwrap_or_else(|| {
                    panic!("JSON missing key `{}` for `{name}`", f.name)
                });
                info!("  .{} @ +{} : {:?}", f.name, f.offset, f.ty);
                unsafe { poke(dst.add(f.offset), &f.ty, fv) };
            }
        }

        Ty::Str(seq) => {
            let bytes = val.as_str().as_bytes();
            let p = alloc_copy(bytes, 1);
            unsafe { write_seq(dst, seq, p, bytes.len(), bytes.len()) };
        }

        Ty::StrRef { ptr_off, len_off } => {
            let s: &'static str =
                Box::leak(val.as_str().to_owned().into_boxed_str());
            unsafe {
                write_word(dst.add(*ptr_off), s.as_ptr() as u64);
                write_word(dst.add(*len_off), s.len() as u64);
            }
        }

        Ty::Vec {
            elem,
            elem_size,
            elem_align,
            seq,
        } => {
            let Json::Array(items) = val else {
                panic!("expected JSON array for Vec, got {val:?}");
            };
            let n = items.len();
            let esz = *elem_size as usize;
            let ealign = (*elem_align as usize).max(1);
            let base = if n == 0 {
                // Mirror `Vec::new()`: no allocation, dangling aligned ptr,
                // cap 0 — so the eventual `Drop` does not try to free it.
                ealign as *mut u8
            } else {
                let layout =
                    Layout::from_size_align(esz * n, ealign).unwrap();
                let p = unsafe { alloc(layout) };
                assert!(!p.is_null(), "allocation failed");
                for (i, item) in items.iter().enumerate() {
                    unsafe { poke(p.add(i * esz), elem, item) };
                }
                p
            };
            unsafe { write_seq(dst, seq, base, n, n) };
        }

        Ty::Unknown(what) => panic!("don't know how to write type `{what}`"),
    }
}

/// Allocate `bytes.len()` bytes (matching `Global`, so the reconstructed
/// value's `Drop` frees it correctly with `cap == len`), copy, return ptr.
fn alloc_copy(bytes: &[u8], align: usize) -> *mut u8 {
    if bytes.is_empty() {
        return align.max(1) as *mut u8;
    }
    let layout = Layout::from_size_align(bytes.len(), align).unwrap();
    let p = unsafe { alloc(layout) };
    assert!(!p.is_null(), "allocation failed");
    unsafe { std::ptr::copy_nonoverlapping(bytes.as_ptr(), p, bytes.len()) };
    p
}

unsafe fn write_seq(
    dst: *mut u8,
    seq: &SeqLayout,
    ptr: *mut u8,
    cap: usize,
    len: usize,
) {
    unsafe {
        write_word(dst.add(seq.ptr_off), ptr as u64);
        write_word(dst.add(seq.cap_off), cap as u64);
        write_word(dst.add(seq.len_off), len as u64);
    }
}

unsafe fn write_word(dst: *mut u8, v: u64) {
    unsafe { std::ptr::write_unaligned(dst as *mut u64, v) };
}

unsafe fn write_bytes(dst: *mut u8, bytes: &[u8]) {
    unsafe {
        std::ptr::copy_nonoverlapping(bytes.as_ptr(), dst, bytes.len())
    };
}
