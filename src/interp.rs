//! The interpreter backend. Given a [`Ty`] (already recovered from DWARF and
//! cached), a [`Json`] value, and a destination pointer, it pokes
//! correctly-laid-out bytes into raw memory — recursing into structs and
//! rebuilding `String`/`Vec`/`&str` honestly (real allocations, three words
//! written at the DWARF-discovered offsets, since Rust doesn't promise their
//! order). No DWARF, no file I/O: this is the hot path.

use std::alloc::{Layout, alloc};

use crate::json::Json;
use crate::plan::{SeqLayout, Ty};

/// # Safety
/// `dst` must point to uninitialized memory laid out exactly as `ty`
/// describes (it does: `ty` came from the DWARF for *this* destination).
pub unsafe fn run(dst: *mut u8, ty: &Ty, val: &Json) {
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

        Ty::Struct { name, fields } => {
            for f in fields {
                let fv = val.get(&f.name).unwrap_or_else(|| {
                    panic!("JSON missing key `{}` for `{name}`", f.name)
                });
                unsafe { run(dst.add(f.offset), &f.ty, fv) };
            }
        }

        Ty::Tuple { fields } => {
            let Json::Array(items) = val else {
                panic!("expected JSON array for tuple, got {val:?}");
            };
            for (f, it) in fields.iter().zip(items) {
                unsafe { run(dst.add(f.offset), &f.ty, it) };
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
                std::ptr::without_provenance_mut(ealign)
            } else {
                let layout =
                    Layout::from_size_align(esz * n, ealign).unwrap();
                let p = unsafe { alloc(layout) };
                assert!(!p.is_null(), "allocation failed");
                for (i, item) in items.iter().enumerate() {
                    unsafe { run(p.add(i * esz), elem, item) };
                }
                p
            };
            unsafe { write_seq(dst, seq, base, n, n) };
        }

        Ty::Unit => { /* zero-sized: consume the JSON value, write nothing */
        }

        Ty::Opt {
            disc_off,
            disc_size,
            none_discr,
            some_discr,
            payload_off,
            size,
            inner,
        } => {
            let ds = *disc_size as usize;
            if matches!(val, Json::Null) {
                unsafe { std::ptr::write_bytes(dst, 0, *size as usize) };
                let nb = none_discr.to_le_bytes();
                unsafe {
                    write_bytes(dst.add(*disc_off), &nb[..ds])
                };
            } else {
                unsafe { run(dst.add(*payload_off), inner, val) };
                if let Some(sd) = some_discr {
                    let sb = sd.to_le_bytes();
                    unsafe {
                        write_bytes(dst.add(*disc_off), &sb[..ds])
                    };
                }
            }
        }

        Ty::Map { .. } => {
            panic!("interp backend doesn't build maps; use the JIT parser")
        }

        Ty::Unknown(what) => panic!("don't know how to write type `{what}`"),
    }
}

/// Allocate `bytes.len()` bytes (matching `Global`, so the reconstructed
/// value's `Drop` frees it correctly with `cap == len`), copy, return ptr.
fn alloc_copy(bytes: &[u8], align: usize) -> *mut u8 {
    if bytes.is_empty() {
        return std::ptr::without_provenance_mut(align.max(1));
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
    unsafe { std::ptr::copy_nonoverlapping(bytes.as_ptr(), dst, bytes.len()) };
}
