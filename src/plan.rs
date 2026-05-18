//! The cached, DWARF-free artifact. A [`Ty`] is everything the hot path
//! needs to bind JSON into raw memory: field offsets, primitive sizes, and
//! the exact `ptr`/`cap`/`len` offsets inside `String`/`Vec` (whose word
//! order Rust does not promise — we learned it from DWARF, once).

#[derive(Debug, Clone)]
pub enum Ty {
    Bool,
    Char,
    U(u8), // size in bytes
    I(u8),
    F32,
    F64,
    Struct {
        name: String,
        fields: Vec<FieldTy>,
    },
    /// `alloc::string::String` — UTF-8 bytes behind a `Vec<u8>`.
    Str(SeqLayout),
    /// `alloc::vec::Vec<T>`.
    Vec {
        elem: Box<Ty>,
        elem_size: u64,
        elem_align: u64,
        seq: SeqLayout,
    },
    /// `&str` fat pointer.
    StrRef {
        ptr_off: usize,
        len_off: usize,
    },
    Unknown(String),
}

#[derive(Debug, Clone)]
pub struct FieldTy {
    pub name: String,
    pub offset: usize,
    pub ty: Ty,
}

/// Where the data pointer / capacity / length words sit inside a `Vec`-like
/// value. Their *order* is not guaranteed by Rust — DWARF told us exactly.
#[derive(Debug, Clone, Copy)]
pub struct SeqLayout {
    pub ptr_off: usize,
    pub cap_off: usize,
    pub len_off: usize,
}
