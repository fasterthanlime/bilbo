//! The cached, DWARF-free artifact. A [`Ty`] is everything the hot path
//! needs to bind JSON into raw memory: field offsets, primitive sizes, and
//! the exact `ptr`/`cap`/`len` offsets inside `String`/`Vec` (whose word
//! order Rust does not promise — we learned it from DWARF, once).

use std::sync::{Arc, OnceLock};

/// A back-edge into a recursive type. While classifying a type whose body
/// reaches itself (`Node::next: Option<Box<Node>>`), we hand the inner
/// occurrence one of these *before* the body is finished, then `set()` it
/// once the body is known — tying the knot. Cloning a [`Ty`] only clones
/// the `Arc`, so a cyclic type is finite in memory yet unfolds at runtime
/// exactly as far as the JSON data demands.
#[derive(Clone)]
pub struct RecCell(pub Arc<OnceLock<Ty>>);

impl RecCell {
    pub fn new() -> Self {
        RecCell(Arc::new(OnceLock::new()))
    }
    /// The resolved body (panics if the knot was never tied).
    pub fn get(&self) -> &Ty {
        self.0.get().expect("recursive Ty cell not set")
    }
    /// Pointer identity, used to dedup cells (one JIT function per cell).
    pub fn id(&self) -> *const () {
        Arc::as_ptr(&self.0) as *const ()
    }
}

impl Default for RecCell {
    fn default() -> Self {
        Self::new()
    }
}

impl std::fmt::Debug for RecCell {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        // Never recurse through the cycle when printing.
        write!(f, "Ref(..)")
    }
}

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
    /// A tuple / tuple-struct (members `__0`, `__1`, …): in JSON it's a
    /// *positional array* `[a, b]`, not an object. `fields` are in index
    /// order, each carrying its byte offset.
    Tuple {
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
    /// A zero-sized field (`()`): consume the JSON value, write nothing.
    Unit,
    /// `Box<T>` — an owned heap pointer (DWARF: a pointer to `T`). Allocate
    /// `T`, parse into it, store the pointer.
    Boxed {
        inner: Box<Ty>,
        size: u64,
        align: u64,
    },
    /// An `Option<T>` (one data variant + one empty variant), covering
    /// both encodings DWARF emits:
    ///
    /// * **niche** (`Option<String>`): no separate tag — the discriminant
    ///   field overlaps the payload, `Some` has no `discr_value`, payload
    ///   sits at offset 0. `some_discr` is `None` (writing a valid `T`
    ///   already encodes `Some`).
    /// * **tagged** (`Option<u64>`): a real tag at `disc_off`, payload at
    ///   `payload_off`. `some_discr` is `Some(tag)`.
    Opt {
        disc_off: usize,
        disc_size: u8,
        none_discr: u128,
        /// `Some`'s tag value, or `None` for the niche encoding.
        some_discr: Option<u128>,
        /// Offset of `Some`'s payload within the option.
        payload_off: usize,
        /// Total size of the option (zeroed before writing `None`).
        size: u64,
        inner: Box<Ty>,
    },
    /// `BTreeMap<String, V>`. Built by calling the real std map via
    /// `bilbo-json`'s `tramp` trampolines, resolved from DWARF (their
    /// link-time addresses end up here as `new_at` / `insert`).
    Map {
        key: Box<Ty>,
        key_size: u64,
        val: Box<Ty>,
        val_size: u64,
        /// Runtime addresses of `map_new_at::<V>` / `map_insert::<V>`.
        new_at: u64,
        insert: u64,
    },
    /// A back-edge to an enclosing recursive type (e.g. the `Node` inside
    /// `Node::next: Option<Box<Node>>`). Dereference via [`RecCell::get`];
    /// the JIT compiles one function per cell and emits a call here.
    Ref(RecCell),
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
