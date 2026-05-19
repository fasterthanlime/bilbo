//! Everything that talks to our own debug info: loading the `.dSYM` *once*
//! into a process-global [`Store`], mapping a (de-ASLR'd) program counter to
//! the subprogram it belongs to, finding *which local* a raw pointer aliases
//! by evaluating DWARF locations, and turning a type DIE into a [`Ty`].

use std::collections::HashMap;
use std::sync::{Arc, OnceLock};

use gimli::{EndianSlice, RunTimeEndian};
use object::{Object, ObjectSection};
use tracing::{info, warn};

use crate::plan::{FieldTy, SeqLayout, Ty};
use crate::platform::dwarf_regs;

pub type Reader = EndianSlice<'static, RunTimeEndian>;
pub type Dwarf = gimli::Dwarf<Reader>;
pub type Unit = gimli::Unit<Reader>;
pub type Off = gimli::UnitOffset<usize>;

/// Our own parsed DWARF, alive for the rest of the process. Built once; the
/// hot path never touches the filesystem again.
pub struct Store {
    pub dwarf: Dwarf,
    pub units: Vec<Unit>,
}

static STORE: OnceLock<Store> = OnceLock::new();

/// Parse our `.dSYM` once and keep it forever. The mapped file and the
/// section buffers are intentionally leaked to `'static` — this is a
/// process-lifetime cache, so there is nothing to free.
pub fn store() -> &'static Store {
    STORE.get_or_init(|| {
        // The object that carries our DWARF is platform-specific (macOS:
        // the Mach-O inside the `.dSYM`; Linux: the executable itself, with
        // debuginfo embedded). The backend maps it and leaks it `'static`.
        let bytes = crate::platform::dwarf_bytes();
        let object =
            object::File::parse(bytes).expect("parse debug object");
        let endian = if object.is_little_endian() {
            RunTimeEndian::Little
        } else {
            RunTimeEndian::Big
        };

        let mut sections: HashMap<String, &'static [u8]> = HashMap::new();
        for section in object.sections() {
            if let Ok(name) = section.name()
                && let Ok(data) = section.uncompressed_data()
            {
                let leaked: &'static [u8] =
                    Box::leak(data.into_owned().into_boxed_slice());
                sections.insert(name.to_string(), leaked);
            }
        }
        let load = |id: gimli::SectionId| -> Result<Reader, gimli::Error> {
            let elf = id.name(); // ".debug_info"
            let macho = format!("__{}", &elf[1..]); // "__debug_info"
            let macho = macho.get(..16).unwrap_or(macho.as_str());
            let data = sections
                .get(elf)
                .or_else(|| sections.get(macho))
                .copied()
                .unwrap_or(&[][..]);
            Ok(EndianSlice::new(data, endian))
        };
        let dwarf = gimli::Dwarf::load(load).expect("load DWARF");

        let mut units = Vec::new();
        let mut headers = dwarf.units();
        while let Some(h) = headers.next().expect("unit header") {
            units.push(dwarf.unit(h).expect("parse unit"));
        }
        info!("loaded {} compilation unit(s)", units.len());
        Store { dwarf, units }
    })
}

// ---------------------------------------------------------------------------
// Low-level DIE helpers
// ---------------------------------------------------------------------------

fn child_offsets(unit: &Unit, parent: Option<Off>) -> Vec<Off> {
    let mut tree = unit.entries_tree(parent).expect("entries_tree");
    let root = tree.root().expect("tree root");
    let mut kids = Vec::new();
    let mut it = root.children();
    while let Some(child) = it.next().expect("child") {
        kids.push(child.entry().offset());
    }
    kids
}

/// Inlined / specialized DIEs carry their name and type on an abstract
/// instance pointed to by `DW_AT_abstract_origin` (or `DW_AT_specification`).
fn origin(unit: &Unit, off: Off) -> Option<Off> {
    let entry = unit.entry(off).ok()?;
    let v = entry
        .attr_value(gimli::DW_AT_abstract_origin)
        .or_else(|| entry.attr_value(gimli::DW_AT_specification))?;
    match v {
        gimli::AttributeValue::UnitRef(o) => Some(o),
        _ => None,
    }
}

fn die_name(dwarf: &Dwarf, unit: &Unit, off: Off) -> Option<String> {
    let entry = unit.entry(off).ok()?;
    if let Some(attr) = entry.attr_value(gimli::DW_AT_name) {
        return Some(
            dwarf
                .attr_string(unit, attr)
                .ok()?
                .to_string_lossy()
                .into_owned(),
        );
    }
    die_name(dwarf, unit, origin(unit, off)?)
}

fn tag(unit: &Unit, off: Off) -> gimli::DwTag {
    unit.entry(off).expect("entry").tag()
}

fn udata(unit: &Unit, off: Off, attr: gimli::DwAt) -> Option<u64> {
    match unit.entry(off).ok()?.attr_value(attr)? {
        gimli::AttributeValue::Udata(n) => Some(n),
        gimli::AttributeValue::Sdata(n) => Some(n as u64),
        _ => None,
    }
}

fn type_ref(unit: &Unit, off: Off) -> Option<Off> {
    if let Some(gimli::AttributeValue::UnitRef(o)) =
        unit.entry(off).ok()?.attr_value(gimli::DW_AT_type)
    {
        return Some(o);
    }
    // An inlined variable's type lives on its abstract origin.
    type_ref(unit, origin(unit, off)?)
}

// ---------------------------------------------------------------------------
// PC -> subprogram
// ---------------------------------------------------------------------------

/// Find the `DW_TAG_subprogram` whose machine-code range contains `pc`
/// (a link-time address). Returns `(unit index, DIE offset)`.
pub fn subprogram_at(
    dwarf: &Dwarf,
    units: &[Unit],
    pc: u64,
) -> Option<(usize, Off)> {
    for (ui, unit) in units.iter().enumerate() {
        let mut entries = unit.entries();
        while let Some(entry) = entries.next_dfs().expect("next_dfs") {
            if entry.tag() != gimli::DW_TAG_subprogram {
                continue;
            }
            let off = entry.offset();
            let mut ranges =
                dwarf.die_ranges(unit, entry).expect("die_ranges");
            while let Some(r) = ranges.next().expect("range") {
                if pc >= r.begin && pc < r.end {
                    let name =
                        die_name(dwarf, unit, off).unwrap_or_default();
                    info!(
                        "pc {pc:#x} is in subprogram `{name}` \
                         (range {:#x}..{:#x})",
                        r.begin, r.end
                    );
                    return Some((ui, off));
                }
            }
        }
    }
    None
}

// ---------------------------------------------------------------------------
// Which local does `ptr` alias?
// ---------------------------------------------------------------------------

/// Walk the subprogram's locals (descending lexical blocks), evaluate each
/// one's `DW_OP_fbreg` location against the caller's `cfa`, and return the
/// `(name, type DIE)` of the variable whose address equals `ptr`.
/// Debug helper: the `DW_AT_name` of a subprogram DIE.
pub fn subprogram_name(dwarf: &Dwarf, unit: &Unit, sp: Off) -> String {
    die_name(dwarf, unit, sp).unwrap_or_else(|| "<anon>".into())
}

pub fn local_at_address(
    dwarf: &Dwarf,
    unit: &Unit,
    sp: Off,
    caller_fp: u64,
    caller_sp: u64,
    pc: u64,
    ptr: u64,
) -> Option<(String, Off)> {
    // `DW_OP_fbreg` is relative to the subprogram's frame base. rustc most
    // often emits `DW_AT_frame_base = DW_OP_call_frame_cfa`; it can also be
    // the bare frame-pointer register (`DW_OP_reg<fp>`) or that register
    // plus an offset (`DW_OP_breg<fp>`). The frame-pointer register number
    // is architecture-specific (see `platform::dwarf_regs`).
    let fb_expr = match unit.entry(sp).ok()?.attr_value(gimli::DW_AT_frame_base)
    {
        Some(gimli::AttributeValue::Exprloc(e)) => e.0.slice().to_vec(),
        other => {
            warn!("unexpected frame_base: {other:?}");
            return None;
        }
    };
    let frame_base = match fb_expr.first() {
        // DW_OP_call_frame_cfa == caller SP at the call site.
        Some(0x9c) => caller_sp,
        // The frame-pointer register itself (`DW_OP_reg<fp>`).
        Some(&dwarf_regs::OP_REG_FP) => caller_fp,
        // Frame-pointer register + SLEB128 (`DW_OP_breg<fp>`).
        Some(&dwarf_regs::OP_BREG_FP) => {
            caller_fp.wrapping_add(sleb128(&fb_expr[1..]).0 as u64)
        }
        _ => {
            warn!("unhandled frame_base expr {fb_expr:02x?}");
            return None;
        }
    };
    let regs = Regs {
        fp: caller_fp,
        sp: caller_sp,
        frame_base,
    };
    info!(
        "frame_base {fb_expr:02x?} -> {frame_base:#x}; \
         caller_fp={caller_fp:#x} caller_sp={caller_sp:#x}; \
         hunting ptr {ptr:#x}"
    );

    let mut hunt = Hunt {
        dwarf,
        unit,
        regs,
        pc,
        ptr,
        seen: Vec::new(),
        fallback: None,
        mu: Vec::new(),
    };
    // Optimized code reuses stack slots: several locals can share `ptr`'s
    // address in disjoint live ranges. The API contract is that the caller
    // passes `&mut MaybeUninit<T>`, so prefer a `MaybeUninit<…>` local;
    // only fall back to another same-address local if none is found.
    let found = hunt
        .scan(sp, 0)
        .or_else(|| hunt.fallback.take())
        .or_else(|| hunt.sole_maybe_uninit());
    if found.is_none() {
        warn!("no local matched ptr {ptr:#x}. locals seen:");
        for (n, expr, addr) in &hunt.seen {
            warn!("  `{n}` loc {expr} -> {addr:x?}");
        }
    }
    found
}

/// The register values a DWARF location expression may reference. We only
/// know the few that matter for stack locals.
#[derive(Clone, Copy)]
struct Regs {
    fp: u64,         // frame pointer (x29 / rbp)
    sp: u64,         // stack pointer at the call site (x31 / rsp)
    frame_base: u64, // resolved DW_AT_frame_base
}

/// The walk that hunts for the local whose address equals `ptr`. Bundled
/// into a struct so the recursion carries one `&mut self`, not eight args.
struct Hunt<'a> {
    dwarf: &'a Dwarf,
    unit: &'a Unit,
    regs: Regs,
    pc: u64,
    ptr: u64,
    seen: Vec<(String, String, Option<u64>)>,
    /// First same-address local that isn't a `MaybeUninit<…>`.
    fallback: Option<(String, Off)>,
    /// Every `MaybeUninit<…>` local in the frame, by `(var name, type
    /// name, type DIE)` — even ones the optimizer left with no usable
    /// location. If nothing address-matches but exactly one *type* of
    /// `MaybeUninit` local exists, the API contract (`from_json` is
    /// handed `&mut MaybeUninit<T>`) says that's our target.
    mu: Vec<(String, String, Off)>,
}

impl Hunt<'_> {
    fn scan(&mut self, parent: Off, depth: u32) -> Option<(String, Off)> {
        if depth > 32 {
            return None;
        }
        for off in child_offsets(self.unit, Some(parent)) {
            let t = tag(self.unit, off);
            // Closures get inlined into their caller (e.g. criterion's
            // measurement loop), so the local we want can be nested in a
            // lexical block *or* an inlined subroutine.
            if t == gimli::DW_TAG_lexical_block
                || t == gimli::DW_TAG_inlined_subroutine
            {
                if let Some(hit) = self.scan(off, depth + 1) {
                    return Some(hit);
                }
                continue;
            }
            if t != gimli::DW_TAG_variable
                && t != gimli::DW_TAG_formal_parameter
            {
                continue;
            }
            let name =
                die_name(self.dwarf, self.unit, off).unwrap_or_default();
            let ty = type_ref(self.unit, off);
            let tn = ty
                .map(|t| {
                    die_name(self.dwarf, self.unit, t).unwrap_or_default()
                })
                .unwrap_or_default();
            let is_mu = tn.starts_with("MaybeUninit<");

            match var_location(self.dwarf, self.unit, off, self.pc) {
                Some(expr) => {
                    let addr = eval_addr(&expr, self.regs);
                    self.seen.push((
                        name.clone(),
                        format!("{expr:02x?}"),
                        addr,
                    ));
                    if addr == Some(self.ptr) && let Some(ty) = ty {
                        if is_mu {
                            info!(
                                "local `{name}` : {tn} is at ptr; \
                                 that's our target"
                            );
                            return Some((name, ty));
                        }
                        if self.fallback.is_none() {
                            self.fallback = Some((name.clone(), ty));
                        }
                    }
                }
                // The optimizer can elide a local's storage entirely
                // (e.g. a tiny fn that returns its `MaybeUninit<T>` by
                // value: NRVO forwards the caller's sret slot, no local
                // location is emitted). The *type* DIE survives, which
                // is all we need.
                None => {
                    self.seen.push((
                        name.clone(),
                        "<no location>".into(),
                        None,
                    ));
                }
            }
            if is_mu && let Some(ty) = ty {
                self.mu.push((name.clone(), tn, ty));
            }
        }
        None
    }

    /// Last resort when no local's *address* matched `ptr` (its storage
    /// was optimized out). If the frame has exactly one *type* of
    /// `MaybeUninit<…>` local, the API contract makes it unambiguous.
    fn sole_maybe_uninit(&self) -> Option<(String, Off)> {
        let distinct: std::collections::HashSet<&str> =
            self.mu.iter().map(|(_, tn, _)| tn.as_str()).collect();
        match distinct.len() {
            1 => {
                let (vn, tn, ty) = &self.mu[0];
                warn!(
                    "no address matched; the frame's only MaybeUninit \
                     local is `{vn}` : {tn} — using it (API contract)"
                );
                Some((vn.clone(), *ty))
            }
            0 => None,
            _ => {
                warn!(
                    "no address matched and >1 MaybeUninit types in \
                     frame ({distinct:?}); refusing to guess"
                );
                None
            }
        }
    }
}

/// The location expression bytes for a variable at program counter `pc`.
/// Handles a plain `Exprloc` and a location list (rustc uses a loclist for
/// e.g. a `MaybeUninit` that lives across a call).
fn var_location(
    dwarf: &Dwarf,
    unit: &Unit,
    off: Off,
    pc: u64,
) -> Option<Vec<u8>> {
    let loc = unit.entry(off).ok()?.attr_value(gimli::DW_AT_location)?;
    if let gimli::AttributeValue::Exprloc(expr) = loc {
        return Some(expr.0.slice().to_vec());
    }
    let mut it = dwarf.attr_locations(unit, loc).ok()??;
    while let Some(entry) = it.next().expect("loclist entry") {
        if pc >= entry.range.begin && pc < entry.range.end {
            return Some(entry.data.0.slice().to_vec());
        }
    }
    None
}

/// Evaluate the address-producing subset of DWARF location expressions rustc
/// emits for stack locals: `fbreg`, and `breg<fp>` / `breg<sp>` for the
/// frame and stack pointers (register numbers are architecture-specific —
/// see `platform::dwarf_regs`). A bare `reg<n>` means the value lives in a
/// register (no address); we can't help with that and return `None`.
fn eval_addr(bytes: &[u8], r: Regs) -> Option<u64> {
    let op = *bytes.first()?;
    let (base, rest) = match op {
        0x91 => (r.frame_base, &bytes[1..]), // DW_OP_fbreg
        0x9c => return Some(r.sp),           // DW_OP_call_frame_cfa
        dwarf_regs::OP_BREG_FP => (r.fp, &bytes[1..]), // breg<fp>
        dwarf_regs::OP_BREG_SP => (r.sp, &bytes[1..]), // breg<sp>
        _ => return None,
    };
    let (offset, used) = sleb128(rest);
    // A trailing DW_OP_deref etc. would mean "the value at this address",
    // not the address itself — not what we want for `&mut local`.
    if rest.len() != used {
        return None;
    }
    Some(base.wrapping_add(offset as u64))
}

fn sleb128(b: &[u8]) -> (i64, usize) {
    let mut result: i64 = 0;
    let mut shift = 0;
    let mut i = 0;
    loop {
        let byte = b[i];
        i += 1;
        result |= ((byte & 0x7f) as i64) << shift;
        shift += 7;
        if byte & 0x80 == 0 {
            if shift < 64 && (byte & 0x40) != 0 {
                result |= -1i64 << shift;
            }
            return (result, i);
        }
    }
}

// ---------------------------------------------------------------------------
// Type DIE -> Ty
// ---------------------------------------------------------------------------

/// Follow typedef / const / volatile / restrict wrappers to the real type.
fn strip(unit: &Unit, mut off: Off) -> Off {
    loop {
        match tag(unit, off) {
            gimli::DW_TAG_typedef
            | gimli::DW_TAG_const_type
            | gimli::DW_TAG_volatile_type
            | gimli::DW_TAG_restrict_type => match type_ref(unit, off) {
                Some(inner) => off = inner,
                None => return off,
            },
            _ => return off,
        }
    }
}

thread_local! {
    /// Stripped DIE offsets *currently being classified*, each mapped to a
    /// freshly-minted [`RecCell`]. When the body of a type reaches the type
    /// itself, `classify` re-enters with an ancestor's offset still present
    /// — that's a cycle. We hand back `Ty::Ref(cell)` immediately, and once
    /// the ancestor's body finishes we tie the knot via `cell.0.set(..)`.
    static MEMO: std::cell::RefCell<
        HashMap<(usize, usize), crate::plan::RecCell>,
    > = std::cell::RefCell::new(HashMap::new());
}

/// Removes our in-flight memo entry on scope exit (insert/remove are
/// strictly LIFO because `classify` is a depth-first walk). Keyed by
/// `(unit index, DIE offset)` because, after stub→definition redirect,
/// a single classify can span several CUs.
struct MemoGuard((usize, usize));
impl Drop for MemoGuard {
    fn drop(&mut self) {
        MEMO.with(|m| {
            m.borrow_mut().remove(&self.0);
        });
    }
}

struct IdxFrame {
    depth: isize,
    off: usize,
    decl: bool,
    name: Option<String>,
    has_child: bool,
}

fn idx_close(
    f: IdxFrame,
    ui: usize,
    map: &mut HashMap<String, (usize, usize)>,
) {
    if f.has_child && !f.decl && let Some(n) = f.name {
        map.entry(n).or_insert((ui, f.off));
    }
}

/// Every type is emitted in many CUs: the CU that needs its layout emits
/// a *complete* DIE (members present); CUs that merely mention it emit a
/// *declaration stub* (`DW_AT_declaration`, or simply no members). Which
/// one a given caller's frame chains into is luck of the draw — and a
/// stub makes `seq_layout`/`structure` fail ("could not locate
/// ptr/cap/len"). So index every *complete* struct/union definition by
/// name, exactly once, and redirect any stub we meet to it. Rust's
/// monomorphized type names carry full generic args, so the name alone
/// identifies the layout.
fn def_index() -> &'static HashMap<String, (usize, usize)> {
    static IDX: OnceLock<HashMap<String, (usize, usize)>> = OnceLock::new();
    IDX.get_or_init(|| {
        let st = store();
        let mut map: HashMap<String, (usize, usize)> = HashMap::new();
        for (ui, unit) in st.units.iter().enumerate() {
            let mut cur = unit.entries();
            let mut stack: Vec<IdxFrame> = Vec::new();
            while let Some(entry) = cur.next_dfs().ok().flatten() {
                // Drain everything off `entry` first; `cur.depth()` /
                // `cur.offset()` need `cur` back (the `&Entry` borrows it).
                let etag = entry.tag();
                let is_su = matches!(
                    etag,
                    gimli::DW_TAG_structure_type
                        | gimli::DW_TAG_union_type
                );
                let decl = matches!(
                    entry.attr_value(gimli::DW_AT_declaration),
                    Some(gimli::AttributeValue::Flag(true))
                );
                let name = if is_su {
                    entry
                        .attr_value(gimli::DW_AT_name)
                        .and_then(|v| st.dwarf.attr_string(unit, v).ok())
                        .map(|s| s.to_string_lossy().into_owned())
                } else {
                    None
                };
                let depth = cur.depth();
                let off = cur.offset().0;
                // Close every frame whose subtree we've now left.
                loop {
                    let pop = match stack.last() {
                        Some(t) => depth <= t.depth,
                        None => false,
                    };
                    if !pop {
                        break;
                    }
                    let f = stack.pop().unwrap();
                    idx_close(f, ui, &mut map);
                }
                if is_su {
                    stack.push(IdxFrame {
                        depth,
                        off,
                        decl,
                        name,
                        has_child: false,
                    });
                } else if matches!(
                    etag,
                    gimli::DW_TAG_member | gimli::DW_TAG_variant_part
                ) && let Some(top) = stack.last_mut()
                    && depth == top.depth + 1
                {
                    top.has_child = true;
                }
            }
            while let Some(f) = stack.pop() {
                idx_close(f, ui, &mut map);
            }
        }
        info!(
            "def index: {} complete struct/union definitions across {} CUs",
            map.len(),
            st.units.len()
        );
        map
    })
}

fn unit_idx(unit: &Unit) -> usize {
    store()
        .units
        .iter()
        .position(|u| std::ptr::eq(u, unit))
        .expect("unit comes from the global store")
}

/// A struct/union DIE we can't read a layout from: a `DW_AT_declaration`
/// stub, or one with a real `byte_size` but no members.
fn is_stub(unit: &Unit, off: Off) -> bool {
    if !matches!(
        tag(unit, off),
        gimli::DW_TAG_structure_type | gimli::DW_TAG_union_type
    ) {
        return false;
    }
    let decl = matches!(
        unit.entry(off)
            .ok()
            .and_then(|e| e.attr_value(gimli::DW_AT_declaration)),
        Some(gimli::AttributeValue::Flag(true))
    );
    if decl {
        return true;
    }
    if udata(unit, off, gimli::DW_AT_byte_size).unwrap_or(0) == 0 {
        return false; // genuine ZST / unit struct is complete
    }
    !child_offsets(unit, Some(off)).into_iter().any(|c| {
        matches!(
            tag(unit, c),
            gimli::DW_TAG_member | gimli::DW_TAG_variant_part
        )
    })
}

/// If `(unit, off)` is a stub, swap in the complete definition (often in
/// another CU). Returns the `(unit index, unit, off)` to use from here.
fn def_site(unit: &Unit, off: Off) -> (usize, &'static Unit, Off) {
    let st = store();
    if is_stub(unit, off)
        && let Some(name) = die_name(&st.dwarf, unit, off)
        && let Some(&(ui, o)) = def_index().get(&name)
    {
        return (ui, &st.units[ui], gimli::UnitOffset(o));
    }
    let ui = unit_idx(unit);
    (ui, &st.units[ui], off)
}

pub fn classify(dwarf: &Dwarf, unit: &Unit, off: Off) -> Ty {
    let off = strip(unit, off);
    // The local's type DIE (or any nested one) may be an incomplete stub
    // in this CU — redirect to the complete definition before reading it.
    let (ui, unit, off) = def_site(unit, off);
    let t = tag(unit, off);
    let name = die_name(dwarf, unit, off).unwrap_or_default();

    // Layout-transparent wrappers: same bytes as `T` at offset 0. They have
    // no identity of their own, so they don't participate in the memo.
    if name.starts_with("MaybeUninit<")
        || name.starts_with("ManuallyDrop<")
        || name.starts_with("UnsafeCell<")
    {
        let inner = template_param(unit, off)
            .unwrap_or_else(|| panic!("`{name}` has no T parameter"));
        return classify(dwarf, unit, inner);
    }

    // Cycle? An ancestor with this DIE is still being built — hand the
    // back-edge its (not-yet-filled) cell. Keyed by `(unit, off)` since
    // the redirect above can move us between CUs.
    let key = (ui, off.0);
    if let Some(cell) = MEMO.with(|m| m.borrow().get(&key).cloned()) {
        return Ty::Ref(cell);
    }
    let cell = crate::plan::RecCell::new();
    MEMO.with(|m| {
        m.borrow_mut().insert(key, cell.clone());
    });
    let _g = MemoGuard(key);

    let body = match t {
        gimli::DW_TAG_base_type => base_type(unit, off, &name),
        gimli::DW_TAG_pointer_type => boxed(dwarf, unit, off, &name),
        gimli::DW_TAG_structure_type => structure(dwarf, unit, off, &name),
        gimli::DW_TAG_enumeration_type => Ty::Unknown(format!("enum {name}")),
        _ => Ty::Unknown(name),
    };

    // The local + the MEMO entry hold two `Arc`s; anything beyond that is a
    // `Ty::Ref` the body handed out, i.e. this type really is recursive —
    // tie the knot. Non-recursive types stay inline (cell is just dropped).
    if Arc::strong_count(&cell.0) > 2 {
        let _ = cell.0.set(body.clone());
    }
    body
}

fn base_type(unit: &Unit, off: Off, name: &str) -> Ty {
    if name == "()" {
        return Ty::Unit;
    }
    let size = udata(unit, off, gimli::DW_AT_byte_size).unwrap_or(0) as u8;
    let enc = match unit
        .entry(off)
        .ok()
        .and_then(|e| e.attr_value(gimli::DW_AT_encoding))
    {
        Some(gimli::AttributeValue::Encoding(e)) => e,
        _ => gimli::DwAte(0),
    };
    match enc {
        gimli::DW_ATE_boolean => Ty::Bool,
        gimli::DW_ATE_float if size == 4 => Ty::F32,
        gimli::DW_ATE_float => Ty::F64,
        gimli::DW_ATE_unsigned_char if name == "char" => Ty::Char,
        gimli::DW_ATE_UTF => Ty::Char,
        gimli::DW_ATE_signed | gimli::DW_ATE_signed_char => Ty::I(size),
        gimli::DW_ATE_unsigned | gimli::DW_ATE_unsigned_char => Ty::U(size),
        _ => Ty::Unknown(name.to_string()),
    }
}

/// `Box<T>` (and any `DW_TAG_pointer_type` we treat as owned): allocate a
/// `T`, parse into it, store the 8-byte pointer. `size`/`align` come from
/// the pointee DIE so the JIT can `alloc`/`dealloc` it correctly.
fn boxed(dwarf: &Dwarf, unit: &Unit, off: Off, name: &str) -> Ty {
    let Some(pointee) = type_ref(unit, off) else {
        return Ty::Unknown(format!("*{name}"));
    };
    let inner = classify(dwarf, unit, pointee);
    let pointee = strip(unit, pointee);
    let size = udata(unit, pointee, gimli::DW_AT_byte_size).unwrap_or(0);
    let align = udata(unit, pointee, gimli::DW_AT_alignment)
        .unwrap_or_else(|| size.max(1).next_power_of_two());
    Ty::Boxed {
        inner: Box::new(inner),
        size,
        align,
    }
}

fn structure(dwarf: &Dwarf, unit: &Unit, off: Off, name: &str) -> Ty {
    if name == "String" {
        return Ty::Str(seq_layout(dwarf, unit, off));
    }
    if name == "&str" {
        let (ptr_off, len_off) = str_ref_offsets(dwarf, unit, off);
        return Ty::StrRef { ptr_off, len_off };
    }
    if name.starts_with("Vec<") {
        let elem_off = template_param(unit, off).expect("Vec<T> param");
        let elem = classify(dwarf, unit, elem_off);
        let elem_off = strip(unit, elem_off);
        let elem_size =
            udata(unit, elem_off, gimli::DW_AT_byte_size).unwrap_or(1);
        let elem_align = udata(unit, elem_off, gimli::DW_AT_alignment)
            .unwrap_or_else(|| elem_size.max(1).next_power_of_two());
        return Ty::Vec {
            elem: Box::new(elem),
            elem_size,
            elem_align,
            seq: seq_layout(dwarf, unit, off),
        };
    }
    if name.starts_with("BTreeMap<") {
        // BTreeMap<String, V, Global> — key is always String here.
        let k_off = template_named(dwarf, unit, off, "K")
            .expect("BTreeMap<K,_> param");
        let v_off = template_named(dwarf, unit, off, "V")
            .expect("BTreeMap<_,V> param");
        let v_name = die_name(dwarf, unit, v_off).unwrap_or_default();
        let key = classify(dwarf, unit, k_off);
        let key_size =
            udata(unit, strip(unit, k_off), gimli::DW_AT_byte_size)
                .unwrap_or(24);
        let val = classify(dwarf, unit, v_off);
        let val_size =
            udata(unit, strip(unit, v_off), gimli::DW_AT_byte_size)
                .unwrap_or(1);
        let new_at = resolve_tramp("map_new_at", &v_name, false)
            .unwrap_or_else(|| panic!("no map_new_at trampoline at all"));
        let insert = resolve_tramp("map_insert", &v_name, true)
            .unwrap_or_else(|| {
                panic!(
                    "no map_insert trampoline for V=`{v_name}` \
                     (call bilbo_json::tramp::force::<{v_name}>())"
                )
            });
        return Ty::Map {
            key: Box::new(key),
            key_size,
            val: Box::new(val),
            val_size,
            new_at,
            insert,
        };
    }
    // Niche-optimized enum (e.g. `Option<String>`): a `variant_part`.
    if let Some(vp) = child_with_tag(unit, off, gimli::DW_TAG_variant_part) {
        return parse_option(dwarf, unit, off, vp, name);
    }

    // A plain aggregate: recurse into its members.
    let mut fields = Vec::new();
    for moff in child_offsets(unit, Some(off)) {
        if tag(unit, moff) != gimli::DW_TAG_member {
            continue;
        }
        let fname = die_name(dwarf, unit, moff).unwrap_or_default();
        let foff =
            udata(unit, moff, gimli::DW_AT_data_member_location).unwrap_or(0)
                as usize;
        let fty = type_ref(unit, moff)
            .map(|t| classify(dwarf, unit, t))
            .unwrap_or(Ty::Unknown("?".into()));
        fields.push(FieldTy {
            name: fname,
            offset: foff,
            ty: fty,
        });
    }
    // A tuple / tuple-struct: members are `__0`, `__1`, … In JSON these
    // are positional arrays, not objects.
    if !fields.is_empty()
        && fields.iter().all(|f| {
            f.name.strip_prefix("__").is_some_and(|n| {
                !n.is_empty() && n.bytes().all(|b| b.is_ascii_digit())
            })
        })
    {
        let mut fields = fields;
        fields.sort_by_key(|f| {
            f.name[2..].parse::<u32>().unwrap_or(u32::MAX)
        });
        return Ty::Tuple { fields };
    }
    Ty::Struct {
        name: name.to_string(),
        fields,
    }
}

fn template_param(unit: &Unit, off: Off) -> Option<Off> {
    for c in child_offsets(unit, Some(off)) {
        if tag(unit, c) == gimli::DW_TAG_template_type_parameter {
            return type_ref(unit, c);
        }
    }
    None
}

/// The `DW_TAG_template_type_parameter` named `want` (e.g. "V").
fn template_named(
    dwarf: &Dwarf,
    unit: &Unit,
    off: Off,
    want: &str,
) -> Option<Off> {
    for c in child_offsets(unit, Some(off)) {
        if tag(unit, c) == gimli::DW_TAG_template_type_parameter
            && die_name(dwarf, unit, c).as_deref() == Some(want)
        {
            return type_ref(unit, c);
        }
    }
    None
}

fn child_with_tag(unit: &Unit, off: Off, t: gimli::DwTag) -> Option<Off> {
    child_offsets(unit, Some(off))
        .into_iter()
        .find(|&c| tag(unit, c) == t)
}

fn attr_udata(unit: &Unit, off: Off, at: gimli::DwAt) -> Option<u64> {
    let e = unit.entry(off).ok()?;
    e.attr(at)?.udata_value()
}

/// Find `bilbo_json::tramp::{fn_prefix}::<V>`'s runtime address: scan every
/// unit for a subprogram whose name starts with `fn_prefix` and whose `V`
/// template parameter resolves to a type named `v_name`. DWARF gives the
/// link-time `DW_AT_low_pc`; add the ASLR slide to get something callable.
/// Normalize a type name so the two spellings DWARF uses agree: the
/// trampoline's monomorphized fn name has the fully-qualified value type
/// (`alloc::string::String`), while a type DIE's name is the short form
/// (`String`). Strip the module path on the head (before any `<`).
fn norm_ty(s: &str) -> String {
    let (head, rest) = match s.find('<') {
        Some(i) => (&s[..i], &s[i..]),
        None => (s, ""),
    };
    let leaf = head.rsplit("::").next().unwrap_or(head);
    format!("{leaf}{rest}")
}

fn resolve_tramp(
    fn_prefix: &str,
    v_name: &str,
    match_v: bool,
) -> Option<u64> {
    let want = norm_ty(v_name);
    let store = store();
    for unit in &store.units {
        let mut entries = unit.entries();
        while let Some(entry) = entries.next_dfs().ok().flatten() {
            if entry.tag() != gimli::DW_TAG_subprogram {
                continue;
            }
            let off = entry.offset();
            let name = die_name(&store.dwarf, unit, off).unwrap_or_default();
            // e.g. "map_new_at<alloc::string::String>"
            if !name.starts_with(fn_prefix) {
                continue;
            }
            // `map_new_at::<V>` is V-independent (`BTreeMap::new()` is the
            // same code for every V — ICF folds them into one symbol), so
            // any instance is correct. `map_insert::<V>` is not: match V.
            if match_v {
                let Some(lt) = name.find('<') else { continue };
                let inner = &name[lt + 1..name.len().saturating_sub(1)];
                if norm_ty(inner) != want {
                    continue;
                }
            }
            if let Some(gimli::AttributeValue::Addr(lo)) =
                unit.entry(off).ok()?.attr_value(gimli::DW_AT_low_pc)
            {
                return Some(lo + crate::frame::image_slide());
            }
        }
    }
    None
}

/// Parse a two-variant `Option<T>` from a `variant_part`, handling both
/// the niche encoding (Some has no `discr_value`, payload at offset 0,
/// discriminant overlaps it) and the tagged encoding (separate tag, Some
/// has a `discr_value`, payload after the tag).
fn parse_option(
    dwarf: &Dwarf,
    unit: &Unit,
    off: Off,
    vp: Off,
    name: &str,
) -> Ty {
    let size = udata(unit, off, gimli::DW_AT_byte_size).unwrap_or(0);
    let unk = || Ty::Unknown(format!("enum {name}"));

    let Some(gimli::AttributeValue::UnitRef(disc_m)) = unit
        .entry(vp)
        .ok()
        .and_then(|e| e.attr_value(gimli::DW_AT_discr))
    else {
        return unk();
    };
    let disc_off =
        udata(unit, disc_m, gimli::DW_AT_data_member_location).unwrap_or(0)
            as usize;
    let disc_size = type_ref(unit, disc_m)
        .map(|t| strip(unit, t))
        .and_then(|t| udata(unit, t, gimli::DW_AT_byte_size))
        .unwrap_or(8) as u8;

    let mut none_discr: Option<u128> = None;
    let mut some: Option<(Option<u128>, usize, Ty)> = None; // (discr, off, T)
    for v in child_offsets(unit, Some(vp)) {
        if tag(unit, v) != gimli::DW_TAG_variant {
            continue;
        }
        let dv = attr_udata(unit, v, gimli::DW_AT_discr_value);
        let Some(pm) = child_with_tag(unit, v, gimli::DW_TAG_member) else {
            continue;
        };
        let pm_off =
            udata(unit, pm, gimli::DW_AT_data_member_location).unwrap_or(0)
                as usize;
        let Some(pstruct) = type_ref(unit, pm) else { continue };
        let pstruct = strip(unit, pstruct);
        let f0 = child_offsets(unit, Some(pstruct)).into_iter().find(|&c| {
            tag(unit, c) == gimli::DW_TAG_member
                && die_name(dwarf, unit, c).as_deref() == Some("__0")
        });
        match f0 {
            None => none_discr = dv.map(|x| x as u128),
            Some(f0) => {
                let foff =
                    udata(unit, f0, gimli::DW_AT_data_member_location)
                        .unwrap_or(0) as usize;
                if let Some(t) = type_ref(unit, f0) {
                    some = Some((
                        dv.map(|x| x as u128),
                        pm_off + foff,
                        classify(dwarf, unit, t),
                    ));
                }
            }
        }
    }
    match (none_discr, some) {
        (Some(none_discr), Some((some_discr, payload_off, inner))) => {
            Ty::Opt {
                disc_off,
                disc_size,
                none_discr,
                some_discr,
                payload_off,
                size,
                inner: Box::new(inner),
            }
        }
        _ => unk(),
    }
}

/// Recursively flatten members of a `Vec`/`String` to locate the absolute
/// offsets of the data pointer, the capacity, and the length.
fn seq_layout(dwarf: &Dwarf, unit: &Unit, off: Off) -> SeqLayout {
    let mut s = SeqLayout {
        ptr_off: usize::MAX,
        cap_off: usize::MAX,
        len_off: usize::MAX,
    };
    walk_seq(dwarf, unit, off, 0, &mut s, 0);
    info!(
        "seq layout: ptr@{} cap@{} len@{}",
        s.ptr_off, s.cap_off, s.len_off
    );
    assert!(
        s.ptr_off != usize::MAX
            && s.cap_off != usize::MAX
            && s.len_off != usize::MAX,
        "could not locate ptr/cap/len in DWARF: {s:?}"
    );
    s
}

fn walk_seq(
    dwarf: &Dwarf,
    unit: &Unit,
    off: Off,
    base: usize,
    s: &mut SeqLayout,
    depth: u32,
) {
    if depth > 16 {
        return;
    }
    let off = strip(unit, off);
    // `RawVec`/`RawVecInner`/`Unique`/… are often stubs in the CU we
    // arrived from — follow each to its complete definition.
    let (_ui, unit, off) = def_site(unit, off);
    match tag(unit, off) {
        gimli::DW_TAG_pointer_type
            if s.ptr_off == usize::MAX => {
                s.ptr_off = base;
            }
        gimli::DW_TAG_structure_type | gimli::DW_TAG_union_type => {
            for moff in child_offsets(unit, Some(off)) {
                if tag(unit, moff) != gimli::DW_TAG_member {
                    continue;
                }
                let abs = base
                    + udata(unit, moff, gimli::DW_AT_data_member_location)
                        .unwrap_or(0) as usize;
                match die_name(dwarf, unit, moff).as_deref() {
                    Some("cap" | "capacity") if s.cap_off == usize::MAX => {
                        s.cap_off = abs;
                    }
                    Some("len" | "length") if s.len_off == usize::MAX => {
                        s.len_off = abs;
                    }
                    _ => {
                        if let Some(mt) = type_ref(unit, moff) {
                            walk_seq(dwarf, unit, mt, abs, s, depth + 1);
                        }
                    }
                }
            }
        }
        _ => {}
    }
}

fn str_ref_offsets(dwarf: &Dwarf, unit: &Unit, off: Off) -> (usize, usize) {
    let (mut ptr_off, mut len_off) = (0usize, 0usize);
    for moff in child_offsets(unit, Some(off)) {
        if tag(unit, moff) != gimli::DW_TAG_member {
            continue;
        }
        let abs = udata(unit, moff, gimli::DW_AT_data_member_location)
            .unwrap_or(0) as usize;
        match die_name(dwarf, unit, moff).as_deref() {
            Some("data_ptr") => ptr_off = abs,
            Some("length") => len_off = abs,
            _ => {}
        }
    }
    (ptr_off, len_off)
}

#[cfg(test)]
mod tests {
    use super::sleb128;

    /// Signed LEB128 is how rustc encodes the offset in `DW_OP_fbreg` /
    /// `DW_OP_breg<n>`, so getting the sign extension right is load-bearing
    /// for finding the aliased local on every platform. Vectors hand-checked
    /// against the DWARF spec encoding.
    #[test]
    fn sleb128_known_vectors() {
        // (bytes, value, bytes_consumed)
        let cases: &[(&[u8], i64, usize)] = &[
            (&[0x00], 0, 1),
            (&[0x02], 2, 1),
            (&[0x7f], -1, 1),
            (&[0x7e], -2, 1),
            // -24: a typical `DW_OP_fbreg` offset for a stack local.
            (&[0x68], -24, 1),
            (&[0x80, 0x01], 128, 2),
            (&[0xff, 0x00], 127, 2),
            (&[0x80, 0x7f], -128, 2),
        ];
        for &(bytes, want, want_used) in cases {
            assert_eq!(
                sleb128(bytes),
                (want, want_used),
                "sleb128({bytes:02x?})"
            );
        }
    }

    /// The decoder must report exactly how many bytes it consumed —
    /// `eval_addr` relies on `rest.len() == used` to reject a trailing
    /// `DW_OP_deref`.
    #[test]
    fn sleb128_reports_length_with_trailing_bytes() {
        let (v, used) = sleb128(&[0x68, 0x9c, 0xde]);
        assert_eq!((v, used), (-24, 1));
    }
}

