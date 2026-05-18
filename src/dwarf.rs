//! Everything that talks to our own debug info: loading the `.dSYM` *once*
//! into a process-global [`Store`], mapping a (de-ASLR'd) program counter to
//! the subprogram it belongs to, finding *which local* a raw pointer aliases
//! by evaluating DWARF locations, and turning a type DIE into a [`Ty`].

use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::OnceLock;

use gimli::{EndianSlice, RunTimeEndian};
use object::{Object, ObjectSection};
use tracing::{info, warn};

use crate::plan::{FieldTy, SeqLayout, Ty};

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
        let path = dsym_path();
        info!("reading our own DWARF from {} (once)", path.display());
        let file = std::fs::File::open(&path).expect("open dSYM");
        let mmap =
            unsafe { memmap2::Mmap::map(&file) }.expect("mmap dSYM");
        let bytes: &'static [u8] = Box::leak(Box::new(mmap));
        let object = object::File::parse(bytes).expect("parse Mach-O");
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

fn dsym_path() -> PathBuf {
    let exe = std::env::current_exe().expect("current_exe");
    let name = exe.file_name().expect("exe name").to_owned();
    let dir = exe.parent().expect("exe dir");
    let mut p = dir.to_path_buf();
    p.push(format!("{}.dSYM", name.to_string_lossy()));
    p.push("Contents/Resources/DWARF");
    // The DWARF binary inside is named after the deps artifact (with a hash),
    // not the final executable, so just take whatever is in there.
    if let Ok(mut entries) = std::fs::read_dir(&p)
        && let Some(Ok(e)) = entries.next()
    {
        return e.path();
    }
    exe // maybe debuginfo is embedded directly
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
pub fn local_at_address(
    dwarf: &Dwarf,
    unit: &Unit,
    sp: Off,
    caller_fp: u64,
    caller_sp: u64,
    pc: u64,
    ptr: u64,
) -> Option<(String, Off)> {
    // `DW_OP_fbreg` is relative to the subprogram's frame base. rustc on
    // AArch64 emits `DW_AT_frame_base = DW_OP_reg29` (the frame pointer).
    let fb_expr = match unit.entry(sp).ok()?.attr_value(gimli::DW_AT_frame_base)
    {
        Some(gimli::AttributeValue::Exprloc(e)) => e.0.slice().to_vec(),
        other => {
            warn!("unexpected frame_base: {other:?}");
            return None;
        }
    };
    let frame_base = match fb_expr.first() {
        Some(0x9c) => caller_sp,  // DW_OP_call_frame_cfa == caller SP at call
        Some(0x6d) => caller_fp,  // DW_OP_reg29 (x29 / fp)
        Some(0x8d) => caller_fp.wrapping_add(sleb128(&fb_expr[1..]).0 as u64),
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
    };
    // Optimized code reuses stack slots: several locals can share `ptr`'s
    // address in disjoint live ranges. The API contract is that the caller
    // passes `&mut MaybeUninit<T>`, so prefer a `MaybeUninit<…>` local;
    // only fall back to another same-address local if none is found.
    let found = hunt.scan(sp, 0).or_else(|| hunt.fallback.take());
    if found.is_none() {
        warn!("no local matched ptr {ptr:#x}. locals seen:");
        for (n, expr, addr) in &hunt.seen {
            warn!("  `{n}` loc {expr} -> {addr:x?}");
        }
    }
    found
}

/// The register values a DWARF location expression may reference. We only
/// know the few that matter for stack locals on AArch64.
#[derive(Clone, Copy)]
struct Regs {
    fp: u64,         // x29
    sp: u64,         // x31 / SP at the call site
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
            let Some(expr) =
                var_location(self.dwarf, self.unit, off, self.pc)
            else {
                continue;
            };
            let addr = eval_addr(&expr, self.regs);
            self.seen.push((name.clone(), format!("{expr:02x?}"), addr));
            if addr == Some(self.ptr) {
                let Some(ty) = type_ref(self.unit, off) else {
                    continue;
                };
                let tn =
                    die_name(self.dwarf, self.unit, ty).unwrap_or_default();
                if tn.starts_with("MaybeUninit<") {
                    info!("local `{name}` : {tn} is at ptr; that's our target");
                    return Some((name, ty));
                }
                if self.fallback.is_none() {
                    self.fallback = Some((name.clone(), ty));
                }
            }
        }
        None
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
/// emits for stack locals: `fbreg`, and `bregN` for the frame pointer (x29)
/// or stack pointer (x31). A bare `regN` means the value lives in a register
/// (no address); we can't help with that and return `None`.
fn eval_addr(bytes: &[u8], r: Regs) -> Option<u64> {
    let op = *bytes.first()?;
    let (base, rest) = match op {
        0x91 => (r.frame_base, &bytes[1..]),       // DW_OP_fbreg
        0x9c => return Some(r.sp),                  // DW_OP_call_frame_cfa
        0x8d => (r.fp, &bytes[1..]),                // DW_OP_breg29 (x29)
        0x8f => (r.sp, &bytes[1..]),                // DW_OP_breg31 (SP)
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

pub fn classify(dwarf: &Dwarf, unit: &Unit, off: Off) -> Ty {
    let off = strip(unit, off);
    let t = tag(unit, off);
    let name = die_name(dwarf, unit, off).unwrap_or_default();

    // Layout-transparent wrappers: same bytes as `T` at offset 0.
    if name.starts_with("MaybeUninit<")
        || name.starts_with("ManuallyDrop<")
        || name.starts_with("UnsafeCell<")
    {
        let inner = template_param(unit, off)
            .unwrap_or_else(|| panic!("`{name}` has no T parameter"));
        return classify(dwarf, unit, inner);
    }

    match t {
        gimli::DW_TAG_base_type => base_type(unit, off, &name),
        gimli::DW_TAG_pointer_type => Ty::Unknown(format!("*{name}")),
        gimli::DW_TAG_structure_type => structure(dwarf, unit, off, &name),
        gimli::DW_TAG_enumeration_type => Ty::Unknown(format!("enum {name}")),
        _ => Ty::Unknown(name),
    }
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
                     (call dwarf_json::tramp::force::<{v_name}>())"
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
        return niche_option(dwarf, unit, off, vp, name);
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

/// Find `crate::tramp::{fn_prefix}::<V>`'s runtime address: scan every
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

/// Parse a niche-optimized two-variant enum (`Option<T>`): one variant has
/// a `DW_AT_discr_value` and an empty payload (`None`), the other has no
/// discr value and a `__0: T` payload (`Some`). The discriminant overlaps
/// the payload (no tag byte).
fn niche_option(
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

    let mut none_val: Option<u128> = None;
    let mut inner: Option<Ty> = None;
    for v in child_offsets(unit, Some(vp)) {
        if tag(unit, v) != gimli::DW_TAG_variant {
            continue;
        }
        let dv = attr_udata(unit, v, gimli::DW_AT_discr_value);
        let Some(pm) = child_with_tag(unit, v, gimli::DW_TAG_member) else {
            continue;
        };
        let Some(pstruct) = type_ref(unit, pm) else { continue };
        let pstruct = strip(unit, pstruct);
        // payload field `__0`, if any
        let f0 = child_offsets(unit, Some(pstruct)).into_iter().find(|&c| {
            tag(unit, c) == gimli::DW_TAG_member
                && die_name(dwarf, unit, c).as_deref() == Some("__0")
        });
        match (dv, f0) {
            (Some(val), None) => none_val = Some(val as u128),
            (_, Some(f0)) => {
                if let Some(t) = type_ref(unit, f0) {
                    inner = Some(classify(dwarf, unit, t));
                }
            }
            _ => {}
        }
    }
    match (none_val, inner) {
        (Some(none_val), Some(inner)) => Ty::NicheOption {
            disc_off,
            disc_size,
            none_val,
            size,
            inner: Box::new(inner),
        },
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
