//! Deserialize JSON into raw memory by reading our *own* DWARF debug info at
//! runtime to recover the type of whatever the caller handed us. This is a
//! very silly idea and it is implemented with great enthusiasm.

use std::borrow::Cow;
use std::collections::HashMap;
use std::path::PathBuf;

use gimli::{EndianSlice, RunTimeEndian};
use object::{Object, ObjectSection};
use tracing::info;

type Reader<'a> = EndianSlice<'a, RunTimeEndian>;
type Dwarf<'a> = gimli::Dwarf<Reader<'a>>;
type Unit<'a> = gimli::Unit<Reader<'a>>;
type UnitOffset = gimli::UnitOffset<usize>;

fn main() {
    tracing_subscriber::fmt()
        .with_max_level(tracing::Level::TRACE)
        .with_target(false)
        .without_time()
        .init();

    #[derive(Debug)]
    struct Endpoint {
        host: String,
        port: u16,
    }

    let mut e: std::mem::MaybeUninit<Endpoint> = std::mem::MaybeUninit::uninit();
    from_json(
        r#"
        {
          "host": "rustweek.org",
          "port": 443,
        }
    "#,
        &mut e as *mut _ as *mut u8,
    );
    // Safety: yolo
    let e = unsafe { e.assume_init() };
    info!("🎉 reconstructed: host={:?} port={}", e.host, e.port);
    info!("   (debug view: {e:#?})");
}

/// Deserialize `s` into `*ptr`, figuring out the destination type from DWARF.
/// Panics if anything goes wrong. Narrates every step because it is absurd.
#[inline(never)]
fn from_json(s: &str, ptr: *mut u8) {
    info!("from_json called; let's go find out what we're writing into");

    let caller = caller_function_name();
    info!("our caller, per the stack trace, is `{caller}`");
    let caller_leaf = caller.rsplit("::").next().unwrap_or(&caller).to_string();

    let dsym = dsym_path();
    info!("reading our own DWARF from {}", dsym.display());
    let file = std::fs::File::open(&dsym).expect("open dSYM");
    let mmap = unsafe { memmap2::Mmap::map(&file) }.expect("mmap dSYM");
    let object = object::File::parse(&*mmap).expect("parse Mach-O");
    let endian = if object.is_little_endian() {
        RunTimeEndian::Little
    } else {
        RunTimeEndian::Big
    };

    // Slurp every section once; gimli will index into these buffers.
    let mut sections: HashMap<String, Cow<[u8]>> = HashMap::new();
    for section in object.sections() {
        if let Ok(name) = section.name()
            && let Ok(data) = section.uncompressed_data() {
                sections.insert(name.to_string(), data);
            }
    }
    let load = |id: gimli::SectionId| -> Result<Reader, gimli::Error> {
        let elf = id.name(); // ".debug_info"
        let macho = format!("__{}", &elf[1..]); // "__debug_info"
        let macho = macho.get(..16).unwrap_or(macho.as_str()); // 16-char cap
        let data = sections
            .get(elf)
            .or_else(|| sections.get(macho))
            .map(|c| c.as_ref())
            .unwrap_or(&[][..]);
        Ok(EndianSlice::new(data, endian))
    };
    let dwarf = gimli::Dwarf::load(load).expect("load DWARF");

    let units: Vec<Unit> = {
        let mut headers = dwarf.units();
        let mut v = Vec::new();
        while let Some(h) = headers.next().expect("unit header") {
            v.push(dwarf.unit(h).expect("parse unit"));
        }
        v
    };
    info!("loaded {} compilation unit(s)", units.len());

    // Find the caller's subprogram, then the MaybeUninit<T> local inside it.
    for unit in &units {
        let Some(sp) = find_subprogram(&dwarf, unit, &caller_leaf) else {
            continue;
        };
        info!("found subprogram DIE for `{caller_leaf}`");
        let Some((var_name, inner)) = find_maybeuninit_local(&dwarf, unit, sp)
        else {
            continue;
        };
        info!("local `{var_name}` is MaybeUninit<{inner}>");
        let leaf = inner.rsplit("::").next().unwrap_or(&inner);
        let struct_die = find_structure(&dwarf, unit, leaf)
            .unwrap_or_else(|| panic!("no DWARF structure_type named `{leaf}`"));
        let layout = read_struct_layout(&dwarf, unit, struct_die);
        info!(
            "target type `{}` is {} bytes with {} field(s)",
            layout.name,
            layout.size,
            layout.fields.len()
        );

        let json = parse_json_object(s);
        info!("parsed {} JSON key(s)", json.len());

        for field in &layout.fields {
            let raw = json
                .iter()
                .find(|(k, _)| k == &field.name)
                .map(|(_, v)| v)
                .unwrap_or_else(|| panic!("JSON has no key `{}`", field.name));
            info!(
                "writing field `{}` : {:?} at offset {} from {raw:?}",
                field.name, field.kind, field.offset
            );
            unsafe { write_field(ptr.add(field.offset), &field.kind, raw) };
        }
        info!("done poking bytes; returning to a hopeful caller");
        return;
    }

    panic!("could not locate `{caller_leaf}` + its MaybeUninit local in DWARF");
}

// ---------------------------------------------------------------------------
// Stack trace -> caller name
// ---------------------------------------------------------------------------

fn caller_function_name() -> String {
    let mut names: Vec<String> = Vec::new();
    backtrace::trace(|frame| {
        backtrace::resolve_frame(frame, |sym| {
            if let Some(name) = sym.name() {
                names.push(strip_hash(&name.to_string()));
            }
        });
        true
    });
    // First frame mentioning from_json is us; the next distinct frame is the
    // caller we care about.
    let mut seen_self = false;
    for n in &names {
        if n.contains("from_json") {
            seen_self = true;
            continue;
        }
        if seen_self
            && !n.contains("backtrace")
            && !n.contains("caller_function_name")
        {
            return n.clone();
        }
    }
    panic!("could not find caller in backtrace: {names:?}");
}

fn strip_hash(name: &str) -> String {
    // drop a trailing rustc disambiguator like `::h3f2a...`
    if let Some(idx) = name.rfind("::h") {
        let tail = &name[idx + 3..];
        if !tail.is_empty() && tail.chars().all(|c| c.is_ascii_hexdigit()) {
            return name[..idx].to_string();
        }
    }
    name.to_string()
}

fn dsym_path() -> PathBuf {
    let exe = std::env::current_exe().expect("current_exe");
    let name = exe.file_name().expect("exe name").to_owned();
    let dir = exe.parent().expect("exe dir");
    let mut p = dir.to_path_buf();
    p.push(format!("{}.dSYM", name.to_string_lossy()));
    p.push("Contents/Resources/DWARF");
    // The DWARF binary inside is named after the *deps* artifact (with a
    // hash), not the final executable, so just take whatever is in there.
    if let Ok(mut entries) = std::fs::read_dir(&p)
        && let Some(Ok(e)) = entries.next() {
            return e.path();
        }
    // maybe debuginfo is embedded in the executable itself
    exe
}

// ---------------------------------------------------------------------------
// DWARF walking
// ---------------------------------------------------------------------------

fn child_offsets(unit: &Unit, parent: Option<UnitOffset>) -> Vec<UnitOffset> {
    let mut tree = unit.entries_tree(parent).expect("entries_tree");
    let root = tree.root().expect("tree root");
    let mut kids = Vec::new();
    let mut it = root.children();
    while let Some(child) = it.next().expect("child") {
        kids.push(child.entry().offset());
    }
    kids
}

fn die_name(dwarf: &Dwarf, unit: &Unit, off: UnitOffset) -> Option<String> {
    let entry = unit.entry(off).ok()?;
    let attr = entry.attr_value(gimli::DW_AT_name)?;
    let s = dwarf.attr_string(unit, attr).ok()?;
    Some(s.to_string_lossy().into_owned())
}

fn tag(unit: &Unit, off: UnitOffset) -> gimli::DwTag {
    unit.entry(off).expect("entry").tag()
}

/// DFS for a `DW_TAG_subprogram` whose name matches `want` and which actually
/// contains our MaybeUninit local (self-selects the right `main`).
fn find_subprogram(dwarf: &Dwarf, unit: &Unit, want: &str) -> Option<UnitOffset> {
    fn dfs(
        dwarf: &Dwarf,
        unit: &Unit,
        parent: Option<UnitOffset>,
        want: &str,
        depth: u32,
    ) -> Option<UnitOffset> {
        if depth > 32 {
            return None;
        }
        for off in child_offsets(unit, parent) {
            let t = tag(unit, off);
            if t == gimli::DW_TAG_subprogram
                && die_name(dwarf, unit, off).as_deref() == Some(want)
                && find_maybeuninit_local(dwarf, unit, off).is_some()
            {
                return Some(off);
            }
            if matches!(
                t,
                gimli::DW_TAG_namespace
                    | gimli::DW_TAG_subprogram
                    | gimli::DW_TAG_structure_type
            )
                && let Some(found) = dfs(dwarf, unit, Some(off), want, depth + 1)
                {
                    return Some(found);
                }
        }
        None
    }
    dfs(dwarf, unit, None, want, 0)
}

/// Within a subprogram, find a `DW_TAG_variable` whose type is
/// `MaybeUninit<...>`; return (variable name, inner type name).
fn find_maybeuninit_local(
    dwarf: &Dwarf,
    unit: &Unit,
    sp: UnitOffset,
) -> Option<(String, String)> {
    for off in child_offsets(unit, Some(sp)) {
        let t = tag(unit, off);
        if t == gimli::DW_TAG_lexical_block {
            // locals live inside lexical blocks in debug builds
            if let Some(found) = find_maybeuninit_local(dwarf, unit, off) {
                return Some(found);
            }
            continue;
        }
        if t != gimli::DW_TAG_variable {
            continue;
        }
        let Some(entry) = unit.entry(off).ok() else {
            continue;
        };
        let Some(gimli::AttributeValue::UnitRef(ty)) =
            entry.attr_value(gimli::DW_AT_type)
        else {
            continue;
        };
        let Some(tyname) = die_name(dwarf, unit, ty) else {
            continue;
        };
        if let Some((_, rest)) = tyname.split_once("MaybeUninit<") {
            let inner = rest.trim_end_matches('>').to_string();
            let var = die_name(dwarf, unit, off).unwrap_or_default();
            return Some((var, inner));
        }
    }
    None
}

fn find_structure(dwarf: &Dwarf, unit: &Unit, want: &str) -> Option<UnitOffset> {
    fn has_byte_size(unit: &Unit, off: UnitOffset) -> bool {
        unit.entry(off)
            .ok()
            .and_then(|e| e.attr_value(gimli::DW_AT_byte_size))
            .is_some()
    }
    fn dfs(
        dwarf: &Dwarf,
        unit: &Unit,
        parent: Option<UnitOffset>,
        want: &str,
        depth: u32,
    ) -> Option<UnitOffset> {
        if depth > 32 {
            return None;
        }
        for off in child_offsets(unit, parent) {
            let t = tag(unit, off);
            if t == gimli::DW_TAG_structure_type
                && die_name(dwarf, unit, off).as_deref() == Some(want)
                && has_byte_size(unit, off)
            {
                return Some(off);
            }
            if matches!(
                t,
                gimli::DW_TAG_namespace
                    | gimli::DW_TAG_subprogram
                    | gimli::DW_TAG_structure_type
                    | gimli::DW_TAG_lexical_block
            )
                && let Some(f) = dfs(dwarf, unit, Some(off), want, depth + 1) {
                    return Some(f);
                }
        }
        None
    }
    dfs(dwarf, unit, None, want, 0)
}

#[derive(Debug)]
struct Layout {
    name: String,
    size: u64,
    fields: Vec<Field>,
}

#[derive(Debug)]
struct Field {
    name: String,
    offset: usize,
    kind: TypeKind,
}

#[derive(Debug, Clone)]
enum TypeKind {
    Uint(u8), // size in bytes
    Sint(u8), // size in bytes
    Bool,
    F32,
    F64,
    StdString,
    StrRef,
    Unknown(String),
}

fn read_struct_layout(dwarf: &Dwarf, unit: &Unit, off: UnitOffset) -> Layout {
    let entry = unit.entry(off).expect("struct entry");
    let size = match entry.attr_value(gimli::DW_AT_byte_size) {
        Some(gimli::AttributeValue::Udata(n)) => n,
        _ => 0,
    };
    let name = die_name(dwarf, unit, off).unwrap_or_default();
    let mut fields = Vec::new();
    for moff in child_offsets(unit, Some(off)) {
        if tag(unit, moff) != gimli::DW_TAG_member {
            continue;
        }
        let m = unit.entry(moff).expect("member entry");
        let fname = die_name(dwarf, unit, moff).unwrap_or_default();
        let offset = match m.attr_value(gimli::DW_AT_data_member_location) {
            Some(gimli::AttributeValue::Udata(n)) => n as usize,
            Some(gimli::AttributeValue::Sdata(n)) => n as usize,
            None => 0,
            other => panic!("unhandled member location for `{fname}`: {other:?}"),
        };
        let Some(gimli::AttributeValue::UnitRef(tyref)) =
            m.attr_value(gimli::DW_AT_type)
        else {
            panic!("member `{fname}` has no type ref");
        };
        let kind = classify_type(dwarf, unit, tyref);
        fields.push(Field {
            name: fname,
            offset,
            kind,
        });
    }
    Layout { name, size, fields }
}

fn classify_type(dwarf: &Dwarf, unit: &Unit, off: UnitOffset) -> TypeKind {
    let entry = unit.entry(off).expect("type entry");
    let t = entry.tag();
    let name = die_name(dwarf, unit, off).unwrap_or_default();
    match t {
        gimli::DW_TAG_base_type => {
            let size = match entry.attr_value(gimli::DW_AT_byte_size) {
                Some(gimli::AttributeValue::Udata(n)) => n as u8,
                _ => 0,
            };
            let enc = match entry.attr_value(gimli::DW_AT_encoding) {
                Some(gimli::AttributeValue::Encoding(e)) => e,
                _ => gimli::DwAte(0),
            };
            match enc {
                gimli::DW_ATE_boolean => TypeKind::Bool,
                gimli::DW_ATE_float if size == 4 => TypeKind::F32,
                gimli::DW_ATE_float => TypeKind::F64,
                gimli::DW_ATE_signed | gimli::DW_ATE_signed_char => {
                    TypeKind::Sint(size)
                }
                gimli::DW_ATE_unsigned | gimli::DW_ATE_unsigned_char => {
                    TypeKind::Uint(size)
                }
                _ => TypeKind::Unknown(name),
            }
        }
        gimli::DW_TAG_structure_type if name == "String" => TypeKind::StdString,
        gimli::DW_TAG_structure_type if name == "&str" => TypeKind::StrRef,
        _ => TypeKind::Unknown(name),
    }
}

// ---------------------------------------------------------------------------
// Writing into raw memory
// ---------------------------------------------------------------------------

unsafe fn write_field(dst: *mut u8, kind: &TypeKind, val: &JsonVal) {
    match kind {
        TypeKind::Uint(n) => {
            let bytes = val.as_int().to_le_bytes();
            unsafe {
                std::ptr::copy_nonoverlapping(bytes.as_ptr(), dst, *n as usize)
            };
        }
        TypeKind::Sint(n) => {
            let bytes = val.as_sint().to_le_bytes();
            unsafe {
                std::ptr::copy_nonoverlapping(bytes.as_ptr(), dst, *n as usize)
            };
        }
        TypeKind::Bool => {
            let b = matches!(val, JsonVal::Bool(true));
            unsafe { *dst = b as u8 };
        }
        TypeKind::F32 => {
            let f = val.as_float() as f32;
            unsafe {
                std::ptr::copy_nonoverlapping(f.to_le_bytes().as_ptr(), dst, 4)
            };
        }
        TypeKind::F64 => {
            let f = val.as_float();
            unsafe {
                std::ptr::copy_nonoverlapping(f.to_le_bytes().as_ptr(), dst, 8)
            };
        }
        TypeKind::StdString => {
            let s = val.as_str().to_owned();
            unsafe { std::ptr::write(dst as *mut String, s) };
        }
        TypeKind::StrRef => {
            let s: &'static str =
                Box::leak(val.as_str().to_owned().into_boxed_str());
            unsafe { std::ptr::write(dst as *mut &'static str, s) };
        }
        TypeKind::Unknown(n) => panic!("don't know how to write type `{n}`"),
    }
}

// ---------------------------------------------------------------------------
// A deliberately tiny, lenient JSON parser (the input has a trailing comma)
// ---------------------------------------------------------------------------

#[derive(Debug)]
enum JsonVal {
    Str(String),
    Num(String),
    Bool(bool),
    Null,
}

impl JsonVal {
    fn as_int(&self) -> u128 {
        match self {
            JsonVal::Num(s) => s.parse().expect("uint"),
            JsonVal::Str(s) => s.parse().expect("uint"),
            _ => panic!("expected integer, got {self:?}"),
        }
    }
    fn as_sint(&self) -> i128 {
        match self {
            JsonVal::Num(s) => s.parse().expect("int"),
            JsonVal::Str(s) => s.parse().expect("int"),
            _ => panic!("expected integer, got {self:?}"),
        }
    }
    fn as_float(&self) -> f64 {
        match self {
            JsonVal::Num(s) => s.parse().expect("float"),
            _ => panic!("expected number, got {self:?}"),
        }
    }
    fn as_str(&self) -> &str {
        match self {
            JsonVal::Str(s) => s,
            _ => panic!("expected string, got {self:?}"),
        }
    }
}

fn parse_json_object(s: &str) -> Vec<(String, JsonVal)> {
    let b = s.as_bytes();
    let mut i = 0;
    skip_ws(b, &mut i);
    assert_eq!(b[i], b'{', "expected object");
    i += 1;
    let mut out = Vec::new();
    loop {
        skip_ws(b, &mut i);
        if b[i] == b'}' {
            break;
        }
        let key = parse_string(b, &mut i);
        skip_ws(b, &mut i);
        assert_eq!(b[i], b':', "expected colon");
        i += 1;
        skip_ws(b, &mut i);
        let val = parse_value(b, &mut i);
        out.push((key, val));
        skip_ws(b, &mut i);
        if b[i] == b',' {
            i += 1; // tolerate a trailing comma: the loop re-checks for `}`
        }
    }
    out
}

fn skip_ws(b: &[u8], i: &mut usize) {
    while *i < b.len() && b[*i].is_ascii_whitespace() {
        *i += 1;
    }
}

fn parse_string(b: &[u8], i: &mut usize) -> String {
    assert_eq!(b[*i], b'"', "expected string");
    *i += 1;
    let mut s = String::new();
    while b[*i] != b'"' {
        if b[*i] == b'\\' {
            *i += 1;
            match b[*i] {
                b'n' => s.push('\n'),
                b't' => s.push('\t'),
                b'"' => s.push('"'),
                b'\\' => s.push('\\'),
                c => s.push(c as char),
            }
        } else {
            s.push(b[*i] as char);
        }
        *i += 1;
    }
    *i += 1;
    s
}

fn parse_value(b: &[u8], i: &mut usize) -> JsonVal {
    match b[*i] {
        b'"' => JsonVal::Str(parse_string(b, i)),
        b't' => {
            *i += 4;
            JsonVal::Bool(true)
        }
        b'f' => {
            *i += 5;
            JsonVal::Bool(false)
        }
        b'n' => {
            *i += 4;
            JsonVal::Null
        }
        _ => {
            let start = *i;
            while *i < b.len()
                && (b[*i].is_ascii_digit()
                    || matches!(b[*i], b'-' | b'+' | b'.' | b'e' | b'E'))
            {
                *i += 1;
            }
            JsonVal::Num(
                std::str::from_utf8(&b[start..*i]).unwrap().to_string(),
            )
        }
    }
}
