//! A deliberately tiny, lenient JSON parser. The demo input has a trailing
//! comma, so a strict parser is off the table anyway. Supports nested
//! objects and arrays so the recursive writer has something to chew on.

#[derive(Debug, Clone)]
pub enum Json {
    Str(String),
    Num(String),
    Bool(bool),
    Null,
    Array(Vec<Json>),
    Object(Vec<(String, Json)>),
}

impl Json {
    pub fn as_u128(&self) -> u128 {
        match self {
            Json::Num(s) => s.parse().expect("uint"),
            Json::Str(s) => s.parse().expect("uint"),
            Json::Bool(b) => *b as u128,
            _ => panic!("expected integer, got {self:?}"),
        }
    }
    pub fn as_i128(&self) -> i128 {
        match self {
            Json::Num(s) => s.parse().expect("int"),
            Json::Str(s) => s.parse().expect("int"),
            _ => panic!("expected integer, got {self:?}"),
        }
    }
    pub fn as_f64(&self) -> f64 {
        match self {
            Json::Num(s) => s.parse().expect("float"),
            _ => panic!("expected number, got {self:?}"),
        }
    }
    pub fn as_str(&self) -> &str {
        match self {
            Json::Str(s) => s,
            _ => panic!("expected string, got {self:?}"),
        }
    }
    pub fn get<'a>(&'a self, key: &str) -> Option<&'a Json> {
        match self {
            Json::Object(kvs) => {
                kvs.iter().find(|(k, _)| k == key).map(|(_, v)| v)
            }
            _ => None,
        }
    }
}

pub fn parse(s: &str) -> Json {
    let b = s.as_bytes();
    let mut i = 0;
    
    parse_value(b, &mut i)
}

fn skip_ws(b: &[u8], i: &mut usize) {
    while *i < b.len() && b[*i].is_ascii_whitespace() {
        *i += 1;
    }
}

fn parse_value(b: &[u8], i: &mut usize) -> Json {
    skip_ws(b, i);
    match b[*i] {
        b'{' => parse_object(b, i),
        b'[' => parse_array(b, i),
        b'"' => Json::Str(parse_string(b, i)),
        b't' => {
            *i += 4;
            Json::Bool(true)
        }
        b'f' => {
            *i += 5;
            Json::Bool(false)
        }
        b'n' => {
            *i += 4;
            Json::Null
        }
        _ => {
            let start = *i;
            while *i < b.len()
                && (b[*i].is_ascii_digit()
                    || matches!(b[*i], b'-' | b'+' | b'.' | b'e' | b'E'))
            {
                *i += 1;
            }
            Json::Num(std::str::from_utf8(&b[start..*i]).unwrap().to_string())
        }
    }
}

fn parse_object(b: &[u8], i: &mut usize) -> Json {
    *i += 1; // {
    let mut out = Vec::new();
    loop {
        skip_ws(b, i);
        if b[*i] == b'}' {
            *i += 1;
            break;
        }
        let key = parse_string(b, i);
        skip_ws(b, i);
        assert_eq!(b[*i], b':', "expected colon");
        *i += 1;
        let val = parse_value(b, i);
        out.push((key, val));
        skip_ws(b, i);
        if b[*i] == b',' {
            *i += 1; // tolerate a trailing comma; the loop re-checks for `}`
        }
    }
    Json::Object(out)
}

fn parse_array(b: &[u8], i: &mut usize) -> Json {
    *i += 1; // [
    let mut out = Vec::new();
    loop {
        skip_ws(b, i);
        if b[*i] == b']' {
            *i += 1;
            break;
        }
        out.push(parse_value(b, i));
        skip_ws(b, i);
        if b[*i] == b',' {
            *i += 1; // trailing comma tolerated here too
        }
    }
    Json::Array(out)
}

fn parse_string(b: &[u8], i: &mut usize) -> String {
    assert_eq!(b[*i], b'"', "expected string");
    *i += 1;
    // Accumulate raw bytes so multibyte UTF-8 (🦀) survives intact.
    let mut bytes = Vec::new();
    while b[*i] != b'"' {
        if b[*i] == b'\\' {
            *i += 1;
            match b[*i] {
                b'n' => bytes.push(b'\n'),
                b't' => bytes.push(b'\t'),
                b'"' => bytes.push(b'"'),
                b'\\' => bytes.push(b'\\'),
                c => bytes.push(c),
            }
        } else {
            bytes.push(b[*i]);
        }
        *i += 1;
    }
    *i += 1;
    String::from_utf8(bytes).expect("valid UTF-8 in JSON string")
}
