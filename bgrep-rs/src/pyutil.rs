//! Python-semantics compatibility helpers.
//!
//! lanes2.py (the frozen v7 pipeline) leans on several CPython/pathlib
//! behaviors that Rust's stdlib does not reproduce out of the box:
//!   - `str.splitlines()` splits on a broader set of Unicode line boundaries
//!     than Rust's `str::lines()` (which only knows `\n`/`\r\n`).
//!   - `sorted(Path.rglob("*"))` orders paths by their **parts tuple**
//!     (component-wise comparison), not by comparing the joined string --
//!     these differ whenever a path component contains a character that
//!     sorts before `/` (e.g. `.`, `-`).
//!   - `str(Path(rel).parent)` renders `"."` for a top-level (no-directory)
//!     relative path, where `Path::parent()` in Rust yields `Some("")`.
//!   - `os.path.normpath(str(Path(base) / spec))` lexically resolves `.`/`..`
//!     components without touching the filesystem.
//!
//! Each helper here exists to make Rust's output byte-identical to the
//! Python reference in these specific spots. See PARITY_NOTES.md.

/// Split `s` exactly like Python's `str.splitlines()`: recognizes `\n`, `\r`,
/// `\r\n` (as one boundary), `\v` (0x0B), `\f` (0x0C), `\x1c`, `\x1d`, `\x1e`,
/// `\x85` (NEL), ` ` (LS), ` ` (PS). No trailing empty element for
/// a string ending in a line boundary; empty input yields an empty Vec.
pub fn py_splitlines(s: &str) -> Vec<&str> {
    let bytes_indices: Vec<(usize, char)> = s.char_indices().collect();
    let mut out = Vec::new();
    let mut start = 0usize;
    let mut i = 0usize;
    while i < bytes_indices.len() {
        let (byte_pos, ch) = bytes_indices[i];
        let is_boundary = matches!(
            ch,
            '\n' | '\r' | '\u{0b}' | '\u{0c}' | '\u{1c}' | '\u{1d}' | '\u{1e}' | '\u{85}'
                | '\u{2028}' | '\u{2029}'
        );
        if is_boundary {
            let line_end = byte_pos;
            let mut next_i = i + 1;
            // \r\n counts as a single boundary
            if ch == '\r' && i + 1 < bytes_indices.len() && bytes_indices[i + 1].1 == '\n' {
                next_i = i + 2;
            }
            out.push(&s[start..line_end]);
            start = if next_i < bytes_indices.len() {
                bytes_indices[next_i].0
            } else {
                s.len()
            };
            i = next_i;
            continue;
        }
        i += 1;
    }
    if start < s.len() {
        out.push(&s[start..]);
    }
    out
}

/// Replicates `str(Path(rel).parent)` for a POSIX-style relative path
/// string using `/` separators: `"."` when `rel` has no directory
/// component, else everything before the last `/`.
pub fn py_parent(rel: &str) -> &str {
    match rel.rfind('/') {
        Some(idx) => &rel[..idx],
        None => ".",
    }
}

/// Replicates `Path(rel).parent.name` -- the last path component of the
/// parent directory (empty string if the parent is "." / root).
pub fn py_parent_name(rel: &str) -> &str {
    let parent = py_parent(rel);
    if parent == "." {
        return "";
    }
    match parent.rfind('/') {
        Some(idx) => &parent[idx + 1..],
        None => parent,
    }
}

/// Split a relative path/parent string into pathlib-style `.parts`: no
/// empty components, no bare "." components, `".."` preserved. `"."` itself
/// yields an empty Vec (matches `PurePosixPath(".").parts == ()`).
pub fn py_parts(s: &str) -> Vec<&str> {
    if s == "." || s.is_empty() {
        return Vec::new();
    }
    s.split('/').filter(|c| !c.is_empty() && *c != ".").collect()
}

/// Replicates `str(Path(base) / spec)` followed by `os.path.normpath(...)`
/// for the JS/TS relative-import resolution in build_import_graph(): joins
/// pathlib-parsed parts (dropping "." components) of `base` and `spec`,
/// then lexically resolves ".." against preceding real components exactly
/// like `os.path.normpath` (never crossing above the relative root: an
/// unresolvable leading ".." is kept as-is). Returns "." if the result is
/// empty.
pub fn normpath_join(base: &str, spec: &str) -> String {
    let mut parts: Vec<&str> = py_parts(base);
    parts.extend(py_parts(spec));

    let mut resolved: Vec<&str> = Vec::with_capacity(parts.len());
    for p in parts {
        if p == ".." {
            match resolved.last() {
                Some(&last) if last != ".." => {
                    resolved.pop();
                }
                _ => resolved.push(".."),
            }
        } else {
            resolved.push(p);
        }
    }
    if resolved.is_empty() {
        ".".to_string()
    } else {
        resolved.join("/")
    }
}

/// Simple relative path join (no "." /".." normalization needed by callers
/// that only ever append a bare component name), matching
/// `str(Path(base) / comp)` when `comp` has no leading dot components.
pub fn path_join_simple(base: &str, comp: &str) -> String {
    if base == "." {
        comp.to_string()
    } else {
        format!("{base}/{comp}")
    }
}

/// Order paths the way `sorted(repo_path.rglob("*"))` orders them: by
/// component-wise (parts-tuple) comparison, NOT by comparing the joined
/// string -- these differ whenever a component contains a character that
/// sorts before `/` (e.g. "test.py" vs "test/a.py": pathlib puts
/// ("test","a.py") before ("test.py",) because "test" < "test.py" as a
/// standalone component, while naive string comparison of the joined form
/// would order "test.py" first since '.' < '/').
pub fn path_sort_key(rel: &str) -> Vec<&str> {
    rel.split('/').collect()
}

/// Lowercase a string the way Python's `str.lower()` does. Rust's
/// `to_lowercase()` uses the Unicode default case-folding tables, which for
/// every ASCII input (the overwhelming case for source-code identifiers and
/// paths) is byte-identical to CPython's `str.lower()`. Kept as a named
/// wrapper so any future divergence is easy to find and patch in one place.
pub fn py_lower(s: &str) -> String {
    s.to_lowercase()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn splitlines_basic() {
        assert_eq!(py_splitlines("a\nb\n"), vec!["a", "b"]);
        assert_eq!(py_splitlines("a\nb"), vec!["a", "b"]);
        assert_eq!(py_splitlines(""), Vec::<&str>::new());
        assert_eq!(py_splitlines("a\n\nb"), vec!["a", "", "b"]);
        assert_eq!(py_splitlines("a\r\nb"), vec!["a", "b"]);
        assert_eq!(py_splitlines("a\rb"), vec!["a", "b"]);
    }

    #[test]
    fn parent_basic() {
        assert_eq!(py_parent("foo.py"), ".");
        assert_eq!(py_parent("a/b/foo.py"), "a/b");
        assert_eq!(py_parent("a/foo.py"), "a");
    }

    #[test]
    fn parent_name_basic() {
        assert_eq!(py_parent_name("a/b/foo.py"), "b");
        assert_eq!(py_parent_name("foo.py"), "");
    }

    #[test]
    fn normpath_join_basic() {
        assert_eq!(normpath_join(".", "./x"), "x");
        assert_eq!(normpath_join("a/b", "../c"), "a/c");
        assert_eq!(normpath_join("a", "../.."), "..");
        assert_eq!(normpath_join("a/b", ".."), "a");
        assert_eq!(normpath_join("a", ".."), ".");
    }

    #[test]
    fn sort_key_matches_pathlib_order() {
        // ("test", "a.py") < ("test.py",) under pathlib parts comparison.
        let mut v = vec!["test.py", "test/a.py"];
        v.sort_by_key(|p| path_sort_key(p));
        assert_eq!(v, vec!["test/a.py", "test.py"]);
    }
}
