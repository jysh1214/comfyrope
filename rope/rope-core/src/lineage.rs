//! Upstream and downstream. Only the made image records its inputs (meta.json `made.inputs`, in order); the
//! images made from one are found by reading every meta.json.

use crate::Meta;
use std::collections::{HashMap, HashSet};

#[derive(Debug, Default)]
pub struct Lineage {
    parents: HashMap<String, Vec<String>>,
    children: HashMap<String, Vec<String>>,
}

impl Lineage {
    pub fn new(metas: &[Meta]) -> Lineage {
        let mut ordered: Vec<&Meta> = metas.iter().collect();
        ordered.sort_by(|a, b| (&a.added, &a.sha256).cmp(&(&b.added, &b.sha256)));
        let mut l = Lineage::default();
        for m in ordered {
            let up = m.upstream();
            let mut seen = HashSet::new();
            for p in &up {
                if seen.insert(p.clone()) {
                    l.children.entry(p.clone()).or_default().push(m.sha256.clone());
                }
            }
            if !up.is_empty() {
                l.parents.insert(m.sha256.clone(), up);
            }
        }
        l
    }

    /// The inputs, in order (image 1, image 2, ...), repeats included.
    pub fn parents(&self, id: &str) -> &[String] {
        self.parents.get(id).map_or(&[], Vec::as_slice)
    }

    /// The images made from this one, oldest first, each once.
    pub fn children(&self, id: &str) -> &[String] {
        self.children.get(id).map_or(&[], Vec::as_slice)
    }

    /// Everything above (`up`) or below an image, depth first: (depth, id, already listed). An image reached a
    /// second time is listed again but not expanded again.
    pub fn walk(&self, id: &str, up: bool) -> Vec<(usize, String, bool)> {
        let mut out = Vec::new();
        let mut seen = HashSet::from([id.to_owned()]);
        self.walk_into(id, up, 1, &mut seen, &mut out);
        out
    }

    fn walk_into(
        &self,
        id: &str,
        up: bool,
        depth: usize,
        seen: &mut HashSet<String>,
        out: &mut Vec<(usize, String, bool)>,
    ) {
        let next = if up { self.parents(id) } else { self.children(id) };
        for n in next {
            let repeat = !seen.insert(n.clone());
            out.push((depth, n.clone(), repeat));
            if !repeat {
                self.walk_into(n, up, depth + 1, seen, out);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn meta(id: &str, added: &str, inputs: &[&str]) -> Meta {
        let inputs: Vec<_> = inputs.iter().map(|i| json!({"id": i})).collect();
        let made = if inputs.is_empty() { json!(null) } else { json!({"inputs": inputs}) };
        serde_json::from_value(json!({"format": 1, "sha256": id, "file": "f", "name": id, "added": added,
                                      "image": {"bytes": 1}, "made": made}))
        .unwrap()
    }

    #[test]
    fn a_chain_with_two_inputs_walks_both_ways() {
        // a -> c, b -> c (c made from a and b), c -> d (d made from c, twice)
        let metas =
            [meta("a", "1", &[]), meta("b", "1", &[]), meta("c", "2", &["a", "b"]), meta("d", "3", &["c", "c"])];
        let l = Lineage::new(&metas);
        assert_eq!(l.parents("c"), ["a", "b"]);
        assert_eq!(l.parents("d"), ["c", "c"]);
        assert_eq!(l.children("c"), ["d"]);
        let up: Vec<_> = l.walk("d", true).into_iter().collect();
        assert_eq!(up, [(1, "c".into(), false), (2, "a".into(), false), (2, "b".into(), false), (1, "c".into(), true)]);
        let down: Vec<_> = l.walk("a", false).into_iter().map(|(_, id, _)| id).collect();
        assert_eq!(down, ["c", "d"]);
    }
}
