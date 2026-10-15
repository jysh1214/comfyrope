//! prompts.json: prompts worth keeping, each with a title and its text, to paste into the next img2img.py
//! command (or `"$(rope prompt show TITLE)"`). They belong to the library, not to any image.

use crate::catalog::random_id;
use crate::{Library, atomic, journal, time};
use anyhow::{Context, Result, bail};
use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};
use std::fs;

pub const FORMAT: u32 = 1;

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Prompt {
    pub id: String,
    pub title: String,
    pub text: String,
    pub created: String,
    pub updated: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Prompts {
    pub format: u32,
    /// In the order they were added.
    #[serde(default)]
    pub prompts: Vec<Prompt>,
    #[serde(flatten)]
    pub other: Map<String, Value>,
}

impl Default for Prompts {
    fn default() -> Self {
        Prompts { format: FORMAT, prompts: Vec::new(), other: Map::new() }
    }
}

impl Prompts {
    /// Reads prompts.json; a library without one has no prompts. A file that does not parse is an error, never an
    /// empty list: saving that would lose every prompt.
    pub fn load(lib: &Library) -> Result<Prompts> {
        let path = lib.prompts_path();
        let bytes = match fs::read(&path) {
            Ok(b) => b,
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Ok(Prompts::default()),
            Err(e) => return Err(e).with_context(|| format!("cannot read {}", path.display())),
        };
        let prompts: Prompts =
            serde_json::from_slice(&bytes).with_context(|| format!("{} does not parse", path.display()))?;
        if prompts.format > FORMAT {
            bail!("{} has format {}; this rope reads up to {FORMAT}", path.display(), prompts.format);
        }
        Ok(prompts)
    }

    pub fn save(&self, lib: &Library) -> Result<()> {
        let mut bytes = serde_json::to_vec_pretty(self)?;
        bytes.push(b'\n');
        atomic::write(&lib.prompts_path(), &bytes)
    }

    pub fn get(&self, id: &str) -> Option<&Prompt> {
        self.prompts.iter().find(|p| p.id == id)
    }

    /// The prompt `name` means: its id, or its title (case does not matter), or a prefix of its id or a part of its
    /// title when that fits only one.
    pub fn find(&self, name: &str) -> Result<&Prompt> {
        let name = name.trim();
        if let Some(p) = self.prompts.iter().find(|p| p.id == name || p.title.eq_ignore_ascii_case(name)) {
            return Ok(p);
        }
        let lower = name.to_lowercase();
        let matches: Vec<&Prompt> = self
            .prompts
            .iter()
            .filter(|p| !lower.is_empty() && (p.id.starts_with(name) || p.title.to_lowercase().contains(&lower)))
            .collect();
        match matches.as_slice() {
            [one] => Ok(one),
            [] => bail!("no prompt {name:?}; rope prompt ls lists them"),
            many => {
                let titles: Vec<String> = many.iter().map(|p| format!("{:?} ({})", p.title, p.id)).collect();
                bail!("{name:?} fits several prompts: {}", titles.join(", "))
            }
        }
    }
}

fn check(title: &str) -> Result<String> {
    let title = title.trim();
    if title.is_empty() {
        bail!("a prompt needs a title");
    }
    if title.chars().any(|c| c.is_control()) {
        bail!("a prompt's title is one line");
    }
    Ok(title.to_owned())
}

/// Changes prompts.json under the library lock and journals it, so the GUI sees it.
fn edit_prompts<T>(lib: &Library, change: impl FnOnce(&mut Prompts) -> Result<T>) -> Result<T> {
    let _lock = lib.lock()?;
    let mut prompts = Prompts::load(lib)?;
    let out = change(&mut prompts)?;
    prompts.save(lib)?;
    journal::append(lib, "prompt", &[])?;
    Ok(out)
}

pub fn add(lib: &Library, title: &str, text: &str) -> Result<Prompt> {
    let title = check(title)?;
    edit_prompts(lib, |ps| {
        let now = time::now();
        let mut id = random_id();
        while ps.get(&id).is_some() {
            id = random_id();
        }
        let p = Prompt { id, title, text: text.trim_end().to_owned(), created: now.clone(), updated: now };
        ps.prompts.push(p.clone());
        Ok(p)
    })
}

/// Changes the title, the text, or both, of the prompt `id`.
pub fn edit(lib: &Library, id: &str, title: Option<&str>, text: Option<&str>) -> Result<Prompt> {
    let title = title.map(check).transpose()?;
    edit_prompts(lib, |ps| {
        let p = ps.prompts.iter_mut().find(|p| p.id == id).with_context(|| format!("no prompt {id}"))?;
        if let Some(t) = title {
            p.title = t;
        }
        if let Some(t) = text {
            p.text = t.trim_end().to_owned();
        }
        p.updated = time::now();
        Ok(p.clone())
    })
}

pub fn remove(lib: &Library, id: &str) -> Result<Prompt> {
    edit_prompts(lib, |ps| {
        let i = ps.prompts.iter().position(|p| p.id == id).with_context(|| format!("no prompt {id}"))?;
        Ok(ps.prompts.remove(i))
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn prompts_are_added_found_edited_and_removed() {
        let dir = tempfile::tempdir().unwrap();
        let lib = Library::init(dir.path()).unwrap();
        assert!(Prompts::load(&lib).unwrap().prompts.is_empty());
        let sea = add(&lib, " 黃昏海邊 ", "把背景換成黃昏的海邊\n").unwrap();
        let sky = add(&lib, "Starry sky", "把天空換成星空").unwrap();
        assert_eq!((sea.title.as_str(), sea.text.as_str()), ("黃昏海邊", "把背景換成黃昏的海邊"));
        assert!(add(&lib, "  ", "x").is_err());

        let ps = Prompts::load(&lib).unwrap();
        assert_eq!(ps.prompts.len(), 2, "in the order added");
        assert_eq!(ps.find("starry SKY").unwrap().id, sky.id);
        assert_eq!(ps.find("海邊").unwrap().id, sea.id);
        assert_eq!(ps.find(&sky.id[..4]).unwrap().id, sky.id);
        assert!(ps.find("把").is_err(), "a part of the text is not a name");
        add(&lib, "Sky at dawn", "黎明的天空").unwrap();
        let ps = Prompts::load(&lib).unwrap();
        assert!(ps.find("sky").unwrap_err().to_string().contains("several"));
        assert_eq!(ps.find("starry").unwrap().id, sky.id);

        let sky2 = edit(&lib, &sky.id, Some("Night sky"), None).unwrap();
        assert_eq!((sky2.title.as_str(), sky2.text.as_str()), ("Night sky", "把天空換成星空"));
        assert!(edit(&lib, "nope", None, Some("x")).is_err());
        remove(&lib, &sea.id).unwrap();
        let ps = Prompts::load(&lib).unwrap();
        assert_eq!(ps.prompts.iter().map(|p| p.title.as_str()).collect::<Vec<_>>(), ["Night sky", "Sky at dawn"]);
        let journal = fs::read_to_string(lib.journal_path()).unwrap();
        assert_eq!(journal.matches(r#""op":"prompt""#).count(), 5);
    }
}
