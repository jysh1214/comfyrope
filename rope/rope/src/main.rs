//! rope: the command line of the rope image library. The on-disk layout is described in rope-core.

use anyhow::{Context, Result, bail};
use clap::{Parser, Subcommand};
use rope_core::catalog::Catalog;
use rope_core::hash::short;
use rope_core::lineage::Lineage;
use rope_core::store::{self, NewImage};
use rope_core::{Library, Meta, check, journal, purge, repro, time};
use serde_json::{Value, json};
use std::collections::{BTreeMap, HashMap};
use std::io::Read;
use std::path::{Path, PathBuf};

const EXAMPLES: &str = "\
Examples:
  rope mkdir -p trips/kyoto
  rope import ~/Pictures/ref --into trips/kyoto --tag 參考 --tag 京都
  rope import --tree ~/Pictures/北海道    # a rope folder 北海道 at the top, with its subfolders
  rope ls trips/kyoto -l
  rope show 3fa9c1
  rope repro 3fa9c1                 # the command that makes it again
  rope tag add 海邊 @3fa9c1 @9b2e0f
  rope fav @3fa9c1                  # a favorite: rope-gui shows it first
  rope mv @3fa9c1 @9b2e0f portraits # images into a folder; / is unfiled
  rope mv trips/kyoto travel/京都   # rename or move a folder
  rope rm @3fa9c1 && rope purge --trash

An image is named by at least 6 hex digits of its id (with or without @), or by a file with the same bytes.
Folders exist only in rope; they are paths like trips/kyoto, and / is the top, where unfiled images are.";

#[derive(Parser)]
#[command(name = "rope", version, about = "An image library that keeps each image by its hash, with where it came \
from, how it was made, its upstream images, a note, folders and tags.", after_help = EXAMPLES)]
struct Cli {
    /// The library [default: $ROPE_LIBRARY, else ~/.local/share/rope]
    #[arg(long, global = true, value_name = "DIR")]
    library: Option<PathBuf>,
    /// Print JSON, for scripts such as img2img.py
    #[arg(long, global = true)]
    json: bool,
    #[command(subcommand)]
    cmd: Cmd,
}

#[derive(Subcommand)]
enum Cmd {
    /// Make a library: at DIR, or at the default location
    Init { dir: Option<PathBuf> },
    /// Print where the library is
    Where {
        /// Also check that this folder exists, and print its id
        #[arg(long, value_name = "FOLDER")]
        folder: Option<String>,
    },
    /// Bring files, or folders of images (recursively), into rope
    Import {
        #[arg(required = true)]
        paths: Vec<PathBuf>,
        /// Put them in this folder, made if there is none
        #[arg(long, value_name = "FOLDER")]
        into: Option<String>,
        /// Add this tag; repeatable
        #[arg(long = "tag", value_name = "TAG")]
        tags: Vec<String>,
        /// Add this to their notes
        #[arg(long)]
        note: Option<String>,
        /// Make each folder given a rope folder of the same name, with its subfolders, in --into or at the top
        #[arg(long)]
        tree: bool,
        #[arg(long, default_value = "import", hide = true)]
        via: String,
        /// Record where the img2img repo is (img2img.py passes this)
        #[arg(long, hide = true)]
        repo: Option<PathBuf>,
    },
    /// Add an image a tool made, with its recipe (img2img.py uses this)
    #[command(hide = true)]
    Add {
        file: PathBuf,
        /// The recipe, a JSON file or - for stdin
        #[arg(long)]
        made: PathBuf,
        /// The API workflow, a JSON file
        #[arg(long)]
        graph: Option<PathBuf>,
        #[arg(long)]
        name: Option<String>,
        #[arg(long, value_name = "FOLDER")]
        into: Option<String>,
        #[arg(long = "tag", value_name = "TAG")]
        tags: Vec<String>,
        #[arg(long)]
        note: Option<String>,
        /// Also put its unfiled inputs in FOLDER
        #[arg(long)]
        file_inputs: bool,
        #[arg(long)]
        repo: Option<PathBuf>,
    },
    /// Store the converted file uploaded in place of an image (img2img.py uses this)
    #[command(hide = true)]
    Rendition {
        id: String,
        file: PathBuf,
        /// How it was made, as JSON
        #[arg(long, default_value = "{}")]
        info: String,
    },
    /// Record a replay of an image: compare FILE's pixels with it, and add FILE if they differ (img2img.py uses this)
    #[command(hide = true)]
    Replayed {
        id: String,
        file: PathBuf,
        #[arg(long)]
        made: PathBuf,
        #[arg(long)]
        graph: Option<PathBuf>,
        #[arg(long)]
        repo: Option<PathBuf>,
    },
    /// Make a folder
    Mkdir {
        /// Make missing parents too, and accept an existing folder
        #[arg(short)]
        parents: bool,
        path: String,
    },
    /// Move images (@id ...) into a folder, or rename or move a folder: mv OLD NEW
    Mv {
        #[arg(required = true, num_args = 2..)]
        args: Vec<String>,
    },
    /// Remove an empty folder; with -r, a folder with everything in it, its images going to the trash
    Rmdir {
        /// Also a folder that is not empty: remove its subfolders too, and move its images to the trash
        #[arg(short)]
        recursive: bool,
        path: String,
    },
    /// List a folder: its subfolders, then its images, oldest first
    Ls {
        /// The folder [default: the top, with the unfiled images]
        folder: Option<String>,
        /// Every image with this tag, wherever it is
        #[arg(long)]
        tag: Option<String>,
        /// What is in the trash
        #[arg(long)]
        trash: bool,
        /// Only the favorites
        #[arg(long)]
        fav: bool,
        /// Include the images in subfolders
        #[arg(short, long)]
        recursive: bool,
        /// Size, tags and folder too
        #[arg(short)]
        long: bool,
    },
    /// The folder tree, with how many images each folder holds, and the tags
    Tree,
    /// Everything about an image
    Show {
        id: String,
        /// With --json: also list the images made from it (reads every meta.json)
        #[arg(long)]
        downstream: bool,
    },
    /// Print the path of an image's file in the library
    Path { id: String },
    /// Change an image's display name (its file is not touched)
    Name { id: String, name: String },
    /// Print an image's note, or change it
    Note {
        id: String,
        /// Replace the note ("" removes it)
        #[arg(long, conflicts_with_all = ["append", "edit"])]
        set: Option<String>,
        /// Add a paragraph
        #[arg(long, conflicts_with = "edit")]
        append: Option<String>,
        /// Edit it in $EDITOR
        #[arg(short, long)]
        edit: bool,
    },
    /// Add, remove or rename tags
    #[command(subcommand)]
    Tag(TagCmd),
    /// Every tag, with how many images have it
    Tags,
    /// Mark images as favorites, which rope-gui shows first; --off unmarks them
    Fav {
        #[arg(required = true)]
        ids: Vec<String>,
        #[arg(long)]
        off: bool,
    },
    /// The images an image was made from (up), or that were made from it (down)
    Lineage {
        id: String,
        #[arg(long, conflicts_with = "down")]
        up: bool,
        #[arg(long)]
        down: bool,
    },
    /// Print the command that makes an image again
    Repro {
        id: String,
        /// The full img2img.py command instead, to change and run as a variation
        #[arg(long)]
        command: bool,
    },
    /// Put images in the trash
    Rm {
        #[arg(required = true)]
        ids: Vec<String>,
    },
    /// Take images out of the trash
    Restore {
        #[arg(required = true)]
        ids: Vec<String>,
    },
    /// Delete trashed images for good. One that others were made from keeps only its bytes, until they go too.
    Purge {
        ids: Vec<String>,
        /// Everything in the trash
        #[arg(long, conflicts_with = "ids")]
        trash: bool,
        /// Purge images that are not in the trash too
        #[arg(long)]
        force: bool,
        /// Delete the bytes of images others were made from as well; those can no longer be replayed
        #[arg(long)]
        break_lineage: bool,
    },
    /// Look for problems in the library
    Check {
        /// Hash every stored file again
        #[arg(long)]
        hash: bool,
        /// Fix what is safe to fix
        #[arg(long)]
        fix: bool,
    },
    /// The latest changes, from the journal
    Log {
        id: Option<String>,
        #[arg(short, default_value_t = 20)]
        n: usize,
    },
}

#[derive(Subcommand)]
enum TagCmd {
    /// Add TAG to images
    Add {
        tag: String,
        #[arg(required = true)]
        ids: Vec<String>,
    },
    /// Remove TAG from images
    Rm {
        tag: String,
        #[arg(required = true)]
        ids: Vec<String>,
    },
    /// Rename a tag on every image
    Rename { old: String, new: String },
}

fn main() {
    let cli = Cli::parse();
    if let Err(e) = run(cli) {
        eprintln!("rope: {e:#}");
        std::process::exit(1);
    }
}

fn run(cli: Cli) -> Result<()> {
    let json = cli.json;
    if let Cmd::Init { dir } = &cli.cmd {
        let root = match dir.clone().or_else(|| cli.library.clone()) {
            Some(d) => d,
            None => Library::default_root()?,
        };
        let lib = Library::init(&root)?;
        return emit(json, json!({"library": lib.root()}), || println!("rope library at {}", lib.root().display()));
    }
    let lib = Library::locate(cli.library.as_deref())?;
    match cli.cmd {
        Cmd::Init { .. } => unreachable!(),
        Cmd::Where { folder } => {
            let id = folder.map(|f| Catalog::load(&lib)?.resolve(&f)).transpose()?;
            emit(json, json!({"library": lib.root(), "default": lib.is_default(), "folder": id.flatten()}), || {
                println!("{}", lib.root().display())
            })
        }
        Cmd::Import { paths, into, tags, note, tree, via, repo } => {
            set_repo(&lib, repo.as_deref())?;
            let new = NewImage { via, folder: into_folder(&lib, into.as_deref())?, tags, note, ..Default::default() };
            import(&lib, json, &paths, tree.then(|| into.as_deref().unwrap_or("")), &new)
        }
        Cmd::Add { file, made, graph, name, into, tags, note, file_inputs, repo } => {
            set_repo(&lib, repo.as_deref())?;
            let made = with_env_stored(&lib, read_json(&made)?)?;
            let graph = graph.map(|g| read_json(&g)).transpose()?;
            let folder = into_folder(&lib, into.as_deref())?;
            let new =
                NewImage { name, made: Some(made), graph, via: "img2img.py".into(), folder, tags, note, file_inputs };
            let added = store::add(&lib, &file, &new)?;
            let meta = store::load(&lib, &added.sha)?;
            emit(json, added_json(&lib, &meta, added.created, added.restored)?, || {
                println!("{}  {}", meta.short(), store::image_path(&lib, &meta).display())
            })
        }
        Cmd::Rendition { id, file, info } => {
            let sha = store::resolve(&lib, &id)?;
            let info: Value = serde_json::from_str(&info).context("--info is not JSON")?;
            let (r, path) = store::add_rendition(&lib, &sha, &file, info)?;
            emit(json, json!({"uploaded": r, "path": path}), || println!("{}", path.display()))
        }
        Cmd::Replayed { id, file, made, graph, repo } => {
            set_repo(&lib, repo.as_deref())?;
            replayed(&lib, json, &id, &file, &made, graph.as_deref())
        }
        Cmd::Mkdir { parents: false, path } => {
            let id = store::edit_catalog(&lib, "mkdir", &[], |c| c.mkdir(&path, false))?;
            emit(json, json!({"folder": id, "created": true}), || {})
        }
        Cmd::Mkdir { parents: true, path } => {
            let (id, created) = store::ensure_folder(&lib, &path)?;
            let id = id.context("a folder needs a name")?;
            emit(json, json!({"folder": id, "created": created}), || {})
        }
        Cmd::Mv { args } => mv(&lib, &args),
        Cmd::Rmdir { path, recursive: false } => store::edit_catalog(&lib, "rmdir", &[], |c| c.rmdir(&path)),
        Cmd::Rmdir { path, recursive: true } => {
            let (folders, trashed) = store::remove_folder_tree(&lib, &path)?;
            emit(json, json!({"folders": folders, "trashed": trashed}), || {
                println!("removed {folders} folder(s); moved {} image(s) to the trash", trashed.len())
            })
        }
        Cmd::Ls { folder, tag, trash, fav, recursive, long } => {
            ls(&lib, json, folder.as_deref(), tag.as_deref(), trash, fav, recursive, long)
        }
        Cmd::Tree => tree(&lib, json),
        Cmd::Show { id, downstream } => show(&lib, json, &id, downstream),
        Cmd::Path { id } => {
            let meta = store::load(&lib, &store::resolve(&lib, &id)?)?;
            let path = store::image_path(&lib, &meta);
            emit(json, json!({"path": path}), || println!("{}", path.display()))
        }
        Cmd::Name { id, name } => store::set_name(&lib, &store::resolve(&lib, &id)?, &name),
        Cmd::Note { id, set, append, edit } => note(&lib, json, &id, set, append, edit),
        Cmd::Tag(t) => tag(&lib, t),
        Cmd::Tags => {
            let tags = Catalog::load(&lib)?.tags();
            emit(json, json!(tags), || {
                for (t, n) in &tags {
                    println!("{n:>6}  {t}");
                }
            })
        }
        Cmd::Fav { ids, off } => {
            let shas = resolve_all(&lib, &ids)?;
            let n = store::edit_catalog(&lib, "fav", &shas, |c| {
                Ok(shas.iter().filter(|s| c.set_favorite(s, !off)).count())
            })?;
            emit(json, json!({"changed": n}), || {})
        }
        Cmd::Lineage { id, up, down } => lineage(&lib, json, &id, up, down),
        Cmd::Repro { id, command } => {
            let sha = store::resolve(&lib, &id)?;
            let meta = store::load(&lib, &sha)?;
            let argv = if command {
                let catalog = Catalog::load(&lib)?;
                let folder = catalog.entry(&sha).folder.map(|f| catalog.folder_path(&f));
                repro::command_argv(&lib, &meta, folder.as_deref())?
            } else {
                repro::replay_argv(&lib, &meta)?
            };
            emit(json, json!({"argv": argv, "command": repro::shell_join(&argv)}), || {
                println!("{}", repro::shell_join(&argv))
            })
        }
        Cmd::Rm { ids } => {
            let shas = resolve_all(&lib, &ids)?;
            let now = time::now();
            let n = store::edit_catalog(&lib, "rm", &shas, |c| Ok(shas.iter().filter(|s| c.trash(s, &now)).count()))?;
            emit(json, json!({"trashed": n}), || println!("{n} image(s) moved to the trash"))
        }
        Cmd::Restore { ids } => {
            let shas = resolve_all(&lib, &ids)?;
            let n = store::edit_catalog(&lib, "restore", &shas, |c| Ok(shas.iter().filter(|s| c.restore(s)).count()))?;
            emit(json, json!({"restored": n}), || println!("{n} image(s) restored"))
        }
        Cmd::Purge { ids, trash, force, break_lineage } => purge_cmd(&lib, json, &ids, trash, force, break_lineage),
        Cmd::Check { hash, fix } => {
            let r = check::check(&lib, hash, fix)?;
            emit(json, json!({"problems": r.problems, "fixed": r.fixed}), || {
                for p in &r.problems {
                    println!("problem: {p}");
                }
                for f in &r.fixed {
                    println!("fixed: {f}");
                }
                if r.problems.is_empty() {
                    println!("no problems");
                }
            })
        }
        Cmd::Log { id, n } => {
            let sha = id.map(|i| store::resolve(&lib, &i)).transpose()?;
            let (events, _) = journal::read_from(&lib, 0)?;
            let picked: Vec<&Value> = events
                .iter()
                .filter(|e| {
                    sha.as_ref().is_none_or(|s| e["ids"].as_array().is_some_and(|a| a.iter().any(|v| v == s.as_str())))
                })
                .collect();
            let tail = &picked[picked.len().saturating_sub(n)..];
            emit(json, json!(tail), || {
                for e in tail {
                    let ids: Vec<&str> = e["ids"]
                        .as_array()
                        .map(|a| a.iter().filter_map(Value::as_str).map(short).collect())
                        .unwrap_or_default();
                    println!(
                        "{}  {:<8} {}",
                        e["at"].as_str().unwrap_or("?"),
                        e["op"].as_str().unwrap_or("?"),
                        ids.join(" ")
                    );
                }
            })
        }
    }
}

// ---------------------------------------------------------------- commands

/// Imports `paths`. With `tree` (the rope folder to mirror into), each folder given becomes a rope folder of the
/// same name, with its subfolders; otherwise folders are walked for images, which all go where `new` says.
fn import(lib: &Library, json: bool, paths: &[PathBuf], tree: Option<&str>, new: &NewImage) -> Result<()> {
    let (dirs, files): (Vec<PathBuf>, Vec<PathBuf>) = match tree {
        Some(_) => paths.iter().cloned().partition(|p| p.is_dir()),
        None => (Vec::new(), paths.to_vec()),
    };
    let mirrored: Vec<(PathBuf, String)> = dirs.iter().flat_map(|d| store::mirror(d, tree.unwrap_or(""))).collect();
    let mut results = store::add_to_folders(lib, &mirrored, new);
    results.extend(collect_files(&files)?.into_iter().map(|f| (f.clone(), store::add(lib, &f, new))));
    let mut out = Vec::new();
    for (f, result) in results {
        let f = &f;
        let added = result.with_context(|| format!("cannot import {}", f.display()))?;
        let meta = store::load(lib, &added.sha)?;
        if !json {
            let what = if added.created {
                "new"
            } else if added.restored {
                "restored"
            } else {
                "known"
            };
            println!("{}  {what:<8} {}", meta.short(), f.display());
        }
        let mut v = added_json(lib, &meta, added.created, added.restored)?;
        v["source"] = json!(std::path::absolute(f)?);
        out.push(v);
    }
    if json {
        println!("{}", serde_json::to_string_pretty(&out)?);
    }
    Ok(())
}

/// The files named, with folders walked for images; hidden files and folders are skipped.
fn collect_files(paths: &[PathBuf]) -> Result<Vec<PathBuf>> {
    let mut out = Vec::new();
    for p in paths {
        if p.is_dir() {
            walk(p, &mut out)?;
        } else if p.is_file() {
            if !store::looks_like_image(p) && imagesize::size(p).is_err() {
                bail!("{} does not look like an image", p.display());
            }
            out.push(p.clone());
        } else {
            bail!("no such file: {}", p.display());
        }
    }
    Ok(out)
}

fn walk(dir: &Path, out: &mut Vec<PathBuf>) -> Result<()> {
    let mut entries: Vec<_> = std::fs::read_dir(dir)?.filter_map(|e| e.ok()).map(|e| e.path()).collect();
    entries.sort();
    for p in entries {
        if p.file_name().is_some_and(|n| n.to_string_lossy().starts_with('.')) {
            continue;
        }
        if p.is_dir() {
            walk(&p, out)?;
        } else if store::looks_like_image(&p) {
            out.push(p);
        }
    }
    Ok(())
}

fn replayed(lib: &Library, json: bool, id: &str, file: &Path, made: &Path, graph: Option<&Path>) -> Result<()> {
    let sha = store::resolve(lib, id)?;
    let orig = store::load(lib, &sha)?;
    let cmp = store::compare_pixels(&store::image_path(lib, &orig), file)?;
    let mut made = with_env_stored(lib, read_json(made)?)?;
    let mut record = json!({
        "at": time::now(),
        "result": if cmp.identical { "identical" } else { "differs" },
        "same_size": cmp.same_size, "max_diff": cmp.max_diff, "differing": cmp.differing,
        "env": made.get("env"), "code": made.get("code"),
    });
    if !cmp.identical {
        made["replay_of"] = json!(sha);
        let catalog = Catalog::load(lib)?;
        let new = NewImage {
            name: Some(format!("{} (replay)", orig.name)),
            made: Some(made),
            graph: graph.map(read_json).transpose()?,
            via: "img2img.py".into(),
            folder: Some(catalog.entry(&sha).folder),
            ..Default::default()
        };
        let added = store::add(lib, file, &new)?;
        record["id"] = json!(added.sha);
    }
    store::add_replay(lib, &sha, record.clone())?;
    record["path"] = match record.get("id").and_then(Value::as_str) {
        Some(new) => json!(store::image_path(lib, &store::load(lib, new)?)),
        None => json!(store::image_path(lib, &orig)),
    };
    emit(json, record.clone(), || {
        if cmp.identical {
            println!("identical to {}", orig.short());
        } else {
            println!(
                "differs from {}: max difference {} of 255 in {:.2}% of the pixels; kept as {}",
                orig.short(),
                cmp.max_diff,
                cmp.differing * 100.0,
                short(record["id"].as_str().unwrap_or("?"))
            );
        }
    })
}

fn mv(lib: &Library, args: &[String]) -> Result<()> {
    let (dest, srcs) = args.split_last().expect("at least two");
    if srcs.iter().all(|s| s.starts_with('@')) {
        let shas = resolve_all(lib, srcs)?;
        store::edit_catalog(lib, "mv", &shas, |c| {
            let folder = c.resolve(dest)?;
            for s in &shas {
                c.set_folder(s, folder.clone());
            }
            Ok(())
        })
    } else if srcs.iter().any(|s| s.starts_with('@')) {
        bail!("give either images (@id ...) and a folder, or one folder and its new path");
    } else if srcs.len() == 1 {
        store::edit_catalog(lib, "mv", &[], |c| c.move_folder(&srcs[0], dest))
    } else {
        bail!("to move images, name them as @id; to rename a folder: rope mv OLD NEW");
    }
}

#[allow(clippy::too_many_arguments)]
fn ls(
    lib: &Library,
    json: bool,
    folder: Option<&str>,
    tag: Option<&str>,
    trash: bool,
    fav: bool,
    recursive: bool,
    long: bool,
) -> Result<()> {
    let catalog = Catalog::load(lib)?;
    let mut subfolders: Vec<String> = Vec::new();
    let wanted: Vec<String> = if trash {
        catalog.items.iter().filter(|(_, e)| e.trashed.is_some()).map(|(k, _)| k.clone()).collect()
    } else if let Some(tag) = tag {
        catalog
            .items
            .iter()
            .filter(|(_, e)| e.trashed.is_none() && e.tags.iter().any(|t| t == tag))
            .map(|(k, _)| k.clone())
            .collect()
    } else {
        let id = catalog.resolve(folder.unwrap_or("/"))?;
        subfolders = catalog.children(id.as_deref()).iter().map(|(_, f)| f.name.clone()).collect();
        let scope = match (&id, recursive) {
            (Some(id), true) => Some(catalog.subtree(id)),
            (Some(id), false) => Some([id.clone()].into()),
            (None, true) => None,                      // everything
            (None, false) => Some(Default::default()), // unfiled only
        };
        match scope {
            None => store::ids(lib)?.into_iter().filter(|s| !catalog.is_trashed(s)).collect(),
            Some(set) if set.is_empty() => store::ids(lib)?
                .into_iter()
                .filter(|s| {
                    let e = catalog.entry(s);
                    e.folder.is_none() && e.trashed.is_none()
                })
                .collect(),
            Some(set) => catalog
                .items
                .iter()
                .filter(|(_, e)| e.trashed.is_none() && e.folder.as_ref().is_some_and(|f| set.contains(f)))
                .map(|(k, _)| k.clone())
                .collect(),
        }
    };
    let mut metas: Vec<Meta> = wanted
        .iter()
        .filter(|s| !fav || catalog.is_favorite(s))
        .filter_map(|s| store::load(lib, s).ok())
        .filter(|m| m.kept.is_none())
        .collect();
    metas.sort_by(|a, b| (&a.added, &a.name).cmp(&(&b.added, &b.name)));
    if json {
        let images: Vec<Value> = metas.iter().map(|m| summary_json(lib, &catalog, m)).collect();
        println!("{}", serde_json::to_string_pretty(&json!({"folders": subfolders, "images": images}))?);
        return Ok(());
    }
    for f in &subfolders {
        println!("{f}/");
    }
    for m in &metas {
        let when = m.added.get(..16).unwrap_or(&m.added).replace('T', " ");
        if long {
            let e = catalog.entry(&m.sha256);
            let size = match (m.image.width, m.image.height) {
                (Some(w), Some(h)) => format!("{w}x{h}"),
                _ => "?".into(),
            };
            let place = e.folder.map(|f| catalog.folder_path(&f)).unwrap_or_else(|| "/".into());
            let tags = if e.tags.is_empty() { String::new() } else { format!("  #{}", e.tags.join(" #")) };
            let made = if m.made.is_some() { "made" } else { "    " };
            let star = if e.favorite { "★ " } else { "" };
            println!("{}  {when}  {size:>9}  {made}  {star}{}  [{place}]{tags}", m.short(), m.name);
        } else {
            let star = if catalog.is_favorite(&m.sha256) { "★ " } else { "" };
            println!("{}  {when}  {star}{}", m.short(), m.name);
        }
    }
    Ok(())
}

fn tree(lib: &Library, json: bool) -> Result<()> {
    let catalog = Catalog::load(lib)?;
    let ids = store::ids(lib)?;
    let mut counts: HashMap<Option<String>, usize> = HashMap::new();
    let mut trashed = 0;
    for s in &ids {
        let e = catalog.entry(s);
        if e.trashed.is_some() { trashed += 1 } else { *counts.entry(e.folder).or_insert(0) += 1 }
    }
    if json {
        let folders: Vec<Value> = catalog
            .folders
            .iter()
            .map(|(id, f)| {
                json!({
                "id": id, "name": f.name, "parent": f.parent, "path": catalog.folder_path(id),
                "images": counts.get(&Some(id.clone())).copied().unwrap_or(0)})
            })
            .collect();
        let v = json!({"folders": folders, "unfiled": counts.get(&None).copied().unwrap_or(0),
                       "trash": trashed, "tags": catalog.tags()});
        println!("{}", serde_json::to_string_pretty(&v)?);
        return Ok(());
    }
    println!("/  ({} unfiled)", counts.get(&None).copied().unwrap_or(0));
    fn rec(c: &Catalog, counts: &HashMap<Option<String>, usize>, parent: Option<&str>, prefix: &str) {
        let kids = c.children(parent);
        for (i, (id, f)) in kids.iter().enumerate() {
            let last = i + 1 == kids.len();
            let n = counts.get(&Some(id.to_string())).copied().unwrap_or(0);
            println!("{prefix}{}{}  ({n})", if last { "└── " } else { "├── " }, f.name);
            rec(c, counts, Some(id), &format!("{prefix}{}", if last { "    " } else { "│   " }));
        }
    }
    rec(&catalog, &counts, None, "");
    if trashed > 0 {
        println!("trash  ({trashed})");
    }
    let tags = catalog.tags();
    if !tags.is_empty() {
        println!("tags   {}", tags.iter().map(|(t, n)| format!("#{t} ({n})")).collect::<Vec<_>>().join("  "));
    }
    Ok(())
}

fn show(lib: &Library, json: bool, id: &str, downstream: bool) -> Result<()> {
    let sha = store::resolve(lib, id)?;
    let meta = store::load(lib, &sha)?;
    let catalog = Catalog::load(lib)?;
    let entry = catalog.entry(&sha);
    let note = store::note(lib, &sha)?;
    let dir = lib.item_dir(&sha);
    let inputs: Vec<Value> = meta
        .inputs()
        .iter()
        .enumerate()
        .map(|(i, input)| {
            let found = input.id.as_ref().and_then(|id| store::load(lib, id).ok().map(|m| (id, m)));
            match found {
                Some((id, m)) => {
                    let path = match input
                        .uploaded
                        .as_ref()
                        .and_then(|u| m.uploaded.get(u))
                        .and_then(|u| u["file"].as_str())
                    {
                        Some(f) => lib.item_dir(id).join("uploaded").join(f),
                        None => store::image_path(lib, &m),
                    };
                    json!({"slot": i + 1, "id": id, "short": short(id), "name": m.name, "path": path,
                           "upload": input.upload, "uploaded": input.uploaded, "kept": m.kept.is_some()})
                }
                None => json!({"slot": i + 1, "id": input.id, "missing": true, "upload": input.upload}),
            }
        })
        .collect();
    let children: Option<Vec<String>> = if !json || downstream {
        let (metas, _) = store::scan(lib)?;
        Some(Lineage::new(&metas).children(&sha).to_vec())
    } else {
        None
    };
    if json {
        let graph = std::fs::read(dir.join("graph.json")).ok().and_then(|b| serde_json::from_slice::<Value>(&b).ok());
        let v = json!({
            "id": sha, "short": meta.short(), "dir": dir, "path": store::image_path(lib, &meta), "name": meta.name,
            "folder": entry.folder.as_ref().map(|f| catalog.folder_path(f)), "tags": entry.tags,
            "trashed": entry.trashed, "favorite": entry.favorite, "kept": meta.kept, "note": note, "inputs": inputs,
            "downstream": children, "meta": meta, "graph": graph,
        });
        println!("{}", serde_json::to_string_pretty(&v)?);
        return Ok(());
    }
    println!("{}  {}", meta.short(), meta.name);
    println!("  file     {}", store::image_path(lib, &meta).display());
    let size = match (meta.image.width, meta.image.height) {
        (Some(w), Some(h)) => format!("{w}x{h} "),
        _ => String::new(),
    };
    println!(
        "  image    {size}{} {}, added {}",
        meta.image.format.as_deref().unwrap_or("?"),
        human(meta.image.bytes),
        meta.added.replace('T', " ")
    );
    println!(
        "  folder   {}",
        entry.folder.as_ref().map(|f| catalog.folder_path(f)).unwrap_or_else(|| "/ (unfiled)".into())
    );
    if !entry.tags.is_empty() {
        println!("  tags     {}", entry.tags.join(", "));
    }
    if entry.favorite {
        println!("  favorite ★");
    }
    if let Some(t) = &entry.trashed {
        println!("  trash    since {t}");
    }
    if meta.kept.is_some() {
        println!("  kept     purged; its bytes stay because other images were made from it");
    }
    for s in &meta.sources {
        let gone = if Path::new(&s.path).exists() { "" } else { "  (gone)" };
        println!("  source   {}{gone}", s.path);
    }
    if meta.made.is_some() {
        println!("  recipe   {}", repro::shell_join(&meta.flags()));
        if let Some(p) = meta.made_str("prompt") {
            println!("  prompt   {}", p.replace('\n', "\n           "));
        }
        if let Some(p) = meta.made_str("enhanced_prompt").filter(|p| !p.is_empty()) {
            println!("  enhanced {}", p.replace('\n', "\n           "));
        }
    }
    for v in &inputs {
        let label = if v["slot"] == 1 { "up" } else { "" };
        let name = if v["missing"] == true { "(missing)".into() } else { v["name"].as_str().unwrap_or("").to_owned() };
        let id = v["id"].as_str().map(short).unwrap_or("?");
        println!("  {label:<8} {}: {id} {name}", v["slot"]);
    }
    if let Some(children) = &children {
        for (i, c) in children.iter().enumerate() {
            let name = store::load(lib, c).map(|m| m.name).unwrap_or_default();
            println!("  {:<8} {} {name}", if i == 0 { "down" } else { "" }, short(c));
        }
    }
    for r in &meta.replays {
        let extra = r.get("id").and_then(Value::as_str).map(|i| format!(" -> {}", short(i))).unwrap_or_default();
        println!("  replay   {} {}{extra}", r["at"].as_str().unwrap_or("?"), r["result"].as_str().unwrap_or("?"));
    }
    if !note.is_empty() {
        println!("  note");
        for l in note.lines() {
            println!("    {l}");
        }
    }
    Ok(())
}

fn note(lib: &Library, json: bool, id: &str, set: Option<String>, append: Option<String>, edit: bool) -> Result<()> {
    let sha = store::resolve(lib, id)?;
    if let Some(text) = set {
        return store::set_note(lib, &sha, &text);
    }
    if let Some(text) = append {
        return store::add_note(lib, &sha, &text);
    }
    if edit {
        let old = store::note(lib, &sha)?;
        let tmp = lib.tmp_dir().join(format!("note-{}.txt", short(&sha)));
        std::fs::write(&tmp, &old)?;
        let editor = std::env::var("VISUAL").or_else(|_| std::env::var("EDITOR")).unwrap_or_else(|_| "vi".into());
        let status =
            std::process::Command::new("sh").arg("-c").arg(format!("{editor} \"$1\"")).arg("sh").arg(&tmp).status()?;
        let new = std::fs::read_to_string(&tmp)?;
        let _ = std::fs::remove_file(&tmp);
        if !status.success() {
            bail!("{editor} failed; the note is unchanged");
        }
        if new != old {
            store::set_note(lib, &sha, &new)?;
        }
        return Ok(());
    }
    let text = store::note(lib, &sha)?;
    emit(json, json!({"note": text}), || print!("{text}"))
}

fn tag(lib: &Library, cmd: TagCmd) -> Result<()> {
    match cmd {
        TagCmd::Add { tag, ids } => {
            let shas = resolve_all(lib, &ids)?;
            store::edit_catalog(lib, "tag", &shas, |c| shas.iter().try_for_each(|s| c.add_tag(s, &tag).map(|_| ())))
        }
        TagCmd::Rm { tag, ids } => {
            let shas = resolve_all(lib, &ids)?;
            store::edit_catalog(lib, "tag", &shas, |c| {
                shas.iter().for_each(|s| {
                    c.remove_tag(s, &tag);
                });
                Ok(())
            })
        }
        TagCmd::Rename { old, new } => {
            let n = store::edit_catalog(lib, "tag", &[], |c| c.rename_tag(&old, &new))?;
            if n == 0 {
                bail!("no image has the tag {old:?}");
            }
            Ok(())
        }
    }
}

fn lineage(lib: &Library, json: bool, id: &str, up: bool, down: bool) -> Result<()> {
    let sha = store::resolve(lib, id)?;
    let (metas, _) = store::scan(lib)?;
    let names: BTreeMap<&str, &str> = metas.iter().map(|m| (m.sha256.as_str(), m.name.as_str())).collect();
    let l = Lineage::new(&metas);
    let dirs: Vec<bool> = match (up, down) {
        (true, _) => vec![true],
        (_, true) => vec![false],
        _ => vec![true, false],
    };
    if json {
        let mut v = json!({"id": sha});
        for &u in &dirs {
            let rows: Vec<Value> =
                l.walk(&sha, u).into_iter().map(|(d, i, r)| json!({"depth": d, "id": i, "repeat": r})).collect();
            v[if u { "up" } else { "down" }] = json!(rows);
        }
        println!("{}", serde_json::to_string_pretty(&v)?);
        return Ok(());
    }
    println!("{}  {}", short(&sha), names.get(sha.as_str()).unwrap_or(&""));
    for &u in &dirs {
        let rows = l.walk(&sha, u);
        if rows.is_empty() {
            println!("  {} nothing", if u { "↑" } else { "↓" });
        }
        for (depth, i, repeat) in rows {
            let name = names.get(i.as_str()).copied().unwrap_or("(not in the library)");
            let again = if repeat { "  (listed above)" } else { "" };
            println!("{}{} {}  {name}{again}", "  ".repeat(depth), if u { "↑" } else { "↓" }, short(&i));
        }
    }
    Ok(())
}

fn purge_cmd(lib: &Library, json: bool, ids: &[String], trash: bool, force: bool, break_lineage: bool) -> Result<()> {
    let catalog = Catalog::load(lib)?;
    let shas: Vec<String> = if trash {
        catalog.items.iter().filter(|(_, e)| e.trashed.is_some()).map(|(k, _)| k.clone()).collect()
    } else if ids.is_empty() {
        bail!("name the images to purge, or pass --trash to empty the trash");
    } else {
        let shas = resolve_all(lib, ids)?;
        if !force && let Some(s) = shas.iter().find(|s| !catalog.is_trashed(s)) {
            bail!("{} is not in the trash; rope rm it first, or pass --force", short(s));
        }
        shas
    };
    if shas.is_empty() {
        return emit(json, json!({"deleted": [], "kept": [], "in_use": []}), || println!("the trash is empty"));
    }
    let out = purge::purge(lib, &shas, break_lineage)?;
    let kept: Vec<Value> = out.kept.iter().map(|(k, f)| json!({"id": k, "for": f})).collect();
    let in_use: Vec<Value> = out.in_use.iter().map(|(k, pid)| json!({"id": k, "pid": pid})).collect();
    emit(json, json!({"deleted": out.deleted, "kept": kept, "in_use": in_use}), || {
        println!("deleted {} image(s)", out.deleted.len());
        for (k, f) in &out.kept {
            println!(
                "kept the bytes of {}: {} image(s) were made from it (rope purge --break-lineage to delete it anyway)",
                short(k),
                f.len()
            );
        }
        for (k, pid) in &out.in_use {
            println!(
                "left {} alone: the img2img.py run with pid {pid} uses it; purge it when that run is done",
                short(k)
            );
        }
    })
}

// ---------------------------------------------------------------- helpers

/// Prints `v` as JSON with --json, else runs `text`.
fn emit(json: bool, v: Value, text: impl FnOnce()) -> Result<()> {
    if json {
        println!("{}", serde_json::to_string_pretty(&v)?);
    } else {
        text();
    }
    Ok(())
}

fn added_json(lib: &Library, meta: &Meta, created: bool, restored: bool) -> Result<Value> {
    let catalog = Catalog::load(lib)?;
    Ok(json!({
        "id": meta.sha256, "short": meta.short(), "path": store::image_path(lib, meta), "name": meta.name,
        "created": created, "restored": restored, "pixels": meta.image.pixels,
        "folder": catalog.entry(&meta.sha256).folder.map(|f| catalog.folder_path(&f)),
    }))
}

fn summary_json(lib: &Library, catalog: &Catalog, m: &Meta) -> Value {
    let e = catalog.entry(&m.sha256);
    json!({"id": m.sha256, "short": m.short(), "name": m.name, "added": m.added, "path": store::image_path(lib, m),
           "width": m.image.width, "height": m.image.height, "made": m.made.is_some(),
           "folder": e.folder.map(|f| catalog.folder_path(&f)), "tags": e.tags, "trashed": e.trashed,
           "favorite": e.favorite})
}

/// `--into FOLDER` as the folder to file into, made if there is none: `None` without the flag.
fn into_folder(lib: &Library, into: Option<&str>) -> Result<Option<Option<String>>> {
    into.map(|f| store::ensure_folder(lib, f).map(|(id, _)| id)).transpose()
}

fn resolve_all(lib: &Library, ids: &[String]) -> Result<Vec<String>> {
    let mut out = Vec::new();
    for i in ids {
        let s = store::resolve(lib, i)?;
        if !out.contains(&s) {
            out.push(s);
        }
    }
    Ok(out)
}

fn read_json(path: &Path) -> Result<Value> {
    let mut text = String::new();
    if path == Path::new("-") {
        std::io::stdin().read_to_string(&mut text)?;
    } else {
        text = std::fs::read_to_string(path).with_context(|| format!("cannot read {}", path.display()))?;
    }
    serde_json::from_str(&text).with_context(|| format!("{} is not JSON", path.display()))
}

/// A recipe whose `env` is a whole snapshot gets it stored once in envs/, and refers to it by id.
fn with_env_stored(lib: &Library, mut made: Value) -> Result<Value> {
    if made.get("env").is_some_and(Value::is_object) {
        let id = store::add_env(lib, &made["env"])?;
        made["env"] = json!(id);
    }
    Ok(made)
}

fn set_repo(lib: &Library, repo: Option<&Path>) -> Result<()> {
    match repo {
        Some(r) => lib.set_repo(r),
        None => Ok(()),
    }
}

fn human(bytes: u64) -> String {
    let b = bytes as f64;
    if b >= 1e9 {
        format!("{:.1} GB", b / 1e9)
    } else if b >= 1e6 {
        format!("{:.1} MB", b / 1e6)
    } else if b >= 1e3 {
        format!("{:.0} kB", b / 1e3)
    } else {
        format!("{bytes} B")
    }
}
