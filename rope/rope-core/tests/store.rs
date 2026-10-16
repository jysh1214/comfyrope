use rope_core::store::{self, NewImage};
use rope_core::{Catalog, Library, purge};
use serde_json::json;
use std::fs;
use std::path::{Path, PathBuf};

fn library() -> (tempfile::TempDir, Library) {
    let dir = tempfile::tempdir().unwrap();
    let lib = Library::init(&dir.path().join("lib")).unwrap();
    (dir, lib)
}

/// A small PNG whose pixels depend on `seed`.
fn png(dir: &Path, name: &str, seed: u8) -> PathBuf {
    let img = image::RgbImage::from_fn(4, 3, |x, y| image::Rgb([seed, x as u8, y as u8]));
    let path = dir.join(name);
    img.save(&path).unwrap();
    path
}

fn import(lib: &Library, path: &Path) -> String {
    store::add(lib, path, &NewImage { via: "import".into(), ..Default::default() }).unwrap().sha
}

fn made_from(lib: &Library, path: &Path, inputs: &[&str]) -> String {
    let inputs: Vec<_> = inputs.iter().map(|id| json!({"id": id})).collect();
    let new = NewImage {
        via: "img2img.py".into(),
        made: Some(json!({"flags": ["--model=2511", "--seed=1"], "prompt": "p", "inputs": inputs})),
        graph: Some(json!({"1": {"class_type": "LoadImage", "inputs": {}}})),
        ..Default::default()
    };
    store::add(lib, path, &new).unwrap().sha
}

#[test]
fn the_same_bytes_from_two_places_are_one_image_with_two_sources() {
    let (dir, lib) = library();
    let a = png(dir.path(), "a.png", 1);
    let copy = dir.path().join("copy.png");
    fs::copy(&a, &copy).unwrap();
    let first = store::add(&lib, &a, &NewImage { via: "import".into(), ..Default::default() }).unwrap();
    let second = store::add(&lib, &copy, &NewImage { via: "import".into(), ..Default::default() }).unwrap();
    assert!(first.created && !second.created);
    assert_eq!(first.sha, second.sha);
    let meta = store::load(&lib, &first.sha).unwrap();
    assert_eq!(meta.sources.len(), 2);
    assert_eq!((meta.image.width, meta.image.height), (Some(4), Some(3)));
    assert_eq!(meta.name, "a");
    let file = store::image_path(&lib, &meta);
    assert_eq!(fs::read(&file).unwrap(), fs::read(&a).unwrap());
    assert!(fs::metadata(&file).unwrap().permissions().readonly());
    assert!(fs::read_dir(lib.tmp_dir()).unwrap().next().is_none(), "tmp/ is left empty");
}

#[test]
fn a_file_brought_in_from_inside_the_library_adds_no_source() {
    let (dir, lib) = library();
    let id = import(&lib, &png(dir.path(), "a.png", 1));
    let inside = store::image_path(&lib, &store::load(&lib, &id).unwrap());
    import(&lib, &inside);
    assert_eq!(store::load(&lib, &id).unwrap().sources.len(), 1);
}

#[test]
fn a_made_image_gets_its_recipe_files_and_pixel_hash() {
    let (dir, lib) = library();
    let input = import(&lib, &png(dir.path(), "in.png", 1));
    let out = made_from(&lib, &png(dir.path(), "out.png", 2), &[&input]);
    let item = lib.item_dir(&out);
    assert!(item.join("graph.json").exists());
    let script = fs::read_to_string(item.join("repro.sh")).unwrap();
    assert!(script.contains("--replay"));
    let meta = store::load(&lib, &out).unwrap();
    assert_eq!(meta.upstream(), [input]);
    assert_eq!(
        meta.image.pixels.as_deref(),
        Some(store::pixel_hash(&store::image_path(&lib, &meta)).unwrap().as_str())
    );
}

#[test]
fn folder_tags_and_note_are_attached_on_the_way_in_and_inputs_follow_their_result() {
    let (dir, lib) = library();
    let input = import(&lib, &png(dir.path(), "in.png", 1));
    let folder = store::edit_catalog(&lib, "mkdir", &[], |c| c.mkdir("trips/kyoto", true)).unwrap();
    let new = NewImage {
        via: "img2img.py".into(),
        made: Some(json!({"inputs": [{"id": input}]})),
        folder: Some(Some(folder.clone())),
        tags: vec!["海邊".into()],
        note: Some("first try".into()),
        file_inputs: true,
        ..Default::default()
    };
    let out = store::add(&lib, &png(dir.path(), "out.png", 2), &new).unwrap().sha;
    let c = Catalog::load(&lib).unwrap();
    assert_eq!(c.entry(&out).folder.as_deref(), Some(folder.as_str()));
    assert_eq!(c.entry(&out).tags, ["海邊"]);
    assert_eq!(c.entry(&input).folder.as_deref(), Some(folder.as_str()), "the unfiled input joins its result");
    assert_eq!(store::note(&lib, &out).unwrap(), "first try\n");
}

#[test]
fn ensure_folder_makes_a_missing_folder_once_and_leaves_the_top_alone() {
    let (_dir, lib) = library();
    let (id, created) = store::ensure_folder(&lib, "trips/kyoto").unwrap();
    let id = id.expect("a folder, not the top");
    assert!(created);
    assert_eq!(store::ensure_folder(&lib, "trips/kyoto").unwrap(), (Some(id.clone()), false));
    assert_eq!(store::ensure_folder(&lib, "/").unwrap(), (None, false));
    let c = Catalog::load(&lib).unwrap();
    assert_eq!(c.folder_path(&id), "trips/kyoto");
    assert_eq!(c.find("trips"), Some(Some(c.folders[&id].parent.clone().unwrap())));
    assert_eq!(fs::read_to_string(lib.journal_path()).unwrap().matches("mkdir").count(), 1);
}

#[test]
fn bringing_in_a_trashed_image_takes_it_out_of_the_trash() {
    let (dir, lib) = library();
    let p = png(dir.path(), "a.png", 1);
    let id = import(&lib, &p);
    store::edit_catalog(&lib, "rm", std::slice::from_ref(&id), |c| Ok(c.trash(&id, "now"))).unwrap();
    let again = store::add(&lib, &p, &NewImage { via: "import".into(), ..Default::default() }).unwrap();
    assert!(again.restored);
    assert!(!Catalog::load(&lib).unwrap().is_trashed(&id));
}

#[test]
fn purge_keeps_the_bytes_of_an_input_until_its_last_descendant_goes() {
    let (dir, lib) = library();
    let a = import(&lib, &png(dir.path(), "a.png", 1));
    store::set_note(&lib, &a, "private").unwrap();
    let b = made_from(&lib, &png(dir.path(), "b.png", 2), &[&a]);
    let c = made_from(&lib, &png(dir.path(), "c.png", 3), &[&a, &b]);

    let out = purge::purge(&lib, std::slice::from_ref(&a), false).unwrap();
    assert_eq!(out.kept, [(a.clone(), vec![b.clone(), c.clone()])]);
    let kept = store::load(&lib, &a).unwrap();
    assert!(kept.kept.is_some() && kept.name.is_empty() && kept.sources.is_empty());
    assert!(store::image_path(&lib, &kept).exists());
    assert!(!lib.item_dir(&a).join("note.txt").exists());

    purge::purge(&lib, std::slice::from_ref(&b), false).unwrap();
    assert!(lib.item_dir(&a).exists(), "c still needs a");
    let out = purge::purge(&lib, std::slice::from_ref(&c), false).unwrap();
    assert!(out.deleted.contains(&a) && out.deleted.contains(&c));
    assert!(store::ids(&lib).unwrap().is_empty());
}

#[test]
fn purge_with_break_lineage_deletes_an_input_all_the_same() {
    let (dir, lib) = library();
    let a = import(&lib, &png(dir.path(), "a.png", 1));
    let b = made_from(&lib, &png(dir.path(), "b.png", 2), &[&a]);
    purge::purge(&lib, std::slice::from_ref(&a), true).unwrap();
    assert!(!lib.item_dir(&a).exists());
    assert!(lib.item_dir(&b).exists());
}

#[test]
fn purge_waits_while_a_run_brings_its_inputs_in() {
    let (dir, lib) = library();
    let a = import(&lib, &png(dir.path(), "a.png", 1));
    let run = fs::File::create(lib.root().join("run.lock")).unwrap();
    run.lock_shared().unwrap();
    assert!(purge::purge(&lib, std::slice::from_ref(&a), false).is_err());
    drop(run);
    purge::purge(&lib, &[a], false).unwrap();
}

#[test]
fn purge_leaves_the_inputs_of_a_running_img2img_alone_and_deletes_the_rest() {
    let (dir, lib) = library();
    let a = import(&lib, &png(dir.path(), "a.png", 1));
    let b = import(&lib, &png(dir.path(), "b.png", 2));
    let path = lib.runs_dir().join("777.json");
    fs::write(&path, format!(r#"{{"pid": 777, "inputs": ["{a}"]}}"#)).unwrap();
    let held = fs::File::open(&path).unwrap();
    held.lock().unwrap();
    let out = purge::purge(&lib, &[a.clone(), b.clone()], false).unwrap();
    assert_eq!(out.deleted, std::slice::from_ref(&b));
    assert_eq!(out.in_use, [(a.clone(), 777)]);
    assert!(lib.item_dir(&a).exists() && !lib.item_dir(&b).exists());
    drop(held);
    let out = purge::purge(&lib, std::slice::from_ref(&a), false).unwrap();
    assert_eq!((out.deleted, out.in_use), (vec![a.clone()], vec![]));
    assert!(!path.exists(), "the file of the run that ended is gone");
}

#[test]
fn resolve_takes_prefixes_files_in_the_library_and_files_elsewhere_with_the_same_bytes() {
    let (dir, lib) = library();
    let p = png(dir.path(), "a.png", 1);
    let id = import(&lib, &p);
    assert_eq!(store::resolve(&lib, &id[..6]).unwrap(), id);
    assert_eq!(store::resolve(&lib, &format!("@{}", &id[..12])).unwrap(), id);
    assert!(store::resolve(&lib, &id[..4]).is_err(), "too short");
    let inside = store::image_path(&lib, &store::load(&lib, &id).unwrap());
    assert_eq!(store::resolve(&lib, inside.to_str().unwrap()).unwrap(), id);
    assert_eq!(store::resolve(&lib, p.to_str().unwrap()).unwrap(), id);
    let other = png(dir.path(), "b.png", 9);
    assert!(store::resolve(&lib, other.to_str().unwrap()).unwrap_err().to_string().contains("rope import"));
}

#[test]
fn a_rendition_is_stored_with_its_image() {
    let (dir, lib) = library();
    let id = import(&lib, &png(dir.path(), "a.png", 1));
    let conv = png(dir.path(), "conv.png", 5);
    let (r, path) = store::add_rendition(&lib, &id, &conv, json!({"steps": "first frame"})).unwrap();
    let meta = store::load(&lib, &id).unwrap();
    assert_eq!(meta.uploaded[&r]["steps"], "first frame");
    assert_eq!(path, lib.item_dir(&id).join("uploaded").join(format!("{r}.png")));
    assert!(path.exists());
    assert_eq!(store::add_rendition(&lib, &id, &conv, json!({})).unwrap().0, r, "the same file again is a no-op");
}

#[test]
fn concurrent_writers_lose_no_update() {
    let (dir, lib) = library();
    let folder = store::edit_catalog(&lib, "mkdir", &[], |c| c.mkdir("f", false)).unwrap();
    let files: Vec<PathBuf> = (0..32).map(|i| png(dir.path(), &format!("{i}.png"), i as u8)).collect();
    std::thread::scope(|s| {
        for chunk in files.chunks(4) {
            let (lib, folder) = (lib.clone(), folder.clone());
            s.spawn(move || {
                for f in chunk {
                    let new = NewImage {
                        via: "import".into(),
                        folder: Some(Some(folder.clone())),
                        tags: vec!["t".into()],
                        ..Default::default()
                    };
                    store::add(&lib, f, &new).unwrap();
                }
            });
        }
    });
    let c = Catalog::load(&lib).unwrap();
    assert_eq!(c.items.len(), 32);
    assert_eq!(c.tags()["t"], 32);
    assert_eq!(store::ids(&lib).unwrap().len(), 32);
}

#[test]
fn compare_pixels_reports_identical_and_how_different() {
    let (dir, _lib) = library();
    let a = png(dir.path(), "a.png", 1);
    let b = png(dir.path(), "b.png", 4);
    assert!(store::compare_pixels(&a, &a).unwrap().identical);
    let c = store::compare_pixels(&a, &b).unwrap();
    assert!(!c.identical && c.same_size && c.max_diff == 3 && (c.differing - 1.0).abs() < 1e-9);
}

#[test]
fn a_disk_folder_becomes_a_rope_folder_with_its_subfolders() {
    let (dir, lib) = library();
    let disk = dir.path().join("京都 2026");
    fs::create_dir_all(disk.join("day 1/morning")).unwrap();
    fs::create_dir_all(disk.join(".hidden")).unwrap();
    fs::create_dir_all(disk.join("empty")).unwrap();
    png(&disk, "top.png", 1);
    png(&disk.join("day 1"), "a.png", 2);
    png(&disk.join("day 1/morning"), "b.png", 3);
    png(&disk.join(".hidden"), "c.png", 4);
    fs::write(disk.join("notes.txt"), "not an image").unwrap();

    let files = store::mirror(&disk, "");
    let places: Vec<&str> = files.iter().map(|(_, f)| f.as_str()).collect();
    assert_eq!(places, ["京都 2026/day 1", "京都 2026/day 1/morning", "京都 2026"]);
    assert_eq!(store::mirror(&disk, "/trips/")[0].1, "trips/京都 2026/day 1");

    let results = store::add_to_folders(&lib, &files, &NewImage { via: "gui".into(), ..Default::default() });
    assert!(results.iter().all(|(_, r)| r.as_ref().is_ok_and(|a| a.created)));
    let c = Catalog::load(&lib).unwrap();
    let paths: Vec<String> = {
        let mut v: Vec<_> = c.folders.keys().map(|id| c.folder_path(id)).collect();
        v.sort();
        v
    };
    assert_eq!(
        paths,
        ["京都 2026", "京都 2026/day 1", "京都 2026/day 1/morning"],
        "no folder for empty or hidden ones"
    );
    let morning = c.find("京都 2026/day 1/morning").unwrap();
    assert_eq!(c.items.values().filter(|e| e.folder == morning).count(), 1);

    // importing the same folder again merges into it
    let again =
        store::add_to_folders(&lib, &store::mirror(&disk, ""), &NewImage { via: "gui".into(), ..Default::default() });
    assert!(again.iter().all(|(_, r)| r.as_ref().is_ok_and(|a| !a.created)));
    assert_eq!(Catalog::load(&lib).unwrap().folders.len(), 3);
}

/// A one-second test clip, 64x48 at 16 fps, made with the ffmpeg on the system; None without one.
fn clip(dir: &Path, name: &str) -> Option<PathBuf> {
    let path = dir.join(name);
    let made = std::process::Command::new("ffmpeg")
        .args([
            "-v",
            "error",
            "-y",
            "-f",
            "lavfi",
            "-i",
            "testsrc=size=64x48:rate=16",
            "-t",
            "1",
            "-pix_fmt",
            "yuv420p",
        ])
        .arg(&path)
        .status()
        .is_ok_and(|s| s.success());
    made.then_some(path)
}

#[test]
fn a_clip_is_kept_with_its_length_and_its_first_frame_and_check_fix_restores_them() {
    let (dir, lib) = library();
    let Some(path) = clip(dir.path(), "clip.mp4") else {
        eprintln!("no ffmpeg here: the clip test is skipped");
        return;
    };
    assert!(store::looks_like_video(&path) && !store::looks_like_image(&path) && store::looks_like_media(&path));
    let sha = made_from(&lib, &path, &[]);
    let meta = store::load(&lib, &sha).unwrap();
    assert!(meta.is_clip() && meta.image.pixels.is_none(), "a clip has no pixel hash");
    assert_eq!((meta.image.width, meta.image.height, meta.image.format.as_deref()), (Some(64), Some(48), Some("mp4")));
    let v = meta.video.clone().unwrap();
    assert_eq!((v.fps, v.frames, v.codec.as_deref()), (Some(16.0), Some(16), Some("h264")));
    assert!((v.duration.unwrap() - 1.0).abs() < 0.1, "{v:?}");
    let poster = lib.item_dir(&sha).join("poster.png");
    assert_eq!(meta.poster.as_deref(), Some("poster.png"));
    let frame = image::open(&poster).unwrap();
    assert_eq!((frame.width(), frame.height()), (64, 48));

    // the poster gone: check says so, and --fix makes it again
    fs::remove_file(&poster).unwrap();
    let report = rope_core::check::check(&lib, false, false).unwrap();
    assert!(report.problems.iter().any(|p| p.contains("no poster frame")), "{:?}", report.problems);
    let report = rope_core::check::check(&lib, false, true).unwrap();
    assert!(report.fixed.iter().any(|f| f.contains("poster")), "{:?}", report.fixed);
    assert!(poster.exists() && store::load(&lib, &sha).unwrap().poster.is_some());
    assert!(rope_core::check::check(&lib, false, false).unwrap().problems.is_empty());

    // a clip kept by an older rope as an image, with no length: --fix reads it
    let item = lib.item_dir(&sha);
    let mut json: serde_json::Value = serde_json::from_slice(&fs::read(item.join("meta.json")).unwrap()).unwrap();
    json.as_object_mut().unwrap().remove("video");
    json["image"]["format"] = json!("heif");
    fs::write(item.join("meta.json"), serde_json::to_vec_pretty(&json).unwrap()).unwrap();
    let report = rope_core::check::check(&lib, false, true).unwrap();
    assert!(report.fixed.iter().any(|f| f.contains("length")), "{:?}", report.fixed);
    let meta = store::load(&lib, &sha).unwrap();
    assert_eq!(meta.image.format.as_deref(), Some("mp4"));
    assert_eq!(meta.video.and_then(|v| v.frames), Some(16));
}
