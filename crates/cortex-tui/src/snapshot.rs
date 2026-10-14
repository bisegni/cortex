//! Export / import of the learned brain (`weights/cortex-0`) as timestamped snapshots.
use std::{io, path::{Path, PathBuf}};

const LIVE: &str = "weights/cortex-0";
const SNAPSHOTS: &str = "weights/snapshots";

fn copy_files(from: &Path, to: &Path) -> io::Result<usize> {
    std::fs::create_dir_all(to)?;
    let mut n = 0;
    for entry in std::fs::read_dir(from)? {
        let path = entry?.path();
        if path.is_file() && path.extension().is_some_and(|e| e != "tmp") {
            std::fs::copy(&path, to.join(path.file_name().unwrap_or_default()))?;
            n += 1;
        }
    }
    Ok(n)
}

/// Copy the live weights into `weights/snapshots/<unix seconds>/`.
pub fn export(root: &Path) -> io::Result<PathBuf> {
    let ts = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap_or_default().as_secs();
    let dest = root.join(SNAPSHOTS).join(ts.to_string());
    if copy_files(&root.join(LIVE), &dest)? == 0 { return Err(io::Error::other("nothing to export yet")); }
    Ok(dest)
}

/// Replace the live weights with the newest snapshot (the nodes must be restarted to load it).
pub fn import_latest(root: &Path) -> io::Result<PathBuf> {
    let mut snaps: Vec<_> = std::fs::read_dir(root.join(SNAPSHOTS))?.filter_map(|e| e.ok().map(|e| e.path())).filter(|p| p.is_dir()).collect();
    snaps.sort();
    let latest = snaps.pop().ok_or_else(|| io::Error::other("no snapshots"))?;
    copy_files(&latest, &root.join(LIVE))?;
    Ok(latest)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn export_then_import_restores_files() {
        let root = std::env::temp_dir().join(format!("cortex-snap-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&root);
        std::fs::create_dir_all(root.join(LIVE)).unwrap();
        std::fs::write(root.join(LIVE).join("links.json"), "v1").unwrap();
        std::fs::write(root.join(LIVE).join("x.bin.tmp"), "partial").unwrap();
        let snap = export(&root).unwrap();
        assert!(snap.join("links.json").exists() && !snap.join("x.bin.tmp").exists());
        std::fs::write(root.join(LIVE).join("links.json"), "v2").unwrap();
        import_latest(&root).unwrap();
        assert_eq!(std::fs::read_to_string(root.join(LIVE).join("links.json")).unwrap(), "v1");
        let _ = std::fs::remove_dir_all(&root);
    }

    #[test]
    fn export_without_weights_fails_cleanly() {
        let root = std::env::temp_dir().join(format!("cortex-snap-empty-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&root);
        std::fs::create_dir_all(root.join(LIVE)).unwrap();
        assert!(export(&root).is_err());
        let _ = std::fs::remove_dir_all(&root);
    }
}
