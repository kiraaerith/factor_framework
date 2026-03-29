"""
Move etf_factor_framework to factor_framework, with backup and restore support.

Usage:
    python move_factor_framework.py move      # Move and create backup
    python move_factor_framework.py restore   # Restore from backup
"""

import sys
import shutil
import json
import os
from pathlib import Path
from datetime import datetime

# Script lives in factor_framework/, so parent = code_project_v2
SCRIPT_DIR = Path(__file__).resolve().parent       # factor_framework
PROJECT_ROOT = SCRIPT_DIR.parent                   # code_project_v2
SRC = PROJECT_ROOT / "etf_cross_ml-master" / "etf_factor_framework"
DST = SCRIPT_DIR / "etf_factor_framework"
BACKUP_META_DIR = SCRIPT_DIR / "backup_meta_data"
BACKUP_META = BACKUP_META_DIR / "factor_framework_move_backup.json"


def confirm(prompt: str) -> bool:
    answer = input(f"{prompt} (Yes/No): ").strip()
    return answer == "Yes"


def do_move():
    if not SRC.exists():
        print(f"[ERROR] Source not found: {SRC}")
        sys.exit(1)

    # Count files for info
    file_count = sum(1 for _ in SRC.rglob("*") if _.is_file())
    print(f"Source: {SRC}")
    print(f"Target: {DST}")
    print(f"Files to copy: {file_count}")
    if DST.exists():
        print(f"[WARN] Target already exists, will be overwritten.")

    if not confirm("Proceed with move?"):
        print("Aborted.")
        sys.exit(0)

    # Step 1: Copy to target
    if DST.exists():
        shutil.rmtree(DST)
    print("Copying files...")
    shutil.copytree(SRC, DST)
    print(f"Copied to {DST}")

    # Step 2: Save backup metadata
    BACKUP_META_DIR.mkdir(parents=True, exist_ok=True)
    meta = {
        "moved_at": datetime.now().isoformat(),
        "source": str(SRC),
        "target": str(DST),
        "file_count": file_count,
    }
    BACKUP_META.write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Backup metadata saved to {BACKUP_META}")

    print("Move completed successfully. Original source kept.")


def do_restore():
    if not BACKUP_META.exists():
        print(f"[ERROR] No backup metadata found at {BACKUP_META}")
        print("Cannot restore without prior move record.")
        sys.exit(1)

    meta = json.loads(BACKUP_META.read_text(encoding="utf-8"))
    src = Path(meta["source"])
    dst = Path(meta["target"])

    print(f"Restore from: {dst}")
    print(f"Restore to:   {src}")
    print(f"Original move time: {meta['moved_at']}")

    if not dst.exists():
        print(f"[ERROR] Target directory not found: {dst}")
        sys.exit(1)

    if src.exists():
        print(f"[WARN] Source directory already exists: {src}")
        if not confirm("Overwrite existing source directory?"):
            print("Aborted.")
            sys.exit(0)
        shutil.rmtree(src)

    if not confirm("Proceed with restore?"):
        print("Aborted.")
        sys.exit(0)

    # Copy back
    print("Restoring files...")
    shutil.copytree(dst, src)
    print(f"Restored to {src}")

    # Remove target
    if not confirm("Delete the target directory (factor_framework)?"):
        print("Target kept. Restore completed (copy only).")
        return

    shutil.rmtree(dst)
    print(f"Target removed: {dst}")

    # Clean up metadata
    BACKUP_META.unlink()
    print("Backup metadata cleaned up.")
    print("Restore completed successfully.")


def main():
    if len(sys.argv) != 2 or sys.argv[1] not in ("move", "restore"):
        print(__doc__.strip())
        sys.exit(1)

    action = sys.argv[1]
    if action == "move":
        do_move()
    elif action == "restore":
        do_restore()


if __name__ == "__main__":
    main()
