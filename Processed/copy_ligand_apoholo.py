"""
Copies ligand.pdb files from source to destination folders in the ApoHolo dataset based on folder prefixes.
"""

import os
import shutil


# TODO: Replace with environment variables or command-line arguments
SRC_ROOT = r"F:\workspace\pycharm\ProtGeoNet-Pocket\Datasets\full_datasets"
DST_ROOT = r"F:\workspace\pycharm\ProtGeoNet-Pocket\Datasets\ApoHolo\apo"


def get_base(name: str) -> str:
    """Extract folder prefix: before underscore or first 4 dot-separated parts."""
    if '_' in name:
        return name.split('_', 1)[0]
    parts = name.split('.')
    return '.'.join(parts[:4]) if len(parts) >= 4 else name


def copy_ligand_files() -> int:
    """Copy ligand.pdb files from source to matching destination folders."""
    dst_prefix_map = {
        get_base(d): os.path.join(DST_ROOT, d)
        for d in os.listdir(DST_ROOT)
        if os.path.isdir(os.path.join(DST_ROOT, d))
    }

    copied_count = 0
    for root, _, files in os.walk(SRC_ROOT):
        if 'ligand.pdb' not in files:
            continue

        src_prefix = get_base(os.path.basename(root))
        dst_folder = dst_prefix_map.get(src_prefix)
        if dst_folder:
            src_file = os.path.join(root, 'ligand.pdb')
            dst_file = os.path.join(dst_folder, 'ligand.pdb')
            shutil.copy2(src_file, dst_file)
            copied_count += 1
            print(f"Copied {src_file} -> {dst_file}")
        else:
            print(f"Warning: No destination folder found for prefix '{src_prefix}'")

    return copied_count


def main() -> None:
    """Execute the ligand file copying process."""
    copied_count = copy_ligand_files()
    print(f"\nCopied {copied_count} ligand.pdb files.")


if __name__ == "__main__":
    main()