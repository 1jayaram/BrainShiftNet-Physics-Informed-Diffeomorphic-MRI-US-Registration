"""
scripts/convert_minc_to_nifti.py
Converts RESECT MINC2 files to NIfTI format for easier handling.
Requires: minc-tools system package + pyminc
"""
import os
import argparse
import subprocess
from pathlib import Path
from tqdm import tqdm


def convert_minc_to_nifti(input_dir: str, output_dir: str):
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    mnc_files = list(input_path.rglob("*.mnc"))
    print(f"Found {len(mnc_files)} MINC files")

    failed = []
    for mnc_file in tqdm(mnc_files, desc="Converting"):
        # Mirror directory structure
        rel_path = mnc_file.relative_to(input_path)
        out_file = output_path / rel_path.with_suffix(".nii.gz")
        out_file.parent.mkdir(parents=True, exist_ok=True)

        if out_file.exists():
            continue

        # Use nibabel to convert
        try:
            import nibabel as nib
            img = nib.load(str(mnc_file))
            nib.save(img, str(out_file))
        except Exception as e:
            print(f"  ERROR: {mnc_file.name}: {e}")
            failed.append(mnc_file)

    print(f"\nDone. Converted {len(mnc_files)-len(failed)}/{len(mnc_files)} files.")
    if failed:
        print(f"Failed: {[f.name for f in failed]}")


def convert_tag_to_csv(input_dir: str, output_dir: str):
    """Convert MNI .tag landmark files to CSV for easier loading."""
    import re
    input_path = Path(input_dir)
    output_path = Path(output_dir)

    tag_files = list(input_path.rglob("*.tag"))
    print(f"Found {len(tag_files)} tag files")

    for tag_file in tag_files:
        rel = tag_file.relative_to(input_path)
        out_csv = output_path / rel.with_suffix(".csv")
        out_csv.parent.mkdir(parents=True, exist_ok=True)

        landmarks_1 = []
        landmarks_2 = []
        with open(tag_file, "r") as f:
            content = f.read()

        # Parse MNI tag file format
        point_block = re.search(r"Points = (.*?);", content, re.DOTALL)
        if not point_block:
            continue
        points_str = point_block.group(1).strip()
        lines = [l.strip() for l in points_str.split("\n") if l.strip()]

        rows = []
        for line in lines:
            nums = re.findall(r"[-+]?\d*\.?\d+", line)
            if len(nums) >= 6:
                rows.append({
                    "x1": float(nums[0]), "y1": float(nums[1]), "z1": float(nums[2]),
                    "x2": float(nums[3]), "y2": float(nums[4]), "z2": float(nums[5])
                })
            elif len(nums) >= 3:
                rows.append({
                    "x1": float(nums[0]), "y1": float(nums[1]), "z1": float(nums[2]),
                    "x2": None, "y2": None, "z2": None
                })

        import csv
        with open(out_csv, "w", newline="") as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=["x1","y1","z1","x2","y2","z2"])
            writer.writeheader()
            writer.writerows(rows)

    print("Tag files converted.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert RESECT MINC to NIfTI")
    parser.add_argument("--input", required=True, help="Input RESECT directory")
    parser.add_argument("--output", required=True, help="Output NIfTI directory")
    parser.add_argument("--tags", action="store_true", help="Also convert .tag landmark files")
    args = parser.parse_args()

    convert_minc_to_nifti(args.input, args.output)
    if args.tags:
        convert_tag_to_csv(args.input, args.output)
