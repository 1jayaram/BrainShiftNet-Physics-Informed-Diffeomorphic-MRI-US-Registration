"""
data/dataset.py
PyTorch Dataset classes for RESECT and ReMIND databases.
"""
import os
import json
import re
import numpy as np
import torch
from torch.utils.data import Dataset
from pathlib import Path
import nibabel as nib
import SimpleITK as sitk
from typing import Optional, List, Tuple, Dict


def load_volume(path: str, target_shape: Optional[Tuple] = None) -> np.ndarray:
    """Load NIfTI volume as numpy array, optionally resizing."""
    img = nib.load(path)
    data = img.get_fdata(dtype=np.float32)
    if target_shape is not None:
        from scipy.ndimage import zoom
        factors = [t / s for t, s in zip(target_shape, data.shape)]
        data = zoom(data, factors, order=1)
    return data


def pad_or_crop(volume: np.ndarray, target_shape: Tuple[int,...]) -> np.ndarray:
    """Pad or centre-crop a volume to target_shape."""
    result = np.zeros(target_shape, dtype=volume.dtype)
    # Compute crop/pad slices
    slices_in  = []
    slices_out = []
    for dim_in, dim_out in zip(volume.shape, target_shape):
        if dim_in >= dim_out:
            start = (dim_in - dim_out) // 2
            slices_in.append(slice(start, start + dim_out))
            slices_out.append(slice(0, dim_out))
        else:
            start = (dim_out - dim_in) // 2
            slices_in.append(slice(0, dim_in))
            slices_out.append(slice(start, start + dim_in))
    result[tuple(slices_out)] = volume[tuple(slices_in)]
    return result


class BrainShiftDataset(Dataset):
    """
    Dataset for brain shift prediction.
    Each sample:
      - fixed: pre-operative MRI (T2-FLAIR)  — the "anchor"
      - moving: also MRI or concatenated T1+FLAIR
      - target: intra-operative US (US_before — right after dura opening)
      - landmarks: (N, 6) array of paired landmark coordinates [x1,y1,z1,x2,y2,z2]
    
    The model learns to predict the deformation field that maps
    the pre-op MRI to the intra-op US space.
    """

    def __init__(
        self,
        data_dir: str,
        split: str = "train",           # "train" | "val" | "test"
        split_file: Optional[str] = None,
        target_shape: Tuple[int,...] = (160, 192, 160),
        use_t1: bool = True,
        use_flair: bool = True,
        augment: bool = False,
        us_stage: str = "before",       # "before" | "during" | "after"
        segmentation_dir: Optional[str] = None,  # Path to RESECT segmentation directory
        landmark_dir: Optional[str] = None,      # Path to RESECT landmark .tag files
    ):
        self.data_dir     = Path(data_dir)
        self.split        = split
        self.target_shape = target_shape
        self.use_t1       = use_t1
        self.use_flair    = use_flair
        self.augment      = augment
        self.us_stage     = us_stage
        self.seg_dir      = Path(segmentation_dir) if segmentation_dir else None
        self.landmark_dir = Path(landmark_dir) if landmark_dir else self._default_landmark_dir()

        # Load case list
        if split_file and Path(split_file).exists():
            import csv
            with open(split_file) as f:
                reader = csv.DictReader(f)
                self.cases = [row["case_id"] for row in reader if row["split"] == split]
        else:
            # Auto-discover from directory
            all_cases = sorted([d.name for d in self.data_dir.iterdir() if d.is_dir()])
            n = len(all_cases)
            if split == "train":
                self.cases = all_cases[:int(0.7 * n)]
            elif split == "val":
                self.cases = all_cases[int(0.7 * n):int(0.85 * n)]
            else:
                self.cases = all_cases[int(0.85 * n):]

        print(f"[Dataset] {split}: {len(self.cases)} cases from {data_dir}")

    def __len__(self) -> int:
        return len(self.cases)

    def _default_landmark_dir(self) -> Optional[Path]:
        candidates = [
            Path("EASY-RESECT (1)") / "EASY-RESECT" / "landmarks" / "Coordinates",
            Path("EASY-RESECT") / "landmarks" / "Coordinates",
        ]
        for candidate in candidates:
            if candidate.exists():
                return candidate
        return None

    @staticmethod
    def _map_voxel_to_target(coord: np.ndarray,
                             original_shape: Tuple[int, ...],
                             target_shape: Tuple[int, ...]) -> np.ndarray:
        mapped = coord.astype(np.float32).copy()
        for axis, (dim_in, dim_out) in enumerate(zip(original_shape, target_shape)):
            if dim_in >= dim_out:
                mapped[axis] -= (dim_in - dim_out) / 2.0
            else:
                mapped[axis] += (dim_out - dim_in) / 2.0
        return mapped

    @staticmethod
    def _world_to_voxel(coord: np.ndarray,
                        affine: np.ndarray,
                        original_shape: Tuple[int, ...],
                        target_shape: Tuple[int, ...]) -> np.ndarray:
        hom = np.append(coord.astype(np.float32), 1.0)
        voxel = np.linalg.inv(affine) @ hom
        return BrainShiftDataset._map_voxel_to_target(voxel[:3], original_shape, target_shape)

    @staticmethod
    def _parse_tag_file(tag_path: Path) -> Optional[np.ndarray]:
        with open(tag_path) as f:
            content = f.read()

        point_block = re.search(r"Points\s*=\s*(.*?);", content, re.DOTALL)
        if not point_block:
            return None

        rows = []
        for line in point_block.group(1).splitlines():
            nums = re.findall(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", line)
            if len(nums) >= 6:
                rows.append([float(v) for v in nums[:6]])
        return np.array(rows, dtype=np.float32) if rows else None

    def _load_landmarks(self, case_dir: Path, case_id: str) -> Optional[np.ndarray]:
        """Load paired landmarks and convert MNI/world coordinates to target voxels."""
        landmark_files = list(case_dir.glob("*.csv"))
        raw_landmarks = None
        if landmark_files:
            import csv
            rows = []
            with open(landmark_files[0]) as f:
                reader = csv.DictReader(f)
                for row in reader:
                    try:
                        rows.append([float(row[k]) for k in ["x1","y1","z1","x2","y2","z2"]])
                    except (ValueError, KeyError):
                        pass
            raw_landmarks = np.array(rows, dtype=np.float32) if rows else None
        elif self.landmark_dir and self.landmark_dir.exists():
            tag_files = sorted(self.landmark_dir.glob(f"{case_id}-MRI-*US.tag"))
            if tag_files:
                raw_landmarks = self._parse_tag_file(tag_files[0])

        if raw_landmarks is None:
            return None

        t1_path = case_dir / "T1_processed.nii.gz"
        us_path = case_dir / f"US_{self.us_stage}_processed.nii.gz"
        if not t1_path.exists() or not us_path.exists():
            return raw_landmarks

        mri_img = nib.load(str(t1_path))
        us_img = nib.load(str(us_path))
        converted = []
        for row in raw_landmarks:
            src = self._world_to_voxel(row[:3], mri_img.affine, mri_img.shape, self.target_shape)
            tgt = self._world_to_voxel(row[3:], us_img.affine, us_img.shape, self.target_shape)
            if (np.all(src >= 0) and np.all(src < np.array(self.target_shape)) and
                    np.all(tgt >= 0) and np.all(tgt < np.array(self.target_shape))):
                converted.append(np.concatenate([src, tgt]))
        return np.array(converted, dtype=np.float32) if converted else None

    def _load_segmentation(self, case_id: str) -> Optional[Dict[str, np.ndarray]]:
        """Load tumor and resection segmentation masks from RESECT_segmentation directory."""
        if not self.seg_dir or not self.seg_dir.exists():
            return None
        
        case_seg_dir = self.seg_dir / case_id
        if not case_seg_dir.exists():
            return None
        
        segs = {}
        
        # Try to load tumor mask
        tumor_files = list(case_seg_dir.glob(f"*{self.us_stage}*tumor*.nii.gz"))
        if tumor_files:
            tumor = load_volume(str(tumor_files[0]))
            tumor = pad_or_crop(tumor, self.target_shape)
            segs["tumor"] = tumor
        
        # Try to load resection mask
        resection_files = list(case_seg_dir.glob(f"*{self.us_stage}*resection*.nii.gz"))
        if resection_files:
            resection = load_volume(str(resection_files[0]))
            resection = pad_or_crop(resection, self.target_shape)
            segs["resection"] = resection
        
        return segs if segs else None

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        case_id  = self.cases[idx]
        case_dir = self.data_dir / case_id

        sample = {"case_id": case_id}

        # ── Load MRI channels ──
        mri_channels = []
        if self.use_t1:
            t1_path = case_dir / "T1_processed.nii.gz"
            if t1_path.exists():
                t1 = load_volume(str(t1_path))
                t1 = pad_or_crop(t1, self.target_shape)
                mri_channels.append(t1)

        if self.use_flair:
            flair_path = case_dir / "FLAIR_processed.nii.gz"
            if flair_path.exists():
                flair = load_volume(str(flair_path))
                flair = pad_or_crop(flair, self.target_shape)
                mri_channels.append(flair)

        if not mri_channels:
            raise FileNotFoundError(f"No MRI found for {case_id}")

        mri = np.stack(mri_channels, axis=0)  # (C, D, H, W)
        sample["mri"] = torch.from_numpy(mri).float()

        # ── Load intra-operative US ──
        us_path = case_dir / f"US_{self.us_stage}_processed.nii.gz"
        if us_path.exists():
            us = load_volume(str(us_path))
            us = pad_or_crop(us, self.target_shape)
            sample["us"] = torch.from_numpy(us[np.newaxis]).float()

        # ── Load landmarks ──
        lm = self._load_landmarks(case_dir, case_id)
        if lm is not None:
            sample["landmarks"] = torch.from_numpy(lm).float()

        # ── Load segmentation masks ──
        segs = self._load_segmentation(case_id)
        if segs:
            if "tumor" in segs:
                sample["tumor_seg"] = torch.from_numpy(segs["tumor"][np.newaxis]).float()
            if "resection" in segs:
                sample["resection_seg"] = torch.from_numpy(segs["resection"][np.newaxis]).float()

        # ── Augmentation ──
        if self.augment and self.split == "train":
            sample = self._augment(sample)

        return sample

    def _augment(self, sample: Dict) -> Dict:
        """Random flips and small intensity jitter."""
        # Random left-right flip
        if np.random.rand() > 0.5:
            for key in ["mri", "us"]:
                if key in sample:
                    sample[key] = torch.flip(sample[key], dims=[-1])

        # Intensity jitter on MRI
        if "mri" in sample:
            noise = torch.randn_like(sample["mri"]) * 0.02
            sample["mri"] = (sample["mri"] + noise).clamp(0, 1)

        return sample


class ResectDataset(BrainShiftDataset):
    """RESECT-specific dataset with landmark support."""
    pass


class RemindDataset(Dataset):
    """
    ReMIND dataset (TCIA).
    Data is in DICOM format — use pydicom or MONAI DICOMReader.
    """
    def __init__(self, data_dir: str, split: str = "train",
                 target_shape=(160, 192, 160)):
        self.data_dir     = Path(data_dir)
        self.target_shape = target_shape
        self.split        = split

        all_cases = sorted([d for d in self.data_dir.iterdir() if d.is_dir()])
        n = len(all_cases)
        if split == "train":
            self.cases = all_cases[:int(0.7 * n)]
        elif split == "val":
            self.cases = all_cases[int(0.7 * n):int(0.85 * n)]
        else:
            self.cases = all_cases[int(0.85 * n):]

        print(f"[ReMIND] {split}: {len(self.cases)} cases")

    def __len__(self):
        return len(self.cases)

    def __getitem__(self, idx):
        import monai.transforms as mt
        case_dir = self.cases[idx]
        # ReMIND has NIFTI exports after conversion
        sample = {}
        for fname, key in [
            ("*T1*.nii.gz", "mri"),
            ("*US*.nii.gz", "us"),
            ("*tumor*.nrrd", "seg")
        ]:
            files = list(case_dir.glob(fname))
            if files:
                vol = load_volume(str(files[0]))
                vol = pad_or_crop(vol, self.target_shape)
                sample[key] = torch.from_numpy(vol[np.newaxis]).float()
        return sample


def create_loocv_splits(data_dir: str, output_file: str, fold: int):
    """
    Create Leave-One-Out Cross Validation splits.
    For RESECT (23 cases): one patient is test, rest are train.
    """
    import csv
    data_path = Path(data_dir)
    cases = sorted([d.name for d in data_path.iterdir() if d.is_dir()])
    n = len(cases)
    assert 0 <= fold < n, f"fold must be 0..{n-1}"

    with open(output_file, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["case_id", "split"])
        writer.writeheader()
        for i, case in enumerate(cases):
            if i == fold:
                split = "test"
            elif i == (fold - 1) % n:
                split = "val"
            else:
                split = "train"
            writer.writerow({"case_id": case, "split": split})

    print(f"Saved LOOCV split (fold {fold}/{n}) -> {output_file}")
