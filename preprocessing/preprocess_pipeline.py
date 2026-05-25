"""
preprocessing/preprocess_pipeline.py

Full preprocessing pipeline for brain shift project.
Steps:
  1. Load NIfTI MRI + US volumes
  2. Skull stripping (MRI)
  3. Bias field correction (MRI)
  4. Rigid registration: T1w -> T2-FLAIR
  5. Intensity normalisation
  6. Resample to isotropic 1mm resolution
  7. Save processed volumes + metadata JSON
"""
import os
import json
import argparse
import numpy as np
import nibabel as nib
import SimpleITK as sitk
from pathlib import Path
from tqdm import tqdm


# ─── UTILITIES ─────────────────────────────────────────────────────────────

def load_nifti(path: str) -> sitk.Image:
    return sitk.ReadImage(str(path), sitk.sitkFloat32)


def save_nifti(image: sitk.Image, path: str):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    sitk.WriteImage(image, str(path))


def resample_to_spacing(image: sitk.Image, new_spacing=(1.0, 1.0, 1.0),
                         interpolator=sitk.sitkLinear) -> sitk.Image:
    """Resample image to isotropic spacing."""
    original_spacing = image.GetSpacing()
    original_size = image.GetSize()
    new_size = [
        int(round(osz * osp / nsp))
        for osz, osp, nsp in zip(original_size, original_spacing, new_spacing)
    ]
    resampler = sitk.ResampleImageFilter()
    resampler.SetOutputSpacing(new_spacing)
    resampler.SetSize(new_size)
    resampler.SetOutputDirection(image.GetDirection())
    resampler.SetOutputOrigin(image.GetOrigin())
    resampler.SetTransform(sitk.Transform())
    resampler.SetDefaultPixelValue(0)
    resampler.SetInterpolator(interpolator)
    return resampler.Execute(image)


def normalise_intensity(image: sitk.Image, lower_percentile=1.0,
                         upper_percentile=99.0) -> sitk.Image:
    """Clip and normalise intensity to [0, 1]."""
    arr = sitk.GetArrayFromImage(image).astype(np.float32)
    # Only compute percentiles on non-zero (brain) voxels
    nonzero = arr[arr > 0]
    if len(nonzero) == 0:
        return image
    lo = np.percentile(nonzero, lower_percentile)
    hi = np.percentile(nonzero, upper_percentile)
    arr = np.clip(arr, lo, hi)
    arr = (arr - lo) / (hi - lo + 1e-8)
    out = sitk.GetImageFromArray(arr)
    out.CopyInformation(image)
    return out


def bias_field_correction(image: sitk.Image) -> sitk.Image:
    """N4 bias field correction for MRI."""
    corrector = sitk.N4BiasFieldCorrectionImageFilter()
    corrector.SetMaximumNumberOfIterations([50, 50, 30, 20])
    mask = sitk.OtsuThreshold(image, 0, 1, 200)
    return corrector.Execute(image, mask)


def skull_strip_simple(image: sitk.Image, threshold: float = 0.1) -> sitk.Image:
    """
    Simple Otsu-based skull stripping.
    For production: use HD-BET or antspynet brain extraction instead.
    """
    arr = sitk.GetArrayFromImage(image)
    # Normalise first
    lo, hi = arr.min(), arr.max()
    norm = (arr - lo) / (hi - lo + 1e-8)

    # Otsu threshold
    from skimage.filters import threshold_otsu
    from scipy.ndimage import binary_fill_holes, label, binary_erosion, binary_dilation
    thresh = threshold_otsu(norm[norm > 0.05]) if (norm > 0.05).sum() > 0 else 0.3
    mask = norm > thresh * 0.6  # slightly permissive

    # Keep only largest connected component
    labeled, n_comp = label(mask)
    if n_comp > 0:
        sizes = [(labeled == i).sum() for i in range(1, n_comp+1)]
        largest = np.argmax(sizes) + 1
        mask = labeled == largest

    mask = binary_fill_holes(mask)
    mask = binary_dilation(mask, iterations=3)

    arr_stripped = arr * mask.astype(arr.dtype)
    out = sitk.GetImageFromArray(arr_stripped)
    out.CopyInformation(image)
    return out


def rigid_register(fixed: sitk.Image, moving: sitk.Image) -> tuple:
    """
    Rigid registration of moving to fixed image.
    Returns (registered_image, transform)
    """
    # Initialise using image centres
    initial_transform = sitk.CenteredTransformInitializer(
        fixed, moving,
        sitk.Euler3DTransform(),
        sitk.CenteredTransformInitializerFilter.GEOMETRY
    )
    registration_method = sitk.ImageRegistrationMethod()

    # Metric: Mutual Information (good for MRI-MRI and MRI-US)
    registration_method.SetMetricAsMattesMutualInformation(numberOfHistogramBins=50)
    registration_method.SetMetricSamplingStrategy(registration_method.RANDOM)
    registration_method.SetMetricSamplingPercentage(0.01)

    registration_method.SetInterpolator(sitk.sitkLinear)

    # Multi-resolution
    registration_method.SetOptimizerAsGradientDescent(
        learningRate=1.0, numberOfIterations=200,
        convergenceMinimumValue=1e-6, convergenceWindowSize=10
    )
    registration_method.SetOptimizerScalesFromPhysicalShift()
    registration_method.SetShrinkFactorsPerLevel([4, 2, 1])
    registration_method.SetSmoothingSigmasPerLevel([2, 1, 0])
    registration_method.SmoothingSigmasAreSpecifiedInPhysicalUnitsOn()
    registration_method.SetInitialTransform(initial_transform, inPlace=False)

    final_transform = registration_method.Execute(
        sitk.Cast(fixed, sitk.sitkFloat32),
        sitk.Cast(moving, sitk.sitkFloat32)
    )

    # Apply transform
    resampler = sitk.ResampleImageFilter()
    resampler.SetReferenceImage(fixed)
    resampler.SetInterpolator(sitk.sitkLinear)
    resampler.SetDefaultPixelValue(0)
    resampler.SetTransform(final_transform)
    registered = resampler.Execute(moving)

    return registered, final_transform


# ─── MAIN PIPELINE ─────────────────────────────────────────────────────────

def preprocess_case(case_dir: Path, output_dir: Path,
                    target_spacing=(1.0, 1.0, 1.0),
                    do_bias_correction=True,
                    do_skull_strip=True):
    """
    Process one patient case.
    Expected input structure (after MINC->NIfTI conversion):
      case_dir/
        CaseXX-T1.nii.gz
        CaseXX-FLAIR.nii.gz
        CaseXX-US-before.nii.gz
        CaseXX-US-during.nii.gz
        CaseXX-US-after.nii.gz        (may not always exist)
    """
    case_id = case_dir.name
    out_case = output_dir / case_id
    out_case.mkdir(parents=True, exist_ok=True)

    meta = {"case_id": case_id, "steps": []}

    # Find files
    t1_files = list(case_dir.glob("*T1*.nii*")) + list(case_dir.glob("*t1*.nii*"))
    flair_files = list(case_dir.glob("*FLAIR*.nii*")) + list(case_dir.glob("*flair*.nii*"))
    us_before_files = list(case_dir.glob("*US-before*.nii*")) + list(case_dir.glob("*before*.nii*"))
    us_during_files = list(case_dir.glob("*US-during*.nii*")) + list(case_dir.glob("*during*.nii*"))
    us_after_files  = list(case_dir.glob("*US-after*.nii*"))  + list(case_dir.glob("*after*.nii*"))

    if not t1_files or not flair_files:
        print(f"  [SKIP] {case_id}: missing T1 or FLAIR")
        return None

    t1_path    = t1_files[0]
    flair_path = flair_files[0]

    # ── Load ──
    t1    = load_nifti(t1_path)
    flair = load_nifti(flair_path)

    # ── Resample to isotropic ──
    t1    = resample_to_spacing(t1, target_spacing)
    flair = resample_to_spacing(flair, target_spacing)
    meta["steps"].append("resample_1mm")

    # ── Bias field correction (MRI only) ──
    if do_bias_correction:
        t1    = bias_field_correction(t1)
        flair = bias_field_correction(flair)
        meta["steps"].append("n4_bias_correction")

    # ── Skull strip T1 ──
    if do_skull_strip:
        t1_stripped = skull_strip_simple(t1)
        meta["steps"].append("skull_strip")
    else:
        t1_stripped = t1

    # ── Register T1 -> FLAIR space ──
    t1_reg, t1_to_flair_tfm = rigid_register(fixed=flair, moving=t1_stripped)
    meta["steps"].append("rigid_register_t1_to_flair")

    # ── Normalise intensities ──
    t1_norm    = normalise_intensity(t1_reg)
    flair_norm = normalise_intensity(flair)
    meta["steps"].append("intensity_normalise")

    # ── Save MRI ──
    save_nifti(t1_norm,    str(out_case / "T1_processed.nii.gz"))
    save_nifti(flair_norm, str(out_case / "FLAIR_processed.nii.gz"))

    # ── Process US volumes ──
    for us_files, tag in [
        (us_before_files, "US_before"),
        (us_during_files, "US_during"),
        (us_after_files,  "US_after")
    ]:
        if not us_files:
            continue
        us = load_nifti(us_files[0])
        us = resample_to_spacing(us, target_spacing, interpolator=sitk.sitkLinear)
        us_norm = normalise_intensity(us, lower_percentile=0, upper_percentile=99.5)
        save_nifti(us_norm, str(out_case / f"{tag}_processed.nii.gz"))
        meta["steps"].append(f"process_{tag}")

    # ── Save metadata ──
    with open(out_case / "metadata.json", "w") as f:
        json.dump(meta, f, indent=2)

    print(f"  [OK] {case_id}")
    return meta


def run_pipeline(input_dir: str, output_dir: str, **kwargs):
    input_path  = Path(input_dir)
    output_path = Path(output_dir)

    case_dirs = sorted([d for d in input_path.iterdir() if d.is_dir()])
    print(f"Processing {len(case_dirs)} cases from {input_path}")

    results = []
    for case_dir in tqdm(case_dirs, desc="Cases"):
        try:
            meta = preprocess_case(case_dir, output_path, **kwargs)
            if meta:
                results.append(meta)
        except Exception as e:
            print(f"  [ERROR] {case_dir.name}: {e}")

    summary = {"total": len(case_dirs), "processed": len(results)}
    with open(output_path / "preprocessing_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nPipeline complete: {len(results)}/{len(case_dirs)} cases processed.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input",  required=True, help="Directory of NIfTI cases")
    parser.add_argument("--output", required=True, help="Output processed directory")
    parser.add_argument("--no-bias",  action="store_true", help="Skip bias correction")
    parser.add_argument("--no-strip", action="store_true", help="Skip skull stripping")
    parser.add_argument("--spacing", nargs=3, type=float, default=[1.0, 1.0, 1.0])
    args = parser.parse_args()

    run_pipeline(
        args.input, args.output,
        target_spacing=tuple(args.spacing),
        do_bias_correction=not args.no_bias,
        do_skull_strip=not args.no_strip
    )
