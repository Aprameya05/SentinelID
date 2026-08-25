"""
Face alignment and landmark extraction preprocessing script.

Processes raw dataset frames and saves .npz files for:
    - Behavioral biometrics (AU-GNN, gaze)
    - Deepfake detection (aligned face crops)
    - Liveness (aligned face crops + optional depth maps)

Uses MediaPipe Face Mesh for 468-point landmarks, converted to 68-point
subset to match the AU-GNN graph topology.

Outputs per frame:
    landmarks:      (68, 2) float32  normalized to [0, 1]
    eye_crop:       (3, 64, 64) float32 normalized
    au_intensities: (12,) float32 zeros (filled externally from AU annotation files)
    face_crop:      (3, 224, 224) float32 normalized (for deepfake/liveness)

Usage:
    python scripts/preprocess_faces.py \\
        --dataset disfa \\
        --raw-root data/raw/disfa \\
        --out-root data/processed/disfa \\
        --workers 8

    python scripts/preprocess_faces.py \\
        --dataset faceforensics \\
        --raw-root data/raw/faceforensics \\
        --out-root data/raw/faceforensics \\
        --workers 8
"""

import argparse
import os
import sys
import multiprocessing as mp
from pathlib import Path
from typing import Optional
import numpy as np
import cv2
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TimeElapsedColumn, BarColumn, TaskProgressColumn

console = Console()

# MediaPipe landmark index -> 68-point dlib-compatible index mapping
# (approximate subset that covers the 68 standard face points)
MP_TO_68 = [
    162, 234, 93, 58, 172, 136, 149, 148, 152, 377,
    378, 365, 397, 288, 323, 454, 389, 71, 63, 105,
    66, 107, 336, 296, 334, 293, 301, 168, 197, 5,
    4, 75, 97, 2, 326, 305, 33, 160, 158, 133,
    153, 144, 362, 385, 387, 263, 373, 380, 61, 39,
    37, 0, 267, 269, 291, 405, 314, 17, 84, 181,
    91, 146, 76, 185, 40, 38, 87, 178,
]

FACE_CROP_SIZE = 224
EYE_CROP_SIZE = 64
N_LANDMARKS = 68


def get_mediapipe():
    """Lazy import mediapipe to avoid import-time errors if not installed."""
    try:
        import mediapipe as mp
        return mp.solutions.face_mesh
    except ImportError:
        console.print("[red]mediapipe not installed. Run: pip install mediapipe[/red]")
        sys.exit(1)


def get_dlib():
    try:
        import dlib
        return dlib
    except ImportError:
        return None


def detect_and_align(
    frame: np.ndarray,
    face_mesh,
    target_size: int = FACE_CROP_SIZE,
) -> Optional[dict]:
    """
    Detect face, extract 68 landmarks and aligned crops.

    Returns dict with: face_crop, eye_crop, landmarks, or None if no face found.
    """
    H, W = frame.shape[:2]
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = face_mesh.process(rgb)

    if not result.multi_face_landmarks:
        return None

    face_lms = result.multi_face_landmarks[0].landmark

    # Extract 68 subset
    pts = np.array([[face_lms[i].x, face_lms[i].y] for i in MP_TO_68], dtype=np.float32)
    # pts are normalized [0,1]; convert to pixel coords for cropping
    pts_px = pts * np.array([W, H])

    # Bounding box with 20% margin
    x_min, y_min = pts_px.min(axis=0)
    x_max, y_max = pts_px.max(axis=0)
    margin_x = (x_max - x_min) * 0.2
    margin_y = (y_max - y_min) * 0.2
    x0 = max(0, int(x_min - margin_x))
    y0 = max(0, int(y_min - margin_y))
    x1 = min(W, int(x_max + margin_x))
    y1 = min(H, int(y_max + margin_y))

    face_bgr = frame[y0:y1, x0:x1]
    if face_bgr.size == 0:
        return None

    face_rgb = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2RGB)
    face_crop = cv2.resize(face_rgb, (target_size, target_size)).astype(np.float32) / 255.0
    # (H, W, 3) -> (3, H, W), normalize
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    face_crop = ((face_crop - mean) / std).transpose(2, 0, 1)

    # Eye crop — use eye landmark centroid
    # Left eye: landmarks 36-41 (in 68-pt scheme, indices 36-41 in MP_TO_68)
    left_eye_pts = pts_px[36:42]
    right_eye_pts = pts_px[42:48]
    all_eye_pts = np.vstack([left_eye_pts, right_eye_pts])
    ex0 = max(0, int(all_eye_pts[:, 0].min()) - 10)
    ey0 = max(0, int(all_eye_pts[:, 1].min()) - 10)
    ex1 = min(W, int(all_eye_pts[:, 0].max()) + 10)
    ey1 = min(H, int(all_eye_pts[:, 1].max()) + 10)

    eye_bgr = frame[ey0:ey1, ex0:ex1]
    if eye_bgr.size == 0:
        eye_crop = np.zeros((3, EYE_CROP_SIZE, EYE_CROP_SIZE), dtype=np.float32)
    else:
        eye_rgb = cv2.cvtColor(eye_bgr, cv2.COLOR_BGR2RGB)
        eye_resized = cv2.resize(eye_rgb, (EYE_CROP_SIZE, EYE_CROP_SIZE)).astype(np.float32) / 255.0
        eye_crop = ((eye_resized - mean) / std).transpose(2, 0, 1)

    return {
        "face_crop": face_crop.astype(np.float32),
        "eye_crop": eye_crop.astype(np.float32),
        "landmarks": pts.astype(np.float32),  # normalized [0,1]
    }


def process_frame(args: tuple) -> Optional[Path]:
    """Worker function: process one frame and save .npz."""
    frame_path, out_path, au_intensities, gaze_vector = args

    face_mesh_module = get_mediapipe()
    with face_mesh_module.FaceMesh(
        static_image_mode=True,
        max_num_faces=1,
        refine_landmarks=False,
        min_detection_confidence=0.5,
    ) as face_mesh:
        frame = cv2.imread(str(frame_path))
        if frame is None:
            return None

        result = detect_and_align(frame, face_mesh)
        if result is None:
            return None

        out_path.parent.mkdir(parents=True, exist_ok=True)
        save_dict = {
            "landmarks": result["landmarks"],
            "face_crop": result["face_crop"],
            "eye_crop": result["eye_crop"],
            "au_intensities": au_intensities if au_intensities is not None
                              else np.zeros(12, dtype=np.float32),
        }
        if gaze_vector is not None:
            save_dict["gaze_vector"] = gaze_vector

        np.savez_compressed(str(out_path), **save_dict)
        return out_path


def collect_disfa_frames(raw_root: Path) -> list[tuple[Path, Path, np.ndarray, None]]:
    """
    Collect DISFA frames with AU annotations.

    DISFA structure:
        raw_root/
            VideoData/  SN001/  SN001_C.avi  (or extracted frames in SN001/)
            ActionUnit/ SN001/  SN001_au1.txt  (intensity per frame)
    """
    samples = []
    video_dir = raw_root / "VideoData"
    au_dir = raw_root / "ActionUnit"
    au_indices = [1, 2, 4, 5, 6, 9, 12, 15, 17, 20, 25, 26]

    if not video_dir.exists():
        console.print(f"[red]DISFA VideoData not found at {video_dir}[/red]")
        return samples

    for subj_dir in sorted(video_dir.iterdir()):
        subj = subj_dir.name
        # Load AU annotations
        au_seqs = {}
        for au in au_indices:
            au_file = au_dir / subj / f"{subj}_au{au}.txt"
            if au_file.exists():
                with open(au_file) as f:
                    au_seqs[au] = [float(l.strip().split(",")[-1]) for l in f if l.strip()]
            else:
                au_seqs[au] = []

        # Collect frame paths
        frame_files = sorted(subj_dir.glob("*.jpg")) + sorted(subj_dir.glob("*.png"))
        for i, fp in enumerate(frame_files):
            aus = np.array([
                au_seqs[au][i] if i < len(au_seqs[au]) else 0.0
                for au in au_indices
            ], dtype=np.float32)
            out_path = raw_root.parent / "processed" / "disfa" / subj / fp.with_suffix(".npz").name
            samples.append((fp, out_path, aus, None))

    return samples


def collect_generic_frames(
    raw_root: Path,
    out_root: Path,
    extensions: tuple = ("*.jpg", "*.png"),
) -> list[tuple[Path, Path, None, None]]:
    """Generic frame collector for datasets without AU annotations."""
    samples = []
    for ext in extensions:
        for fp in sorted(raw_root.rglob(ext)):
            rel = fp.relative_to(raw_root)
            out_path = out_root / rel.with_suffix(".npz")
            samples.append((fp, out_path, None, None))
    return samples


def run_preprocessing(
    samples: list[tuple],
    workers: int = 4,
    skip_existing: bool = True,
) -> int:
    if skip_existing:
        samples = [(fp, op, au, gaze) for fp, op, au, gaze in samples if not op.exists()]

    if not samples:
        console.print("[dim]All frames already processed.[/dim]")
        return 0

    console.print(f"Processing {len(samples):,} frames with {workers} workers...")
    success = 0

    with Progress(
        SpinnerColumn(),
        BarColumn(),
        TaskProgressColumn(),
        TimeElapsedColumn(),
    ) as prog:
        task = prog.add_task("Preprocessing", total=len(samples))

        if workers <= 1:
            for args in samples:
                result = process_frame(args)
                if result:
                    success += 1
                prog.advance(task)
        else:
            with mp.Pool(workers) as pool:
                for result in pool.imap_unordered(process_frame, samples, chunksize=8):
                    if result:
                        success += 1
                    prog.advance(task)

    return success


def main():
    parser = argparse.ArgumentParser(description="SentinelID face preprocessing")
    parser.add_argument(
        "--dataset",
        required=True,
        choices=["disfa", "faceforensics", "celebdf", "nuaa", "siw", "generic"],
    )
    parser.add_argument("--raw-root", required=True, help="Path to raw dataset")
    parser.add_argument("--out-root", required=True, help="Output directory for .npz files")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--skip-existing", action="store_true", default=True)
    args = parser.parse_args()

    raw_root = Path(args.raw_root)
    out_root = Path(args.out_root)

    if args.dataset == "disfa":
        samples = collect_disfa_frames(raw_root)
    else:
        samples = collect_generic_frames(raw_root, out_root)

    if not samples:
        console.print("[red]No frames found. Check --raw-root path.[/red]")
        sys.exit(1)

    n = run_preprocessing(samples, workers=args.workers, skip_existing=args.skip_existing)
    console.print(f"[green]Done. {n:,} frames saved to {out_root}[/green]")


if __name__ == "__main__":
    main()
