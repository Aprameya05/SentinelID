"""
Dataset loaders for behavioral biometrics (AU + gaze).

Supported:
    DISFADataset      — Denver Intensity of Spontaneous Facial Action
    BP4DDataset       — BP4D spontaneous (22 AUs, posed + spontaneous)
    MPIIGazeDataset   — MPIIGaze (15 subjects, gaze estimation)
    ETHXGazeDataset   — ETH-XGaze (high-resolution gaze)

All behavioral datasets return preprocessed .npz files with:
    landmarks: (68, 2) normalized to [0, 1]
    au_intensities: (n_aus,) in [0, 5]
    eye_crop: (3, 64, 64)
    gaze_vector: (3,) unit vector — only if dataset provides it

Run scripts/preprocess_faces.py first to generate .npz from raw frames.
"""

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

DISFA_AUS = [1, 2, 4, 5, 6, 9, 12, 15, 17, 20, 25, 26]
BP4D_AUS = [1, 2, 4, 6, 7, 10, 12, 14, 15, 17, 23, 24, 25, 26, 28, 43]


class DISFADataset(Dataset):
    """
    DISFA dataset — preprocessed .npz frames.

    Expected: root/  subject_id/  frame_*.npz

    Each .npz must have: landmarks (68,2), au_intensities (12,), eye_crop (3,64,64).
    Generate with: python scripts/preprocess_faces.py --dataset disfa --root <raw_root>
    """

    def __init__(self, root: str, n_aus: int = 12):
        self.root = Path(root)
        self.n_aus = n_aus
        self.files = sorted(self.root.rglob("*.npz"))

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, idx: int):
        data = np.load(self.files[idx], allow_pickle=False)
        landmarks = torch.from_numpy(data["landmarks"]).float()
        au_intensities = torch.from_numpy(data["au_intensities"]).float()[:self.n_aus]
        eye_crop = torch.from_numpy(data["eye_crop"]).float()
        has_gaze = torch.tensor(1.0 if "gaze_vector" in data.files else 0.0)
        gaze = torch.from_numpy(data["gaze_vector"]).float() if "gaze_vector" in data.files else torch.zeros(3)
        return landmarks, au_intensities, eye_crop, gaze, has_gaze


class BP4DDataset(Dataset):
    """
    BP4D spontaneous dataset — preprocessed .npz frames.

    Provides 22 AUs but only the 16 most commonly used are returned
    (BP4D_AUS). The full AU index is available via BP4D_AUS constant.
    """

    def __init__(self, root: str, n_aus: int = 12):
        self.root = Path(root)
        self.n_aus = n_aus
        self.files = sorted(self.root.rglob("*.npz"))

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, idx: int):
        data = np.load(self.files[idx], allow_pickle=False)
        landmarks = torch.from_numpy(data["landmarks"]).float()
        au_intensities = torch.from_numpy(data["au_intensities"]).float()[:self.n_aus]
        eye_crop = torch.from_numpy(data["eye_crop"]).float()
        has_gaze = torch.tensor(0.0)
        gaze = torch.zeros(3)
        return landmarks, au_intensities, eye_crop, gaze, has_gaze


class MPIIGazeDataset(Dataset):
    """
    MPIIGaze dataset — preprocessed .npz frames.

    Each .npz: eye_crop (3,64,64), gaze_vector (3,), landmarks (68,2).
    Landmark AUs are not available; au_intensities is returned as zeros.
    """

    def __init__(self, root: str, n_aus: int = 12):
        self.root = Path(root)
        self.n_aus = n_aus
        self.files = sorted(self.root.rglob("*.npz"))

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, idx: int):
        data = np.load(self.files[idx], allow_pickle=False)
        landmarks = torch.from_numpy(data["landmarks"]).float() if "landmarks" in data.files else torch.zeros(68, 2)
        au_intensities = torch.zeros(self.n_aus)
        eye_crop = torch.from_numpy(data["eye_crop"]).float()
        gaze = torch.from_numpy(data["gaze_vector"]).float()
        has_gaze = torch.tensor(1.0)
        return landmarks, au_intensities, eye_crop, gaze, has_gaze


class ETHXGazeDataset(Dataset):
    """
    ETH-XGaze dataset — high-res multi-camera gaze.

    Structure after preprocessing:
        root/  subject_id/  frame_*.npz

    Each .npz: eye_crop (3,64,64), gaze_vector (3,), face_crop (3,224,224).
    """

    def __init__(self, root: str, n_aus: int = 12):
        self.root = Path(root)
        self.n_aus = n_aus
        self.files = sorted(self.root.rglob("*.npz"))

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, idx: int):
        data = np.load(self.files[idx], allow_pickle=False)
        landmarks = torch.zeros(68, 2)
        au_intensities = torch.zeros(self.n_aus)
        eye_crop = torch.from_numpy(data["eye_crop"]).float()
        gaze = torch.from_numpy(data["gaze_vector"]).float()
        has_gaze = torch.tensor(1.0)
        return landmarks, au_intensities, eye_crop, gaze, has_gaze
