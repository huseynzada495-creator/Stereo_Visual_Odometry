from dataclasses import dataclass
from pathlib import Path
from typing import List


@dataclass
class TumVIDataset:
    root: Path

    def __post_init__(self):
        self.root = Path(self.root)
        self.left_dir = self.root / 'mav0' / 'cam0' / 'data'
        self.right_dir = self.root / 'mav0' / 'cam1' / 'data'
        self.gt_file = self.root / 'mav0' / 'mocap0' / 'data.csv'
        if not self.left_dir.exists() or not self.right_dir.exists():
            raise FileNotFoundError(f'Could not find cam0/cam1 data folders under {self.root}')
        self.left_images: List[Path] = sorted(self.left_dir.glob('*.png'))
        self.right_images: List[Path] = sorted(self.right_dir.glob('*.png'))
        if len(self.left_images) == 0 or len(self.right_images) == 0:
            raise RuntimeError('No PNG images found in dataset.')

    @property
    def timestamps_ns(self):
        return [int(p.stem) for p in self.left_images]

    def __len__(self):
        return min(len(self.left_images), len(self.right_images))

    def pair(self, idx: int):
        return self.timestamps_ns[idx], self.left_images[idx], self.right_images[idx]
