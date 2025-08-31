import os
from PIL import Image
import torchvision.transforms as T
from torch.utils.data import Dataset
from typing import Optional, Callable, Tuple, List

class DIV2KDataset(Dataset):
    """
    Dataset for DIV2K *patches* with identical HR/LR filenames.

    Expected top-level folders (siblings of this file):
    - DIV2K_train_HR/                  (HR train patches, 480x480)
    - DIV2K_train_LR_bicubic/X2|X3|X4  (LR train patches, 240/160/120)
    - DIV2K_valid_HR/                  (HR valid patches, 480x480)
    - DIV2K_valid_LR_bicubic/X2|X3|X4  (LR valid patches, 240/160/120)

    Filenames match across HR and LR (e.g., "0001_s001.png" in both).
    For SRCNN, we typically bicubic-upsample LR to HR size before feeding the model.
    """

    def __init__(
        self,
        hr_dir: str,
        lr_dir: str,
        upsample_lr_to_hr: bool = True,
        interpolation: int = Image.BICUBIC,
        transform_hr: Optional[Callable] = None,
        transform_lr: Optional[Callable] = None,
    ) -> None:
        """
        Args:
            hr_dir: path to HR patches (e.g., 'DIV2K_train_HR')
            lr_dir: path to LR patches at a given scale (e.g., 'DIV2K_train_LR_bicubic/x2')
            upsample_lr_to_hr: if True (recommended for SRCNN), resize LR to HR size before returning
            interpolation: PIL interpolation to use for upsampling (default: bicubic)
            transform_hr: optional transform applied to HR PIL image (after any resizing)
            transform_lr: optional transform applied to LR PIL image (after any resizing)
        """
        self.hr_dir = hr_dir
        self.lr_dir = lr_dir
        self.upsample_lr_to_hr = upsample_lr_to_hr
        self.interp = interpolation

        # Build the list of filenames present in BOTH folders
        hr_names = sorted([f for f in os.listdir(hr_dir) if f.lower().endswith(".png")])
        lr_names = set(f for f in os.listdir(lr_dir) if f.lower().endswith(".png"))
        self.filenames: List[str] = [f for f in hr_names if f in lr_names]

        if len(self.filenames) == 0:
            raise RuntimeError(
                f"No matching PNG filenames found between:\n  HR: {hr_dir}\n  LR: {lr_dir}"
            )

        # Default transforms: just ToTensor()
        self.transform_hr = transform_hr or T.ToTensor()
        self.transform_lr = transform_lr or T.ToTensor()

    def __len__(self) -> int:
        return len(self.filenames)

    def __getitem__(self, idx: int) -> Tuple:
        fname = self.filenames[idx]
        hr_path = os.path.join(self.hr_dir, fname)
        lr_path = os.path.join(self.lr_dir, fname)

        hr_img = Image.open(hr_path).convert("RGB")   # 480x480
        lr_img = Image.open(lr_path).convert("RGB")   # 240/160/120 depending on scale

        # For SRCNN: resize LR → HR size so model output matches HR for the loss
        if self.upsample_lr_to_hr:
            lr_img = lr_img.resize(hr_img.size, resample=self.interp)

        hr_tensor = self.transform_hr(hr_img)
        lr_tensor = self.transform_lr(lr_img)

        return lr_tensor, hr_tensor
