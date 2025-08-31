# train_srcnn.py
import os
import time
import torch
import torch.nn as nn
import torchvision.transforms as T
from torch.utils.data import DataLoader

import mlflow
import mlflow.pytorch

from srcnn_model import SRCNN
from utils import save_sample_image
from div2k_dataset import DIV2KDataset
from metrics import calculate_psnr, calculate_ssim

# ------------------------------
# Configuration (edit as needed)
# ------------------------------
config = {
    # Choose scale folder name exactly as on disk: "x2", "x3", or "x4"
    "scale": "x2_10",

    # Dataset folders (must be siblings of this script or give full paths)
    "hr_train_dir": "DIV2K_train_HR_10",
    "lr_train_root": "DIV2K_train_LR_bicubic",
    "hr_val_dir": "DIV2K_valid_HR_10",
    "lr_val_root": "DIV2K_valid_LR_bicubic",

    # Training hyperparams
    "batch_size": 16,
    "epochs": 50,
    "learning_rate": 1e-4,

    # Data loader / IO
    "upsample_lr_to_hr": True,   # SRCNN expects bicubic-upsampled LR -> HR size as input
    "num_workers": 0,            # set 0 to avoid dataloader hangs on some systems; increase if stable
    "pin_memory": True,

    # MLflow
    "mlflow_experiment": "Image Super-Resolution",
    "mlflow_run_name": None,     # if None, will be set automatically to SRCNN-{scale}
}

# ------------------------------
# Device
# ------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[INFO] Using device: {device}")

# ------------------------------
# Prepare dataset paths
# ------------------------------
lr_train_dir = os.path.join(config["lr_train_root"], config["scale"])
lr_val_dir = os.path.join(config["lr_val_root"], config["scale"])

print(f"[INFO] LR train dir: {lr_train_dir}")
print(f"[INFO] HR train dir: {config['hr_train_dir']}")
print(f"[INFO] LR val dir:   {lr_val_dir}")
print(f"[INFO] HR val dir:   {config['hr_val_dir']}")

# ------------------------------
# Transforms
# ------------------------------
# We expect DIV2K patches: HR ~480x480, LR smaller (240/160/120); dataset will upsample LR->HR when configured
transform = T.ToTensor()

# ------------------------------
# Datasets & DataLoaders
# ------------------------------
train_set = DIV2KDataset(
    hr_dir=config["hr_train_dir"],
    lr_dir=lr_train_dir,
    upsample_lr_to_hr=config["upsample_lr_to_hr"],
    transform_hr=transform,
    transform_lr=transform,
)

val_set = DIV2KDataset(
    hr_dir=config["hr_val_dir"],
    lr_dir=lr_val_dir,
    upsample_lr_to_hr=config["upsample_lr_to_hr"],
    transform_hr=transform,
    transform_lr=transform,
)

print(f"[INFO] Train samples: {len(train_set)}")
print(f"[INFO] Val samples:   {len(val_set)}")

train_loader = DataLoader(
    train_set,
    batch_size=config["batch_size"],
    shuffle=True,
    num_workers=config["num_workers"],
    pin_memory=config["pin_memory"]
)

val_loader = DataLoader(
    val_set,
    batch_size=1,
    shuffle=False,
    num_workers=max(0, config["num_workers"] // 2),
    pin_memory=config["pin_memory"]
)

# ------------------------------
# Model, Loss, Optimizer
# ------------------------------
model = SRCNN().to(device)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=config["learning_rate"])

# ------------------------------
# MLflow setup
# ------------------------------
mlflow.set_experiment(config["mlflow_experiment"])
run_name = config["mlflow_run_name"] or f"SRCNN-{config['scale']}"
print(f"[INFO] Starting MLflow run: {run_name}")

with mlflow.start_run(run_name=run_name):
    # log hyperparameters
    mlflow.log_params({
        "scale": config["scale"],
        "batch_size": config["batch_size"],
        "epochs": config["epochs"],
        "learning_rate": config["learning_rate"],
        "upsample_lr_to_hr": config["upsample_lr_to_hr"],
        "num_workers": config["num_workers"],
    })

    # optional: log device info
    mlflow.set_tag("device", str(device))
    if torch.cuda.is_available():
        try:
            mlflow.log_param("cuda_device_name", torch.cuda.get_device_name(0))
        except Exception:
            pass

    # ------------------------------
    # Training loop
    # ------------------------------
    start_time = time.time()
    for epoch in range(config["epochs"]):
        epoch_start = time.time()

        # --- Train ---
        model.train()
        running_loss = 0.0
        for batch_idx, (lr, hr) in enumerate(train_loader):
            lr = lr.to(device, non_blocking=True)
            hr = hr.to(device, non_blocking=True)

            sr = model(lr)
            loss = criterion(sr, hr)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        avg_train_loss = running_loss / max(1, len(train_loader))
        mlflow.log_metric("train_loss", avg_train_loss, step=epoch)

        # --- Validate ---
        model.eval()
        total_psnr = 0.0
        total_ssim = 0.0
        with torch.no_grad():
            for i, (lr_val, hr_val) in enumerate(val_loader):
                lr_val = lr_val.to(device, non_blocking=True)
                hr_val = hr_val.to(device, non_blocking=True)

                sr_val = model(lr_val)

                # metrics functions expect tensors with batch dim
                total_psnr += calculate_psnr(sr_val, hr_val)
                total_ssim += calculate_ssim(sr_val, hr_val)

                # Save a single sample from validation every 5 epochs (first batch only)
                if ((epoch + 1) % 5 == 0) and (i == 0):
                    try:
                        # Save file and log as artifact
                        sample_path = save_sample_image(lr_val, sr_val, hr_val, epoch, save_dir="outputs")
                        mlflow.log_artifact(sample_path, artifact_path="samples")
                        print(f"[INFO] Saved sample image: {sample_path}")
                    except Exception as e:
                        print(f"[WARN] Failed to save/log sample image: {e}")

        avg_psnr = total_psnr / max(1, len(val_loader))
        avg_ssim = total_ssim / max(1, len(val_loader))

        # Log validation metrics
        mlflow.log_metric("val_psnr", avg_psnr, step=epoch)
        mlflow.log_metric("val_ssim", avg_ssim, step=epoch)

        epoch_time = time.time() - epoch_start
        print(f"Epoch [{epoch+1}/{config['epochs']}] | Train Loss: {avg_train_loss:.6f} | "
            f"Val PSNR: {avg_psnr:.2f} dB | Val SSIM: {avg_ssim:.4f} | "
            f"Time: {epoch_time:.1f}s")

    # ------------------------------
    # End of training - save final model to MLflow
    # ------------------------------
    total_time = (time.time() - start_time) / 60.0
    mlflow.log_metric("total_training_time_minutes", total_time)
    # Save local model checkpoint
    model_path = "srcnn_final_model.pth"
    torch.save(model.state_dict(), model_path)
    mlflow.log_artifact(model_path, artifact_path="models")

    # Also save in MLflow model format (makes loading easier)
    try:
        mlflow.pytorch.log_model(model,
                                name="srcnn_model",
                                pip_requirements=["torch==2.7.1+cu128", "torchvision==0.20.1+cu128"])
    except Exception as e:
        print(f"[WARN] mlflow.pytorch.log_model failed: {e}")

    print(f"[INFO] Training finished. Total time: {total_time:.2f} minutes")
