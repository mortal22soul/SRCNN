import os
from PIL import Image

# Add all your image directories to this list
image_dirs = [
    "DIV2K_train_HR",
    "DIV2K_train_LR_bicubic/X2",
    "DIV2K_train_LR_bicubic/X3",
    "DIV2K_train_LR_bicubic/X4",
    "DIV2K_valid_HR",
    "DIV2K_valid_LR_bicubic/X2",
    "DIV2K_valid_LR_bicubic/X3",
    "DIV2K_valid_LR_bicubic/X4",
]

for folder in image_dirs:
    if not os.path.isdir(folder):
        print(f"WARNING: Directory not found, skipping: {folder}")
        continue
    
    print(f"\n--- Checking folder: {folder} ---")
    for filename in os.listdir(folder):
        if filename.lower().endswith(".png"):
            try:
                img_path = os.path.join(folder, filename)
                with Image.open(img_path) as img:
                    img.verify() # Verify image integrity
            except Exception as e:
                print(f"!!!!!!!! CORRUPTED FILE DETECTED !!!!!!!!")
                print(f"File: {img_path}")
                print(f"Error: {e}")

print("\n--- Check complete ---")