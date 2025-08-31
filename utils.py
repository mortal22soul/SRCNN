import os
from torchvision.utils import save_image
from PIL import Image, ImageDraw, ImageFont

def save_sample_image(lr, sr, hr, epoch, save_dir="outputs"):
    os.makedirs(save_dir, exist_ok=True)
    output = f"{save_dir}/epoch{epoch+1}_sample.png"

    # Save the grid first
    save_image([lr.squeeze(), sr.squeeze(), hr.squeeze()], output, nrow=3)

    # Open with PIL to add labels
    img = Image.open(output).convert("RGB")
    draw = ImageDraw.Draw(img)

    # Try to load a default font
    try:
        font = ImageFont.truetype("arial.ttf", 20)
    except:
        font = ImageFont.load_default()

    W, H = img.size
    third = W // 3

    labels = ["LR", "SR", "HR"]
    for i, label in enumerate(labels):
        # get text size (modern Pillow way)
        bbox = draw.textbbox((0, 0), label, font=font)
        text_w = bbox[2] - bbox[0]
        text_h = bbox[3] - bbox[1]

        x = i * third + (third - text_w) // 2
        y = 5  # margin from top
        draw.text((x, y), label, fill=(255, 0, 0), font=font)

    img.save(output)
    return output
