"""Quick helper for batch-resizing dataset images to a fixed size."""
import os
from PIL import Image

SRC = "faces"
DST = "faces_resized"
SIZE = (256, 256)


def main():
    for subj in sorted(os.listdir(SRC)):
        subj_dir = os.path.join(SRC, subj)
        if not os.path.isdir(subj_dir):
            continue
        out_dir = os.path.join(DST, subj)
        os.makedirs(out_dir, exist_ok=True)
        for f in sorted(os.listdir(subj_dir)):
            if not f.lower().endswith(".jpg"):
                continue
            img = Image.open(os.path.join(subj_dir, f))
            img.thumbnail(SIZE)
            img.save(os.path.join(out_dir, f))


if __name__ == "__main__":
    main()
