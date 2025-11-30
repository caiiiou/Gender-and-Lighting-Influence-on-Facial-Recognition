import os, cv2, numpy as np, shutil
from face_recognition import load_image_file

def brightness_score(path):
    # quick brightness metric: mean / std, higher means more evenly lit
    img = load_image_file(path)
    g = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    return np.mean(g) / (np.std(g) + 1e-5)

def best_image(folder):
    # go through all images in this folder and keep the one with the best score
    best, score = None, -1
    for f in os.listdir(folder):
        if not f.lower().endswith(('.jpg','.jpeg','.png')):
            continue
        try:
            s = brightness_score(os.path.join(folder, f))
            if s > score:
                best, score = f, s
        except:
            # skip if image can’t be opened or read for some reason
            pass
    return best, score

def build_refs(src="./faces", dst="./reference"):
    # start clean each time (remove old refs)
    if os.path.exists(dst):
        for f in os.listdir(dst):
            p = os.path.join(dst, f)
            if os.path.isfile(p):
                os.remove(p)
    os.makedirs(dst, exist_ok=True)

    results = []
    for d in sorted(os.listdir(src)):
        path = os.path.join(src, d)
        # skip files and make sure we don’t accidentally process the output folder itself
        if not os.path.isdir(path) or os.path.abspath(path) == os.path.abspath(dst):
            continue
        f, s = best_image(path)
        if f:
            # copy best image and rename to the folder name (e.g. 01.jpg)
            shutil.copy2(os.path.join(path, f), os.path.join(dst, f"{d}.jpg"))
            results.append((d, f, s))
            print(f"{d}: {f} (score={s:.2f})")

    # print everything sorted by brightness score for reference
    print("\nSorted by lighting quality:")
    for d, f, s in sorted(results, key=lambda x: x[2], reverse=True):
        print(f"{d}: {f} (score={s:.2f})")

if __name__ == "__main__":
    # you can change src and dst paths here if needed
    build_refs()

