import os, csv, random, shutil, pandas as pd
from analyze_face_features import analyze_face, compare_geometry

# gender-based facial recognition test
def run_gender_test(meta_path="faces_metadata.csv",
                    faces_dir="./faces",
                    ref_dir="./reference",
                    out_dir="./results/gender"):
    """
    Each user runs one random probe image against all reference faces:
      - 1 genuine (self)
      - n-1 impostors (others)
    All IDs are treated as double-digit strings ('01', '02', ...).
    Clears any previous results before running.
    """

    # fresh start: wipe the previous results folder
    if os.path.exists(out_dir):
        shutil.rmtree(out_dir)
    os.makedirs(out_dir, exist_ok=True)

    out_csv = os.path.join(out_dir, "scores.csv")
    df = pd.read_csv(meta_path)

    # normalize IDs → ensure '01', '02', ... format
    df["id"] = df["id"].apply(lambda x: f"{int(x):02d}")
    users = sorted(df["id"].unique())

    # load all reference faces
    print("\nAnalyzing reference faces...")
    references = {}
    for uid in users:
        # enforce two-digit naming when loading reference images
        ref_path = os.path.join(ref_dir, f"{uid}.jpg")
        if os.path.exists(ref_path):
            references[uid] = analyze_face(ref_path, {"id": uid})
        else:
            print(f"Missing reference for user {uid}")
    print(f"Loaded {len(references)} reference faces.\n")

    # open output CSV
    with open(out_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["probe_id", "ref_id", "gender", "score", "label"])
        writer.writeheader()

        # one random probe image per user
        for uid in users:
            subset = df[df["id"] == uid]
            gender = subset["gender"].iloc[0]
            probe_images = subset["image_path"].tolist()
            if not probe_images:
                print(f"No images for user {uid}")
                continue

            # randomly select a probe image
            probe_path = random.choice(probe_images)
            print(f"User {uid} ({gender}) → probe {os.path.basename(probe_path)}")

            probe_feats = analyze_face(probe_path, {"id": uid, "gender": gender})
            if not probe_feats:
                continue

            # compare this probe with every reference
            for ref_id, ref_feats in references.items():
                score = compare_geometry(probe_feats, ref_feats)
                label = "genuine" if ref_id == uid else "impostor"
                writer.writerow({
                    "probe_id": uid,
                    "ref_id": ref_id,
                    "gender": gender,
                    "score": score,
                    "label": label
                })

        print(f"\nGender test results saved to: {out_csv}")

# ------------------------------------------------------------
if __name__ == "__main__":
    run_gender_test()

