import os, csv, random, shutil, pandas as pd
from analyze_face_features import analyze_face, compare_geometry

# lighting-based facial recognition test
def run_lighting_test(meta_path="faces_metadata.csv",
                      faces_dir="./faces",
                      ref_dir="./reference",
                      out_dir="./results/lighting"):
    """
    Each user runs one random probe image against all reference faces:
      - 1 genuine (self)
      - n-1 impostors (others)
    All IDs use double-digit format ('01', '02', ...).
    Clears previous results before running.
    Records both probe and reference lighting for later analysis.
    """

    # start clean each run
    if os.path.exists(out_dir):
        shutil.rmtree(out_dir)
    os.makedirs(out_dir, exist_ok=True)

    out_csv = os.path.join(out_dir, "scores.csv")
    df = pd.read_csv(meta_path)

    # normalize IDs → ensure '01', '02', ... format
    df["id"] = df["id"].apply(lambda x: f"{int(x):02d}")
    users = sorted(df["id"].unique())

    # load reference features
    print("\nAnalyzing reference faces...")
    references = {}
    for uid in users:
        ref_path = os.path.join(ref_dir, f"{uid}.jpg")
        if os.path.exists(ref_path):
            references[uid] = analyze_face(ref_path, {"id": uid})
        else:
            print(f"Missing reference for user {uid}")
    print(f"Loaded {len(references)} reference faces.\n")

    # open output CSV
    with open(out_csv, "w", newline="") as f:
        fields = [
            "probe_id", "ref_id", "score", "label",
            "probe_lighting", "ref_lighting", "lighting_diff"
        ]
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()

        # one random probe per user
        for uid in users:
            subset = df[df["id"] == uid]
            probe_images = subset["image_path"].tolist()
            if not probe_images:
                print(f"No images for user {uid}")
                continue

            # randomly pick one image for this user
            probe_path = random.choice(probe_images)
            print(f"User {uid} → probe {os.path.basename(probe_path)}")

            probe_feats = analyze_face(probe_path, {"id": uid})
            if not probe_feats:
                continue

            probe_light = probe_feats["lighting_score"]

            # compare probe with every reference
            for ref_id, ref_feats in references.items():
                score = compare_geometry(probe_feats, ref_feats)
                ref_light = ref_feats["lighting_score"]
                diff = abs(probe_light - ref_light)
                label = "genuine" if ref_id == uid else "impostor"

                writer.writerow({
                    "probe_id": uid,
                    "ref_id": ref_id,
                    "score": score,
                    "label": label,
                    "probe_lighting": probe_light,
                    "ref_lighting": ref_light,
                    "lighting_diff": diff
                })

        print(f"\nLighting test results saved to: {out_csv}")

# ------------------------------------------------------------
if __name__ == "__main__":
    run_lighting_test()

