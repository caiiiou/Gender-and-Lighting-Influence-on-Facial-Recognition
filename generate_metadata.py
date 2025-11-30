import os, csv, numpy as np, face_recognition

# brightness check (lighting metric)
def lighting_score(path):
    # load the image (face_recognition gives an RGB numpy array)
    img = face_recognition.load_image_file(path)
    # convert roughly to grayscale by averaging color channels
    gray = np.mean(img, axis=2)
    # mean pixel value = simple brightness estimate
    return float(np.mean(gray))

# main metadata generator
def generate_metadata(base="./faces", out_csv="faces_metadata.csv"):
    """
    Scans each user folder under 'faces' and records:
      - user id (folder name)
      - image name
      - gender (entered as 'm' or 'f')
      - lighting score (mean brightness)
      - full image path
    Saves everything into a CSV for later testing and analysis.
    """

    with open(out_csv, "w", newline="") as f:
        fields = ["id", "image_name", "gender", "lighting_score", "image_path"]
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()

        # loop through each subfolder — one per user
        for user_id in sorted(os.listdir(base)):
            user_path = os.path.join(base, user_id)
            if not os.path.isdir(user_path):
                continue  # skip anything that’s not a folder

            print(f"\nUser {user_id}")

            # loop until valid input provided (m or f)
            while True:
                gender_input = input("Enter gender (m/f): ").strip().lower()
                if gender_input in ("m", "f"):
                    gender = "male" if gender_input == "m" else "female"
                    break
                else:
                    print("Invalid input — please type only 'm' for male or 'f' for female.")

            # loop through images for this user
            for img_name in os.listdir(user_path):
                if not img_name.lower().endswith((".jpg", ".jpeg", ".png")):
                    continue  # skip non-images

                img_path = os.path.join(user_path, img_name)
                try:
                    score = lighting_score(img_path)
                except Exception as e:
                    print(f"Skipped {img_name}: {e}")
                    score = np.nan

                writer.writerow({
                    "id": user_id,
                    "image_name": img_name,
                    "gender": gender,
                    "lighting_score": score,
                    "image_path": img_path
                })
                print(f"  {img_name} → lighting={score:.2f}")

    print(f"\nMetadata saved to: {out_csv}")

if __name__ == "__main__":
    generate_metadata("./faces", "faces_metadata.csv")

