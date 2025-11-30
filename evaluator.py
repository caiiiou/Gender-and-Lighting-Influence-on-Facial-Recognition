import os, pandas as pd, numpy as np, matplotlib.pyplot as plt

# Reusable evaluation class
class Evaluator:
    def __init__(self, genuine, impostor, out_dir, title):
        self.genuine = np.asarray(genuine)
        self.impostor = np.asarray(impostor)
        self.out_dir = out_dir
        self.title = title
        self.thresholds = np.linspace(-0.1, 1.1, 500)
        os.makedirs(out_dir, exist_ok=True)

    # ---------- metrics ----------
    def d_prime(self):
        mu_g, mu_i = np.mean(self.genuine), np.mean(self.impostor)
        sd_g, sd_i = np.std(self.genuine), np.std(self.impostor)
        pooled = np.sqrt(0.5 * (sd_g**2 + sd_i**2))
        return (mu_g - mu_i) / (pooled + 1e-9)

    def compute_rates(self):
        FPR, FNR, TPR = [], [], []
        for t in self.thresholds:
            TP = np.sum(self.genuine >= t)
            FN = np.sum(self.genuine <  t)
            FP = np.sum(self.impostor >= t)
            TN = np.sum(self.impostor <  t)
            FPR.append(FP / (FP + TN + 1e-9))
            FNR.append(FN / (TP + FN + 1e-9))
            TPR.append(TP / (TP + FN + 1e-9))
        return np.array(FPR), np.array(FNR), np.array(TPR)

    def eer(self, FPR, FNR):
        diff = np.abs(FPR - FNR)
        i = np.argmin(diff)
        return 0.5 * (FPR[i] + FNR[i]), self.thresholds[i]

    # ---------- plots ----------
    def plot_all(self, suffix=""):
        FPR, FNR, TPR = self.compute_rates()
        dval, (eer_val, eer_t) = self.d_prime(), self.eer(FPR, FNR)

        # score distribution
        plt.figure(figsize=(7,5))
        plt.hist(self.genuine, bins=40, range=(0,1), histtype="step", color="green", label="Genuine")
        plt.hist(self.impostor, bins=40, range=(0,1), histtype="step", color="red", label="Impostor")
        plt.title(f"Score Distribution – {self.title}\nD′={dval:.3f}")
        plt.xlabel("Score")
        plt.ylabel("Frequency")
        plt.legend()
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(self.out_dir, f"score_distribution{suffix}.png"), dpi=300, bbox_inches="tight")
        plt.close()

        # ROC curve
        plt.figure(figsize=(6,6))
        plt.plot(FPR, TPR, lw=2, color="blue")
        plt.plot([0,1], [0,1], "k--", alpha=0.4)
        plt.title(f"ROC – {self.title}")
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(self.out_dir, f"roc_curve{suffix}.png"), dpi=300, bbox_inches="tight")
        plt.close()

        # DET curve
        plt.figure(figsize=(6,6))
        plt.plot(FPR, FNR, lw=2, color="purple")
        plt.plot([0,1], [0,1], "k--", alpha=0.4)
        plt.scatter([eer_val], [eer_val], c="black", s=40)
        plt.title(f"DET – {self.title}\nEER={eer_val:.3f} @ t={eer_t:.3f}")
        plt.xlabel("False Positive Rate")
        plt.ylabel("False Negative Rate")
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(self.out_dir, f"det_curve{suffix}.png"), dpi=300, bbox_inches="tight")
        plt.close()

        print(f"→ {self.title}{suffix}: D′={dval:.3f}, EER={eer_val:.3f}")

# evaluate a given results folder
def evaluate_gender(folder="./results/gender"):
    csv_path = os.path.join(folder, "scores.csv")
    if not os.path.exists(csv_path):
        print(f"Missing {csv_path}, skipping gender analysis.")
        return

    df = pd.read_csv(csv_path)
    for gender in ["male", "female"]:
        subset = df[df["gender"] == gender]
        genuine = subset[subset["label"] == "genuine"]["score"].values
        impostor = subset[subset["label"] == "impostor"]["score"].values
        if len(genuine) == 0 or len(impostor) == 0:
            print(f"Not enough data for {gender}, skipping.")
            continue

        print(f"\nEvaluating {gender.title()} users...")
        ev = Evaluator(genuine, impostor, folder, f"Gender – {gender.title()}")
        ev.plot_all(suffix=f"_{gender}")

# evaluate lighting by brightness difference buckets
def evaluate_lighting(folder="./results/lighting"):
    csv_path = os.path.join(folder, "scores.csv")
    if not os.path.exists(csv_path):
        print(f"Missing {csv_path}, skipping lighting analysis.")
        return

    df = pd.read_csv(csv_path)
    if "lighting_diff" not in df.columns:
        print("lighting_diff column not found; re-run lighting_test.")
        return

    # define lighting difference buckets
    bins = [0, 15, 40, 255]  # adjust if brightness scale is different
    labels = ["Low", "Medium", "High"]
    df["bucket"] = pd.cut(df["lighting_diff"], bins=bins, labels=labels, include_lowest=True)

    for bucket in labels:
        subset = df[df["bucket"] == bucket]
        genuine = subset[subset["label"] == "genuine"]["score"].values
        impostor = subset[subset["label"] == "impostor"]["score"].values
        if len(genuine) == 0 or len(impostor) == 0:
            print(f"Not enough data for {bucket} lighting, skipping.")
            continue

        print(f"\nEvaluating {bucket} Lighting Difference...")
        ev = Evaluator(genuine, impostor, folder, f"Lighting – {bucket}")
        ev.plot_all(suffix=f"_{bucket.lower()}")

def main():
    print("\n===== Evaluating Facial Recognition Tests =====")
    evaluate_gender("./results/gender")
    evaluate_lighting("./results/lighting")
    print("\nAll subgroup evaluations complete.\n")

if __name__ == "__main__":
    main()

