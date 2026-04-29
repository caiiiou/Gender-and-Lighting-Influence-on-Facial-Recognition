# Feature-Level Fusion Facial Recognition Analysis

An empirical study of how subject gender and ambient lighting conditions affect the accuracy of a feature-level fusion facial recognition pipeline. Rather than relying on deep embeddings, this system extracts geometric facial landmarks (interocular distance, nose bridge, lip area) using `face_recognition` and compares them across subjects to produce similarity scores.

Performance is evaluated using Equal Error Rate (EER), d-prime (d'), Receiver Operating Characteristic (ROC) curves, and Detection Error Tradeoff (DET) curves.

## Components

| File | Purpose |
|------|---------|
| `generate_metadata.py` | Prompts for gender labels (m/f), computes per-image lighting brightness, and writes the dataset to CSV. |
| `select_reference.py` | Selects a representative reference image per subject and copies it into `reference/`. |
| `analyze_face_features.py` | Extracts geometric landmark features and exposes a shared `compare_geometry()` similarity function. |
| `gender_test.py` | Runs reference-vs-probe comparisons grouped by gender. |
| `lighting_test.py` | Buckets images into low/medium/high lighting and evaluates lighting influence on accuracy. |
| `evaluator.py` | Generates ROC, DET, and score-distribution plots for each condition. |

## Feature-Level Approach

For each image, the pipeline computes:

- **Interocular distance** — Euclidean distance between the left and right eye centroids.
- **Lip area** — combined polygon area of the upper and lower lip landmarks.
- **Lighting score** — mean pixel intensity (used for bucketing, not similarity).

The similarity between two images is a normalized inverse Euclidean distance over the geometric features:

```
similarity = 1 / (1 + sqrt((Δeye)^2 + (Δlip)^2))
```

Scores fall in (0, 1], with higher values indicating more similar geometry.

## Dataset

The Caltech Faces 1999 dataset is used: 450 images of 27 individuals captured under controlled lighting and pose variation.

> California Institute of Technology. (1999). *Caltech Faces 1999 Dataset* [Data set]. CaltechDATA. <https://data.caltech.edu/records/6rjah-hdv18>

## Installation

Prerequisites: `dlib` with Python bindings, and `cmake` (e.g. `brew install cmake` on macOS).

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Folder Structure

```
faces/
  01/
    image_0001.jpg
    ...
  02/
    ...
reference/
  01.jpg
  02.jpg
  ...
results/
  gender/
  lighting/
faces_metadata.csv
generate_metadata.py
select_reference.py
analyze_face_features.py
gender_test.py
lighting_test.py
evaluator.py
```

## Running the Pipeline

Generate metadata (gender labels and per-image lighting scores):

```bash
python3 generate_metadata.py
```

Run the gender comparison:

```bash
python3 gender_test.py
```

Run the lighting comparison:

```bash
python3 lighting_test.py
```

Generate ROC, DET, and score-distribution plots:

```bash
python3 evaluator.py
```

## Output

Each test writes to its own subfolder under `results/`:

- `scores.csv` — per-pair similarity scores with ground-truth labels
- `score_distribution_[group].png`
- `roc_curve_[group].png`
- `det_curve_[group].png`

Example layout:

```
results/gender/
  score_distribution_male.png
  score_distribution_female.png
  roc_curve_male.png
  roc_curve_female.png
  det_curve_male.png
  det_curve_female.png
results/lighting/
  det_curve_low.png
  det_curve_medium.png
  det_curve_high.png
```

## Key Findings

| Condition | EER | d' | Observation |
|-----------|-----|----|-------------|
| Male | 0.267 | 1.15 | Slightly higher error rate |
| Female | 0.254 | 1.16 | Marginally better separation |
| Low lighting | 0.294 | 0.95 | Degraded accuracy under poor light |
| Medium lighting | 0.230 | 1.21 | Best accuracy and discriminability |
| High lighting | 0.255 | 1.05 | Mild degradation from glare and specular highlights |

Lighting had a stronger influence on recognition performance than gender. Accuracy across male and female groups was comparable, suggesting minimal demographic bias in the geometric feature set.

## References

1. Abdi, H. (2007). *Signal Detection Theory (SDT).* In *Encyclopedia of Measurement and Statistics.*
2. Li, S. Z., & Jain, A. K. (2011). *Handbook of Face Recognition.* Springer.
3. Viola, P., & Jones, M. (2001). *Rapid Object Detection using a Boosted Cascade of Simple Features.* IEEE CVPR.
4. OpenCV / face_recognition Python documentation (2024).
5. DeepFace API documentation (2025).
6. California Institute of Technology. (1999). *Caltech Faces 1999 Dataset* [Data set]. CaltechDATA. <https://data.caltech.edu/records/6rjah-hdv18>
