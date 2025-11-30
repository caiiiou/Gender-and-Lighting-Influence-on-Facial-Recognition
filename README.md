# Feature-Level Fusion Facial Recognition Analysis  

**CAP 4103 – Final Project**  
University of South Florida  

---

## Overview

This project investigates how **gender** and **lighting conditions** affect the accuracy of a **feature-level fusion facial recognition system**.  
Instead of relying on deep embeddings, the model extracts **geometric facial landmarks** (eye distance, nose bridge, and lip area) using `face_recognition` and compares them across subjects to generate similarity scores.

Performance is evaluated using **Equal Error Rate (EER)**, **d-prime (d′)**, **Receiver Operating Characteristic (ROC)**, and **Detection Error Tradeoff (DET)** metrics.

---

## Core Components

| File | Purpose |
|------|----------|
| `metadata_generator.py` | Prompts user to label gender (m/f), computes lighting brightness for each image, and saves to CSV. |
| `analyze_face_features.py` | Extracts geometric features and includes shared `compare_geometry()` function for scoring. |
| `gender_test.py` | Runs comparisons between each user's reference image and all others, grouped by gender. |
| `lighting_test.py` | Groups images into lighting buckets (low, medium, high) and evaluates lighting influence. |
| `evaluator.py` | Generates ROC, DET, and score distribution plots for each condition (gender and lighting). |

---

## Feature-Level Approach

Each image is analyzed for:

- **Interocular Distance** – distance between left and right eye centroids.  
- **Lip Area** – combined polygon area of upper and lower lips.  
- **Lighting Score** – mean brightness (pixel intensity average).  

A normalized Euclidean difference between these features produces a similarity score between 0 and 1:

```python
1 / (1 + sqrt((Δeye)^2 + (Δlip)^2))
```

---

## Dataset

The faces used in this project come from the publicly available **Caltech Faces 1999 Dataset**, which includes 450 images of 27 individuals with controlled lighting and pose variation.

**Citation:**
> California Institute of Technology. (1999). *Caltech Faces 1999 Dataset* [Data set]. CaltechDATA.  
> <https://data.caltech.edu/records/6rjah-hdv18>

---

## Installation

### 1️⃣ System prerequisites (Linux/macOS)

Ensure you have dlib already installed with Python bindings

Then, make sure you have cmake installed:

```bash
brew install cmake
```

### 2️⃣ Create and activate virtual environment

```bash
python3 -m venv venv
source venv/bin/activate
```

### 3️⃣ Install Python dependencies

```bash
pip install -r requirements_clean.txt
```

---

## 🧩 Folder Structure

```
faces/
 ├── 01/
 │   ├── image_0001.jpg
 │   └── ...
 ├── 02/
 │   ├── ...
reference/
 ├── 01.jpg
 ├── 02.jpg
results/
 ├── gender/
 ├── lighting/
faces_metadata.csv
metadata_generator.py
gender_test.py
lighting_test.py
analyze_face_features.py
evaluator.py
```

---

## Running the Pipeline

### 1️⃣ Generate Metadata

Prompts gender input and computes lighting score:

```bash
python3 metadata_generator.py
```

### 2️⃣ Run Gender Test

Evaluates recognition accuracy across genders:

```bash
python3 gender_test.py
```

### 3️⃣ Run Lighting Test

Tests lighting difference impact:

```bash
python3 lighting_test.py
```

### 4️⃣ Generate Evaluation Plots

Creates DET, ROC, and score distribution plots for both tests:

```bash
python3 evaluator.py
```

---

## Output

Each test creates a folder containing:

- `scores.csv`
- `score_distribution_[group].png`
- `roc_curve_[group].png`
- `det_curve_[group].png`

Example:

```
results/gender/
 ├── score_distribution_male.png
 ├── score_distribution_female.png
results/lighting/
 ├── det_curve_low.png
 ├── det_curve_medium.png
 ├── det_curve_high.png
```

---

## Key Findings

| Condition | EER ↓ | d′ ↑ | Observation |
|------------|--------|------|-------------|
| Male | 0.267 | 1.15 | Slightly higher error rate |
| Female | 0.254 | 1.16 | Marginally better separation |
| Low Lighting | 0.294 | 0.95 | Degraded accuracy under poor light |
| Medium Lighting | 0.230 | 1.21 | Optimal accuracy and discriminability |
| High Lighting | 0.255 | 1.05 | Mild degradation due to glare/specular highlights |

**Summary:** Lighting had a stronger influence on performance than gender.  
Balanced accuracy across male/female groups suggests minimal demographic bias.

---

## References

1. Abdi, H. (2007). *Signal Detection Theory (SDT).* In *Encyclopedia of Measurement and Statistics.*  
2. Li, S. Z., & Jain, A. K. (2011). *Handbook of Face Recognition.* Springer.  
3. Viola, P., & Jones, M. (2001). *Rapid Object Detection using a Boosted Cascade of Simple Features.* IEEE CVPR.  
4. OpenCV / Face_Recognition Python Docs (2024).  
5. DeepFace API Documentation (2025).  
6. California Institute of Technology. (1999). *Caltech Faces 1999 Dataset* [Data set]. CaltechDATA. <https://data.caltech.edu/records/6rjah-hdv18>
