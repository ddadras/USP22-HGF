# 🧠 CNN Image Classification

This repository performs contour-filtered image classification using 5-fold cross-validation with a **ResNet-34 backbone** on light microscopy images of wildtype and USP-22 deficient HT-29 cell lines, automatically producing:
- Classification reports and metrics per fold
- Confusion matrices (raw & normalized)
- ROC/PR curves
- Exported CSV summaries
- Optional PDF summary report (for papers)

---

## 🚀 Quick start

This repository works with Python 3.10, newer versions may require different packages.
Note: If CUDA is used, the respective version should be downloaded from the official website (https://developer.nvidia.com/cuda-downloads).

Follow these steps to set up and run the project locally.

1. Clone the repository
```
git clone https://github.com/ddadras/USP22-HGF
cd USP22-HGF
```

2. Set up a virtual environment

Create and activate a Python virtual environment (recommended):

```
# Create virtual environment
python -m venv venv

# Activate it
# On Linux/Mac:
source venv/bin/activate

# On Windows:
venv\Scripts\activate
```

3. Install dependencies

Make sure pip is up to date and install all required packages:

```
pip install --upgrade pip
pip install -r requirements.txt
```

4. Run inference

Once the environment is ready, run the inference script:

```
python src/infer.py
```

