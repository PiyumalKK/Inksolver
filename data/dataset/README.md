# Training Datasets

The CNN model (`models/symbol_classifier_crohme.h5`) was trained on two combined datasets:

## Dataset 1 — CROHME Extracted Symbols (Primary)
- **Source:** https://www.kaggle.com/datasets/xainano/handwrittenmathsymbols
- **Size:** 100,000+ images
- **Resolution:** 45×45 pixels (grayscale JPG)
- **Classes:** 82 math symbol classes extracted from CROHME competition data
- **Capped at:** 3,000 images per class (to prevent class imbalance)

## Dataset 2 — Handwritten Math Symbols (Supplementary)
- **Source:** https://www.kaggle.com/datasets/sagyamthapa/handwritten-math-symbols
- **Size:** ~10,000 images
- **Resolution:** 400×400 pixels (resized to 45×45 during training)
- **Added because:** Dataset 1 was missing the `=` (equals) sign class

## Note
The datasets are NOT included in this repo (~1.5GB total). The pre-trained model already exists at `models/symbol_classifier_crohme.h5` — you only need the raw data if you want to retrain.
