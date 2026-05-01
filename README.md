# InkSolver - Handwritten Mathematical Equation Solver

A computer vision pipeline that reads handwritten math equations from images, recognizes symbols, and solves them.

## Pipeline

```
Photo → Grayscale → Threshold → Morphology Cleanup → Segment Characters → CNN Classify → Parse Equation → Solve
```

## Tech Stack
- **OpenCV** — Image preprocessing, thresholding, morphology, contours
- **TensorFlow/Keras** — CNN symbol recognition (trained on Colab T4)
- **SymPy** — Equation solving
- **Python 3.10+**

## Setup

```bash
git clone https://github.com/PiyumalKK/Inksolver.git
cd Inksolver
pip install opencv-python numpy matplotlib tensorflow sympy
```

## Quick Test

```bash
python src/solver.py data/raw_samples/synthetic_eq1.png
```

This runs the full pipeline on a sample image of `2x + 3 = 7` and prints:
```
equation: 2*x+3=7
x = [2]
```

To test with your own image, just pass the path:
```bash
python src/solver.py path/to/your/equation.png
```

## Usage

### Preprocess an image
Converts a photo of a handwritten equation into a clean binary image.
```bash
python src/preprocess.py data/raw_samples/synthetic_eq1.png
# saves output_binary.png
```

### Segment characters
Splits the equation into individual character images (sorted left to right).
```bash
python src/segment.py data/raw_samples/synthetic_eq1.png
# saves character crops to data/segments/
```

### Recognize symbols
Runs preprocessing + segmentation + CNN prediction end to end.
```bash
python src/model.py data/raw_samples/synthetic_eq1.png
# prints recognized symbols with confidence scores
```

### Solve equation
Full pipeline: preprocess -> segment -> classify -> parse -> solve.
```bash
python src/solver.py data/raw_samples/synthetic_eq1.png
# prints the equation and solution
```

### Notebooks
The `notebooks/` folder has step-by-step Jupyter notebooks that walk through each stage with visualizations.
```bash
jupyter notebook notebooks/
```

## Project Structure
```
├── notebooks/          # Step-by-step Jupyter notebooks
├── src/                # Source code modules
├── models/             # Trained models
├── data/               # Images and datasets
├── results/            # Output samples
└── report/             # LaTeX proposal & figures
```

## Progress Log

| Step | Description | Status |
|------|------------|--------|
| 0 | Project setup | Done |
| 1 | Preprocessing | Done |
| 2 | Segmentation | Done |
| 3 | CNN training | Done |
| 4 | Equation parsing & solving | Done |
| 5 | System of equations + line splitting | Done |

---

## Phase 0 — Project Setup

Created the repo, folder structure, and gitignore.

## Phase 1 — Image Preprocessing

The first step is turning a raw phone photo of a handwritten equation into a clean binary image that we can actually work with.

The pipeline:
1. **Grayscale** — drop the color channels, we only care about dark (ink) vs light (paper)
2. **Gaussian blur (5x5)** — smooth out the camera noise. Tried 3x3, 5x5, 9x9 and 5x5 was the sweet spot
3. **CLAHE** — adaptive contrast enhancement. Tried global histogram equalization first but it doesn't handle uneven lighting well (one side of paper brighter than the other). CLAHE splits the image into 8x8 tiles and equalizes each one separately, so it handles shadows way better
4. **Adaptive Gaussian threshold** — converts to binary (ink=white, paper=black). Adaptive because a single global threshold fails when lighting is uneven — it calculates a different threshold for each pixel based on its 15x15 neighborhood
5. **Morphological opening** — erosion then dilation with a 3x3 kernel. Removes tiny noise dots that survived thresholding
6. **Morphological closing** — dilation then erosion. Fills small gaps in character strokes where thin parts got broken during thresholding

Output: clean binary image ready for character segmentation.

Files: `notebooks/01_preprocessing.ipynb`, `src/preprocess.py`

## Phase 2 — Character Segmentation

Now that we have a clean binary image, we need to chop it up into individual characters.

The approach:
1. **Find contours** — `cv2.findContours` with `RETR_EXTERNAL` to grab only the outermost blobs. Each connected white region = one contour
2. **Bounding boxes** — get the rectangle around each contour. Filter out anything smaller than 0.1% of the image area (noise specks that survived preprocessing)
3. **Merge overlapping boxes** — this was the tricky part. The `=` sign shows up as two separate horizontal bars, so we get two contours for one symbol. Same issue with `i`, `j`, etc. Fix: if two boxes overlap horizontally by more than 50% of the smaller box's width, merge them into one bigger box
4. **Sort left to right** — sort by x coordinate so characters are in reading order
5. **Crop and resize** — cut each character out, pad it to a square (so it doesn't get stretched), then resize to 45x45. The padding preserves aspect ratio which matters for the CNN later

The `=` merging logic took some trial and error. First tried just checking if boxes are "close" vertically, but that merged things that shouldn't be merged. The horizontal overlap check works much better — if two contours are roughly in the same x range, they're probably parts of the same symbol.

Files: `notebooks/02_segmentation.ipynb`, `src/segment.py`

## Phase 3 — CNN Symbol Classifier

This is the "brain" — a CNN that looks at a 45x45 character crop and tells us what symbol it is.

Used the **HASYv2** dataset (~168k images of handwritten symbols). Filtered it down to 18 classes we need: digits 0-9, operators (+, -, ×, ÷), variables (x, y, X, Y). Note: `=` wasn't in HASYv2 so we handle that separately in the parser.

The dataset was pretty imbalanced — `times` had 1509 samples but `X` only had 54.

**Architecture:**
- 3 conv blocks: Conv2D → BatchNorm → ReLU → MaxPool → Dropout
- Filters: 32 → 64 → 128
- Dense: 256 → softmax
- ~965k parameters

**Training:**
- Data augmentation: small rotations (±10°), shifts (10%), zoom (10%), shear
- Adam optimizer with ReduceLROnPlateau — LR started at 0.001, dropped to 0.0005 around epoch 7, then 0.00025 around epoch 30
- EarlyStopping with patience=10
- Trained on Google Colab T4 GPU, 50 epochs

First few epochs were scary — val accuracy was stuck at 3.6% while train was climbing fast. Looked like total overfitting. But ReduceLROnPlateau kicked in and dropped the learning rate, and after that val accuracy slowly caught up. By epoch 15 it crossed 90%, and by epoch 50 it settled at **95.8%** val accuracy with basically no train/val gap.

**Weak spots from confusion matrix:**
- `x` (lowercase) got 0% — completely confused with `X` and `times` (×). Makes sense, they all look like an X
- `X` recall was only 55%
- We'll handle the x/X/times ambiguity in the equation parser — context tells you if it's a variable or multiplication

The `src/model.py` module wraps the trained model for inference — load once, then predict on character crops.

Files: `notebooks/03_cnn_training_v1.ipynb` (Colab), `src/model.py`, `models/symbol_classifier.h5`, `models/label_map.json`

## Phase 4 — Equation Parsing & Solving

This is where the recognized symbols actually become a solvable equation.

Three main problems to deal with:

1. **Equals sign detection** — `=` wasn't in the HASYv2 dataset at all, so the CNN has no idea what it is. But `=` is just two horizontal bars stacked vertically, and our contour-based segmentation picks those up as two separate `-` symbols. So the fix: if two consecutive predictions are both `-` and their bounding boxes are at roughly the same x position (vertically aligned), merge them into `=`. Check if the x-center distance is less than 60% of the average bar width.

2. **x/X/times confusion** — from the step 3 confusion matrix, the CNN mixes up `x`, `X`, and `times` constantly. Rules:
   - `times` prediction → always treat as multiplication `*`
   - `X` between two operands (like `3 X 4`) → treat as `*`
   - Otherwise → treat as variable `x`
   - `div` → `/`
   
   This isn't perfect but covers the common cases. If someone writes `X + 3 = 7`, the `X` at the start isn't between two operands so it gets treated as variable `x` which is probably correct.

3. **Implicit multiplication** — humans write `2x` but SymPy needs `2*x`. Whenever a digit appears right before a variable (or vice versa), we insert `*` between them. Same for things like `2(x+1)` → `2*(x+1)`.

After all the preprocessing, the equation string goes to **SymPy** which handles the actual math:
- Pure arithmetic (`3+4`) → evaluates to `7`
- Linear equations (`2*x+3=7`) → solves for `x=2`
- Verification (`3+4=7`) → checks if both sides are equal

After all that, SymPy handles the actual math. We also added **line splitting** so the system can handle multiple equations in one image. Uses horizontal projection profile - sum the white pixels per row, find the gaps between lines, split and process each line separately. Then solve them together as a system of equations.

### Test Results

Tested on synthetic samples with the full pipeline (image -> preprocess -> segment -> classify -> parse -> solve):

| Sample | Equation | Parsed | Result | Status |
|--------|----------|--------|--------|--------|
| `synthetic_eq1.png` | `2x + 3 = 7` | `2*x+3=7` | x = 2 | Pass |
| `system_eq1.png` | `3x - y = 7` / `2x + y = 8` | `3*x-y=7` / `2*x+y=8` | x=3, y=2 | Pass |
| `sample_arithmetic.png` | `5 + 3` | `5+3` | 8 | Pass |
| `sample_system2.png` | `x + 2y = 12` / `3x - y = 1` | `x+2*y=12` / `3*x-y=1` | x=2, y=5 | Pass |
| `sample_mixed.png` | `9 - 4 + 2` | `9-4+2` | 7 | Pass |
| `sample_linear.png` | `4x - 8 = 0` | `4*x-8=8` | x=4 | Fail (CNN: 0->8) |

5/6 passed. The one failure is a CNN recognition issue (`0` misread as `8`), not the parser.

Files: `notebooks/04_equation_parser.ipynb`, `src/solver.py`, `src/segment.py` (line splitting)

## Future Improvements

Things that could push this system further:

1. **Better CNN model** — the current model struggles with certain handwriting styles. `0` vs `8`, `y` vs `8`, and the whole `x`/`X`/`times` thing. Training on a larger, more diverse handwriting dataset (like CROHME or custom collected data) would help a lot.

2. **Multi-digit numbers** — right now each digit is a separate segment. `12` shows up as `1` and `2` as separate symbols. Need spacing-based logic to merge consecutive digits into multi-digit numbers.

3. **Quadratic and higher-order equations** — currently handles linear equations only. Adding support for `x^2` would require detecting superscript positioning from bounding box y-coordinates.

4. **Fractions** — handwritten fractions (horizontal bar with numerator on top, denominator below) need special segmentation logic. The horizontal bar overlaps with minus and equals.

5. **Better `=` detection** — currently using aspect ratio heuristics which is fragile. Could train a separate small classifier just for operator symbols, or add `=` to the training data with custom samples.

6. **Confidence-based rejection** — when the CNN confidence is below a threshold, flag it as uncertain instead of guessing. Show the user which symbols it's unsure about.

7. **Web interface** — wrap the whole thing in a Flask/Streamlit app where you can upload or take a photo and get the solution back. Way more user-friendly than CLI.

8. **Real handwriting data collection** — collect actual handwritten equation photos from students, label them, and use that as a test set. Synthetic images are too clean compared to real photos with shadows, angles, and messy handwriting.
