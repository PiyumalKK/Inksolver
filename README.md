# InkSolver — Handwritten Mathematical Equation Solver

An end-to-end computer vision system that captures handwritten math equations from images, segments and classifies each symbol using a CNN, and automatically solves the equation using symbolic algebra.

> **Course:** EE7204/EC7205 Computer Vision & Image Processing — University of Ruhuna  
> **Model accuracy:** 95.66% on 84 symbol classes (CROHME dataset)

## Pipeline Overview

```
┌─────────┐    ┌───────────┐    ┌──────────┐    ┌──────────────┐    ┌─────────┐    ┌───────┐    ┌───────┐
│  Photo  │───▶│ Grayscale │───▶│  Blur +  │───▶│  Adaptive    │───▶│ Morpho- │───▶│Contour│───▶│  CNN  │
│ (input) │    │ + CLAHE   │    │ Denoise  │    │  Threshold   │    │  logy   │    │Segment│    │Predict│
└─────────┘    └───────────┘    └──────────┘    └──────────────┘    └─────────┘    └───────┘    └───────┘
                                                                                                     │
                                        ┌──────────┐    ┌───────────┐    ┌──────────────┐           │
                                        │  SOLVE   │◀───│   Build   │◀───│  Ambiguity   │◀──────────┘
                                        │ (SymPy)  │    │  Equation │    │  Resolution  │
                                        └──────────┘    └───────────┘    └──────────────┘
```

## Tech Stack

| Component | Library | Purpose |
|-----------|---------|---------|
| Image Processing | OpenCV 4.x | Grayscale, blur, CLAHE, adaptive threshold, morphology, contours |
| Deep Learning | TensorFlow/Keras | CNN inference for symbol classification (84 classes) |
| Symbolic Math | SymPy | Equation parsing and algebraic solving |
| Web Interface | Flask | Upload image → get solution (browser-based) |
| Training | Google Colab (T4 GPU) | Model training on CROHME dataset |
| Language | Python 3.10+ | All modules |

## Setup

```bash
git clone https://github.com/PiyumalKK/Inksolver.git
cd Inksolver
```

### Create & Activate Virtual Environment

**Windows (PowerShell):**
```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

**Windows (CMD):**
```cmd
python -m venv venv
venv\Scripts\activate.bat
```

**Linux / macOS:**
```bash
python3 -m venv venv
source venv/bin/activate
```

### Install Dependencies

```bash
pip install -r requirements.txt
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
InkSolver/
├── app.py                  # Flask web application
├── requirements.txt        # Python dependencies
├── src/
│   ├── preprocess.py       # Image → clean binary (grayscale, blur, CLAHE, threshold, morphology)
│   ├── segment.py          # Binary → individual character crops (contours, merge, sort, resize)
│   ├── model.py            # CNN inference wrapper (load model, predict symbols)
│   └── solver.py           # Symbol sequence → equation → solution (SymPy)
├── models/
│   ├── symbol_classifier_crohme.h5   # Trained CNN weights (84 classes)
│   └── label_map_crohme.json         # Class index → symbol name mapping
├── notebooks/
│   ├── 01_preprocessing.ipynb        # Preprocessing experiments & visualization
│   ├── 02_segmentation.ipynb         # Segmentation development & testing
│   ├── 03_cnn_training_v1.ipynb      # First attempt (HASYv2, 15 classes)
│   ├── 03_cnn_training_v2.ipynb      # Final model (CROHME, 84 classes)
│   └── 04_equation_parser.ipynb      # Parsing & solving logic
├── static/                 # Web UI (HTML/CSS/JS)
├── data/
│   ├── dataset/            # Training data info (see README inside)
│   ├── raw_samples/        # Test images
│   └── segments/           # Extracted character crops
├── uploads/                # Flask upload directory
└── results/                # Output samples
```

## Web Application

InkSolver includes a Flask-based web interface for easy use:

```bash
python app.py
```
Open http://localhost:5000 in your browser → upload a photo → get the solution.

The web app uses a **dual-pipeline approach**: it tries two preprocessing variants (full CLAHE pipeline + simple threshold) and picks the result with higher confidence and mathematical validity.

## How It Works (Technical Details)

### Stage 1 — Image Preprocessing (`src/preprocess.py`)

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

### Stage 2 — Character Segmentation (`src/segment.py`)

Now that we have a clean binary image, we need to chop it up into individual characters.

The approach:
1. **Find contours** — `cv2.findContours` with `RETR_EXTERNAL` to grab only the outermost blobs. Each connected white region = one contour
2. **Bounding boxes** — get the rectangle around each contour. Filter out anything smaller than 0.1% of the image area (noise specks that survived preprocessing)
3. **Merge overlapping boxes** — this was the tricky part. The `=` sign shows up as two separate horizontal bars, so we get two contours for one symbol. Same issue with `i`, `j`, etc. Fix: if two boxes overlap horizontally by more than 50% of the smaller box's width, merge them into one bigger box
4. **Sort left to right** — sort by x coordinate so characters are in reading order
5. **Crop and resize** — cut each character out, pad it to a square (so it doesn't get stretched), then resize to 45x45. The padding preserves aspect ratio which matters for the CNN later

Files: `notebooks/02_segmentation.ipynb`, `src/segment.py`

### Stage 3 — CNN Symbol Classifier (`src/model.py`)

This is the "brain" — a CNN that looks at a 45x45 character crop and tells us what symbol it is.

Used two **CROHME**-based datasets combined:
- **xainano/handwrittenmathsymbols** — 100k+ isolated symbol images (45x45 JPG) extracted from the CROHME competition (ICDAR benchmark for handwritten math recognition)
- **sagyamthapa/handwritten-math-symbols** — 10k real-world handwritten symbols (400x400, resized to 45x45), specifically improved for real-world scenarios

Combined dataset covers **84 classes** including digits 0-9, operators (+, -, =, ×, ÷), variables (x, y, z, X, Y), brackets, Greek letters, trig functions, and more. The `=` sign is now in the training data directly — no more heuristic-only detection.

Capped dataset 1 at 3000 samples per class to balance with the smaller dataset 2.

**Architecture:**
- 3 conv blocks: Conv2D → BatchNorm → ReLU → MaxPool → Dropout
- Filters: 32 → 64 → 128
- Dense: 256 → softmax
- Input: 45x45x1 grayscale

**Training:**
- Data augmentation: small rotations (±10°), shifts (10%), zoom (10%), shear
- Adam optimizer with ReduceLROnPlateau
- EarlyStopping with patience=10
- Trained on Google Colab T4 GPU, 50 epochs
- Final val accuracy: **95.66%** across 84 classes

The `src/model.py` module wraps the trained model for inference — load once, then predict on character crops.

Files: `notebooks/03_cnn_training_v2.ipynb` (Colab), `src/model.py`, `models/symbol_classifier_crohme.h5`, `models/label_map_crohme.json`

### Stage 4 — Equation Parsing & Solving (`src/solver.py`)

This is where the recognized symbols actually become a solvable equation.

Three main problems to deal with:

1. **Equals sign detection** — the CNN can now recognize `=` directly from CROHME training data. As a fallback, if segmentation splits the two bars into separate contours, the parser still merges two consecutive `-` predictions at the same x position into `=`.

2. **x/X/times confusion** — from the step 3 confusion matrix, the CNN mixes up `x`, `X`, and `times` constantly. All three are handled the same way:
   - If between two operands (like `3 x 4`) → treat as multiplication `*`
   - Otherwise → treat as variable `x`
   - `div` → `/`
   
   This works because in a math context, whether someone writes `x`, `X`, or the multiplication symbol `×`, the meaning depends on position. A letter between two numbers or after a closing bracket is almost always multiplication.

3. **Implicit multiplication** — humans write `2x` but SymPy needs `2*x`. Whenever a digit appears right before a variable (or vice versa), we insert `*` between them. Same for things like `2(x+1)` → `2*(x+1)`.

After all the preprocessing, the equation string goes to **SymPy** which handles the actual math:
- Pure arithmetic (`3+4`) → evaluates to `7`
- Linear equations (`2*x+3=7`) → solves for `x=2`
- Verification (`3+4=7`) → checks if both sides are equal

The system also supports **line splitting** to handle multiple equations in one image. Uses horizontal projection profile — sum the white pixels per row, find the gaps between lines, split and process each line separately. Then solve them together as a system of equations.

### Test Results

Tested on synthetic and handwritten samples with the full pipeline (image -> preprocess -> segment -> classify -> parse -> solve):

| Sample | Equation | Parsed | Result | Status |
|--------|----------|--------|--------|--------|
| `synthetic_eq1.png` | `2x + 3 = 7` | `2*x+3=7` | x = 2 | Pass |
| `system_eq1.png` | `3x - y = 7` / `2x + y = 8` | `3*x-y=7` / `2*x+y=8` | x=3, y=2 | Pass |
| `sample_arithmetic.png` | `5 + 3` | `5+3` | 8 | Pass |
| `sample_mixed.png` | `9 - 4 + 2` | `9-4+2` | 7 | Pass |
| `sample_linear.png` | `4x - 8 = 0` | `4*x-8=0` | x=2 | Pass |
| `handwritten.jpg` | `3x - y = 7` / `2x + y = 8` | `3*x-y=7` / `2*x+y=8` | x=3, y=2 | Pass |

6/6 passed on the test samples.

Files: `notebooks/04_equation_parser.ipynb`, `src/solver.py`, `src/segment.py` (line splitting)

## Datasets

- **xainano/handwrittenmathsymbols** — 100k+ CROHME symbols: https://www.kaggle.com/datasets/xainano/handwrittenmathsymbols
- **sagyamthapa/handwritten-math-symbols** — 10k supplementary symbols: https://www.kaggle.com/datasets/sagyamthapa/handwritten-math-symbols
