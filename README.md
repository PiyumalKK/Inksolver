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
