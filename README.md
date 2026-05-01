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
