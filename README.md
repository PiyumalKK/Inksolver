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
