import cv2
import numpy as np


def preprocess(image_path):
    """takes a photo of handwritten equation, returns clean binary image"""
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f'cant load {image_path}')

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(blurred)

    binary = cv2.adaptiveThreshold(enhanced, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                    cv2.THRESH_BINARY_INV, 15, 10)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    cleaned = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=1)
    cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, kernel, iterations=1)

    return cleaned


if __name__ == '__main__':
    import sys
    if len(sys.argv) < 2:
        print('usage: python preprocess.py <image_path>')
        sys.exit(1)
    result = preprocess(sys.argv[1])
    cv2.imwrite('output_binary.png', result)
    print(f'saved output_binary.png ({result.shape})')
