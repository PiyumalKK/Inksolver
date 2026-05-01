import cv2
import numpy as np
import json
import os

# will be set when load_model() is called
_model = None
_label_map = None


def load_model(model_path='models/symbol_classifier.h5', label_path='models/label_map.json'):
    """load the trained CNN and label mapping"""
    global _model, _label_map

    # lazy import so tensorflow doesn't load until needed
    from tensorflow import keras

    _model = keras.models.load_model(model_path)

    with open(label_path, 'r') as f:
        _label_map = json.load(f)
    # keys come as strings from json, convert to int
    _label_map = {int(k): v for k, v in _label_map.items()}

    print(f'loaded model from {model_path}')
    print(f'classes: {list(_label_map.values())}')


def predict(char_img):
    """predict symbol from a 45x45 grayscale character image"""
    if _model is None:
        raise RuntimeError('call load_model() first')

    # normalize
    img = char_img.astype(np.float32) / 255.0
    img = img.reshape(1, 45, 45, 1)

    pred = _model.predict(img, verbose=0)
    class_idx = int(np.argmax(pred))
    confidence = float(pred[0][class_idx])
    label = _label_map[class_idx]

    return label, confidence


def predict_batch(char_images):
    """predict symbols for a list of character images"""
    if _model is None:
        raise RuntimeError('call load_model() first')

    batch = np.array([img.astype(np.float32) / 255.0 for img in char_images])
    batch = batch.reshape(-1, 45, 45, 1)

    preds = _model.predict(batch, verbose=0)
    results = []
    for pred in preds:
        idx = int(np.argmax(pred))
        results.append((_label_map[idx], float(pred[idx])))

    return results


if __name__ == '__main__':
    import sys
    sys.path.insert(0, os.path.dirname(__file__))
    from preprocess import preprocess
    from segment import segment

    if len(sys.argv) < 2:
        print('usage: python model.py <image_path>')
        sys.exit(1)

    load_model()
    binary = preprocess(sys.argv[1])
    chars, boxes = segment(binary)
    results = predict_batch(chars)

    print(f'\nrecognized {len(results)} symbols:')
    for i, (label, conf) in enumerate(results):
        print(f'  [{i}] {label} ({conf:.2f})')
