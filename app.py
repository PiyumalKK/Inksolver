import os
import sys
import re
import base64
import tempfile
import cv2
import numpy as np
from flask import Flask, request, jsonify, send_from_directory

# add src/ to path so we can import the pipeline modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from preprocess import preprocess
from segment import segment, split_lines
from model import load_model, predict_batch
from solver import detect_equals, resolve_ambiguity, build_equation, solve_equation, solve_system

app = Flask(__name__, static_folder='static', static_url_path='/static')

# ── globals ──────────────────────────────────────────────────────────
_model_loaded = False
UPLOAD_DIR = os.path.join(os.path.dirname(__file__), 'uploads')
os.makedirs(UPLOAD_DIR, exist_ok=True)

SAMPLE_DIR = os.path.join(os.path.dirname(__file__), 'data', 'raw_samples')


def _ensure_model():
    """lazy-load the CNN model on first request"""
    global _model_loaded
    if not _model_loaded:
        model_path = os.path.join(os.path.dirname(__file__), 'models', 'symbol_classifier.h5')
        label_path = os.path.join(os.path.dirname(__file__), 'models', 'label_map.json')
        load_model(model_path, label_path)
        _model_loaded = True


def _img_to_base64(img):
    """encode a cv2 image as base64 PNG string"""
    _, buffer = cv2.imencode('.png', img)
    return base64.b64encode(buffer).decode('utf-8')


def _draw_boxes(binary_img, boxes, labels=None):
    """draw bounding boxes (and optional labels) on the image"""
    vis = cv2.cvtColor(binary_img, cv2.COLOR_GRAY2BGR)
    for i, (x, y, w, h) in enumerate(boxes):
        cv2.rectangle(vis, (x, y), (x + w, y + h), (0, 220, 255), 2)
        if labels and i < len(labels):
            cv2.putText(vis, labels[i], (x, max(y - 6, 14)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 220, 255), 1)
    return vis


def _read_image_robust(image_path):
    """read an image, handling PNG transparency and converting to BGR."""
    # try reading with alpha channel first (for PNGs with transparency)
    img = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
    if img is None:
        return None

    # handle alpha channel — composite onto white background
    if len(img.shape) == 2:
        # already grayscale
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    elif img.shape[2] == 4:
        # RGBA → composite onto white
        alpha = img[:, :, 3] / 255.0
        white_bg = np.ones_like(img[:, :, :3], dtype=np.uint8) * 255
        for c in range(3):
            white_bg[:, :, c] = (img[:, :, c] * alpha + 255 * (1 - alpha)).astype(np.uint8)
        img = white_bg

    return img


def _upscale_if_small(img, min_height=120):
    """upscale small images so the CNN (45x45 crops) has enough pixels to work with."""
    h, w = img.shape[:2]
    if h < min_height:
        scale = min_height / h
        new_w = int(w * scale)
        new_h = int(h * scale)
        img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
    return img


def _preprocess_simple(image_path):
    """robust preprocessing for clean/digital/typeset images.
    tries multiple threshold strategies and returns the one that
    produces the most character-like contours."""
    img = _read_image_robust(image_path)
    if img is None:
        return None

    img = _upscale_if_small(img)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    candidates = []

    # strategy 1: Otsu (no blur) — best for clean digital text
    _, bin1 = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    candidates.append(bin1)

    # strategy 2: Otsu with light blur
    blurred = cv2.GaussianBlur(gray, (3, 3), 0)
    _, bin2 = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    candidates.append(bin2)

    # strategy 3: fixed threshold at multiple levels (for low-contrast images)
    for thresh_val in [100, 128, 160]:
        _, bin_fixed = cv2.threshold(gray, thresh_val, 255, cv2.THRESH_BINARY_INV)
        candidates.append(bin_fixed)

    # strategy 4: adaptive threshold with no morphology (gentle)
    bin_adapt = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                       cv2.THRESH_BINARY_INV, 21, 8)
    candidates.append(bin_adapt)

    # pick the candidate that produces the most reasonable contours
    best = None
    best_score = -1

    for binary in candidates:
        # light morphological close to connect broken strokes
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
        cleaned = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=1)

        # count contours of reasonable size
        contours, _ = cv2.findContours(cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        img_area = cleaned.shape[0] * cleaned.shape[1]
        min_area = img_area * 0.001
        max_area = img_area * 0.5
        valid = [c for c in contours if min_area < cv2.contourArea(c) < max_area]

        # score: prefer 3-10 characters (typical equation length)
        n = len(valid)
        if n == 0:
            score = 0
        elif 3 <= n <= 15:
            score = n * 10  # bonus for reasonable count
        else:
            score = n

        if score > best_score:
            best_score = score
            best = cleaned

    return best


def _validate_equation(eq_str):
    """check if a parsed equation string looks valid enough to send to SymPy.
    returns (is_valid, reason) tuple."""
    if not eq_str or not eq_str.strip():
        return False, 'Empty equation'

    # must contain at least one digit or variable
    if not re.search(r'[0-9xy]', eq_str):
        return False, 'No digits or variables found'

    # check for consecutive operators (like ==, ++, //, etc.)
    if re.search(r'[=+\-*/]{3,}', eq_str):
        return False, 'Too many consecutive operators'

    # equation starting or ending with operator (except minus for negative)
    if re.match(r'^[=+*/]', eq_str):
        return False, 'Equation starts with an operator'
    if re.search(r'[=+\-*/]$', eq_str):
        return False, 'Equation ends with an operator'

    # if there's '=', both sides must have content
    if '=' in eq_str:
        parts = eq_str.split('=')
        for part in parts:
            if not part.strip():
                return False, 'Empty side of equation'

    return True, 'OK'


def _avg_confidence(predictions):
    """compute average confidence of predictions"""
    if not predictions:
        return 0.0
    return sum(p[1] for p in predictions) / len(predictions)


def _try_pipeline(binary_img):
    """run segment → classify → parse on a binary image.
    returns (eq_str, predictions, boxes, resolved, avg_conf) or None on failure."""
    from segment import segment
    from model import predict_batch

    chars, boxes = segment(binary_img)
    if not chars:
        return None

    predictions = predict_batch(chars)
    preds_eq, boxes_eq = detect_equals(predictions, boxes)
    resolved = resolve_ambiguity(preds_eq)
    eq_str = build_equation(resolved)
    avg_conf = _avg_confidence(predictions)

    return {
        'eq_str': eq_str,
        'predictions': predictions,
        'boxes': boxes,
        'resolved': resolved,
        'avg_conf': avg_conf,
        'chars': chars,
    }


def _process_image(image_path):
    """run the full pipeline and return structured results"""
    _ensure_model()

    # read original image (handles transparency, etc.)
    original = _read_image_robust(image_path)
    if original is None:
        return {'success': False, 'error': 'Could not read image file'}

    # try both preprocessing approaches and pick the best one
    # approach 1: original pipeline (tuned for handwritten photos)
    binary_orig = preprocess(image_path)
    # approach 2: simple pipeline (tuned for digital/typeset images)
    binary_simple = _preprocess_simple(image_path)

    # step 2: split into lines
    lines_orig = split_lines(binary_orig)

    if len(lines_orig) == 1:
        # ── single equation ─────────────────────────────────────────
        # try original pipeline
        pipeline_orig = _try_pipeline(lines_orig[0])

        # try simple pipeline
        pipeline_simple = None
        lines_simple = None
        if binary_simple is not None:
            lines_simple = split_lines(binary_simple)
            if len(lines_simple) >= 1:
                pipeline_simple = _try_pipeline(lines_simple[0])

        # pick the best pipeline result
        pipeline = None
        binary = binary_orig
        lines = lines_orig

        if pipeline_orig and pipeline_simple:
            eq_orig = pipeline_orig['eq_str']
            eq_simple = pipeline_simple['eq_str']
            valid_orig, _ = _validate_equation(eq_orig)
            valid_simple, _ = _validate_equation(eq_simple)

            if valid_simple and not valid_orig:
                # simple is valid, original isn't — use simple
                pipeline = pipeline_simple
                binary = binary_simple
                lines = lines_simple
            elif not valid_simple and valid_orig:
                # original is valid, simple isn't — use original
                pipeline = pipeline_orig
            elif valid_simple and valid_orig:
                # both valid — prefer original (tuned for handwritten) unless
                # simple has significantly higher confidence (>10% margin)
                if pipeline_simple['avg_conf'] > pipeline_orig['avg_conf'] + 0.10:
                    pipeline = pipeline_simple
                    binary = binary_simple
                    lines = lines_simple
                else:
                    pipeline = pipeline_orig
            else:
                # neither valid — use whichever has higher confidence
                if pipeline_simple['avg_conf'] > pipeline_orig['avg_conf']:
                    pipeline = pipeline_simple
                    binary = binary_simple
                    lines = lines_simple
                else:
                    pipeline = pipeline_orig
        elif pipeline_orig:
            pipeline = pipeline_orig
        elif pipeline_simple:
            pipeline = pipeline_simple
            binary = binary_simple
            lines = lines_simple

        if not pipeline:
            return {'success': False, 'error': 'No characters detected in the image. Try a clearer image with dark ink on a white background.'}

        eq_str = pipeline['eq_str']
        predictions = pipeline['predictions']
        boxes = pipeline['boxes']
        resolved = pipeline['resolved']

        # build recognition details
        raw_symbols = [{'symbol': p[0], 'confidence': round(p[1] * 100, 1)} for p in predictions]
        avg_conf = round(pipeline['avg_conf'] * 100, 1)

        # validate before solving
        is_valid, reason = _validate_equation(eq_str)
        if not is_valid:
            segmented_vis = _draw_boxes(lines[0], boxes, [p[0] for p in predictions])
            return {
                'success': True,
                'mode': 'single',
                'equation': eq_str,
                'type': 'error',
                'result': {
                    'error': f'Could not parse equation: {reason}. '
                             f'The CNN recognized "{eq_str}" with {avg_conf}% avg confidence. '
                             f'Try a clearer image with dark ink on a white background.'
                },
                'steps': {
                    'characters_found': len(pipeline['chars']),
                    'recognition': raw_symbols,
                    'parsed_equation': eq_str,
                    'resolved_symbols': [p[0] for p in resolved],
                    'avg_confidence': avg_conf,
                },
                'images': {
                    'original': _img_to_base64(original),
                    'preprocessed': _img_to_base64(binary),
                    'segmented': _img_to_base64(segmented_vis),
                },
            }

        result = solve_equation(eq_str)

        # draw segmented image with boxes
        seg_labels = [p[0] for p in resolved]
        segmented_vis = _draw_boxes(lines[0], boxes, [p[0] for p in predictions])

        response = {
            'success': True,
            'mode': 'single',
            'equation': eq_str,
            'type': result.get('type', 'unknown'),
            'steps': {
                'characters_found': len(pipeline['chars']),
                'recognition': raw_symbols,
                'parsed_equation': eq_str,
                'resolved_symbols': seg_labels,
                'avg_confidence': avg_conf,
            },
            'images': {
                'original': _img_to_base64(original),
                'preprocessed': _img_to_base64(binary),
                'segmented': _img_to_base64(segmented_vis),
            },
        }

        # format result based on type
        if result['type'] == 'arithmetic':
            response['result'] = {'value': result['result']}
        elif result['type'] == 'equation':
            response['result'] = {
                'variable': result['variable'],
                'solutions': result['solutions'],
            }
        elif result['type'] == 'verification':
            response['result'] = {'is_true': result['result']}
        elif result['type'] == 'multi_variable':
            response['result'] = {
                'simplified': result['simplified'],
                'variables': result['variables'],
            }
        elif result['type'] == 'error':
            response['result'] = {
                'error': f'{result["error"]}. Try a clearer image with dark ink on white background.'
            }

        return response

    else:
        # ── system of equations ──────────────────────────────────────
        eq_strings = []
        all_recognition = []

        for i, line_img in enumerate(lines_orig):
            chars, boxes = segment(line_img)
            if not chars:
                continue
            predictions = predict_batch(chars)
            raw_symbols = [{'symbol': p[0], 'confidence': round(p[1] * 100, 1)} for p in predictions]
            all_recognition.append({'line': i + 1, 'symbols': raw_symbols})

            preds_eq, boxes_eq = detect_equals(predictions, boxes)
            resolved = resolve_ambiguity(preds_eq)
            eq_str = build_equation(resolved)
            eq_strings.append(eq_str)

        if not eq_strings:
            return {'success': False, 'error': 'No equations detected'}

        result = solve_system(eq_strings)

        # draw segmented overview on full binary image
        all_chars_img, all_boxes = segment(binary_orig)
        seg_vis = _draw_boxes(binary_orig, all_boxes) if all_boxes else cv2.cvtColor(binary_orig, cv2.COLOR_GRAY2BGR)

        response = {
            'success': True,
            'mode': 'system',
            'equations': eq_strings,
            'type': result.get('type', 'unknown'),
            'steps': {
                'lines_detected': len(lines_orig),
                'equations': eq_strings,
                'recognition': all_recognition,
            },
            'images': {
                'original': _img_to_base64(original),
                'preprocessed': _img_to_base64(binary_orig),
                'segmented': _img_to_base64(seg_vis),
            },
        }

        if result['type'] == 'system':
            response['result'] = {'solutions': result['solutions']}
        elif result['type'] == 'error':
            response['result'] = {'error': result['error']}

        return response


# ── routes ───────────────────────────────────────────────────────────

@app.route('/')
def index():
    return send_from_directory('static', 'index.html')


@app.route('/api/solve', methods=['POST'])
def solve():
    """accept image upload and return solution"""
    if 'image' not in request.files:
        return jsonify({'success': False, 'error': 'No image file provided'}), 400

    file = request.files['image']
    if file.filename == '':
        return jsonify({'success': False, 'error': 'No file selected'}), 400

    # save to temp file
    ext = os.path.splitext(file.filename)[1] or '.png'
    fd, tmp_path = tempfile.mkstemp(suffix=ext, dir=UPLOAD_DIR)
    os.close(fd)

    try:
        file.save(tmp_path)
        result = _process_image(tmp_path)
        return jsonify(result)
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500
    finally:
        # clean up temp file
        if os.path.exists(tmp_path):
            os.remove(tmp_path)


@app.route('/api/solve-sample', methods=['POST'])
def solve_sample():
    """solve one of the built-in sample images"""
    data = request.get_json()
    if not data or 'filename' not in data:
        return jsonify({'success': False, 'error': 'No filename provided'}), 400

    filename = os.path.basename(data['filename'])  # prevent path traversal
    image_path = os.path.join(SAMPLE_DIR, filename)

    if not os.path.exists(image_path):
        return jsonify({'success': False, 'error': f'Sample not found: {filename}'}), 404

    try:
        result = _process_image(image_path)
        return jsonify(result)
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/api/samples', methods=['GET'])
def list_samples():
    """list available sample images"""
    if not os.path.exists(SAMPLE_DIR):
        return jsonify({'samples': []})

    samples = []
    for f in sorted(os.listdir(SAMPLE_DIR)):
        if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
            path = os.path.join(SAMPLE_DIR, f)
            img = cv2.imread(path)
            thumb = None
            if img is not None:
                # create small thumbnail
                h, w = img.shape[:2]
                scale = 120 / max(h, w)
                thumb_img = cv2.resize(img, (int(w * scale), int(h * scale)))
                thumb = _img_to_base64(thumb_img)

            # human readable name from filename
            name = os.path.splitext(f)[0].replace('_', ' ').title()
            samples.append({
                'filename': f,
                'name': name,
                'thumbnail': thumb,
            })

    return jsonify({'samples': samples})


if __name__ == '__main__':
    print('='*50)
    print('  InkSolver Web App')
    print('  http://localhost:5000')
    print('='*50)
    app.run(debug=True, host='0.0.0.0', port=5000)
