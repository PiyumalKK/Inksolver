import cv2
import numpy as np


def merge_overlapping_boxes(boxes, overlap_thresh=0.5):
    """merge boxes that overlap horizontally — handles = sign, i dot, etc"""
    if not boxes:
        return []

    boxes = sorted(boxes, key=lambda b: b[0])
    merged = [list(boxes[0])]

    for box in boxes[1:]:
        x, y, w, h = box
        prev = merged[-1]
        px, py, pw, ph = prev

        # horizontal overlap check
        overlap_start = max(px, x)
        overlap_end = min(px + pw, x + w)
        overlap_width = max(0, overlap_end - overlap_start)
        min_width = min(pw, w)

        if min_width > 0 and overlap_width / min_width > overlap_thresh:
            new_x = min(px, x)
            new_y = min(py, y)
            new_w = max(px + pw, x + w) - new_x
            new_h = max(py + ph, y + h) - new_y
            merged[-1] = [new_x, new_y, new_w, new_h]
        else:
            merged.append(list(box))

    return [tuple(b) for b in merged]


def extract_characters(binary_img, boxes, target_size=45, padding=4):
    """crop each box, pad to square, resize to target_size"""
    chars = []
    for (x, y, w, h) in boxes:
        y1 = max(0, y - padding)
        y2 = min(binary_img.shape[0], y + h + padding)
        x1 = max(0, x - padding)
        x2 = min(binary_img.shape[1], x + w + padding)
        crop = binary_img[y1:y2, x1:x2]

        # pad to square so we don't stretch the character
        ch, cw = crop.shape[:2]
        side = max(ch, cw)
        square = np.zeros((side, side), dtype=np.uint8)
        off_y = (side - ch) // 2
        off_x = (side - cw) // 2
        square[off_y:off_y+ch, off_x:off_x+cw] = crop

        resized = cv2.resize(square, (target_size, target_size), interpolation=cv2.INTER_AREA)
        chars.append(resized)

    return chars


def segment(binary_img, min_area_ratio=0.001, overlap_thresh=0.5, target_size=45, padding=4):
    """takes binary image, returns (list of character images, list of bounding boxes) sorted L-R"""
    contours, _ = cv2.findContours(binary_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # filter tiny contours
    img_area = binary_img.shape[0] * binary_img.shape[1]
    min_area = img_area * min_area_ratio
    boxes = []
    for cnt in contours:
        if cv2.contourArea(cnt) < min_area:
            continue
        boxes.append(cv2.boundingRect(cnt))

    # merge overlapping boxes (= sign, etc)
    merged = merge_overlapping_boxes(boxes, overlap_thresh)

    # sort left to right
    merged = sorted(merged, key=lambda b: b[0])

    # extract character images
    chars = extract_characters(binary_img, merged, target_size, padding)

    return chars, merged


def split_lines(binary_img, min_gap_ratio=0.05):
    """split image into horizontal lines using projection profile.
    looks for horizontal gaps (rows with no/few white pixels) to find line breaks."""
    h, w = binary_img.shape
    # horizontal projection - sum white pixels per row
    projection = np.sum(binary_img, axis=1) / 255

    # threshold - rows with very few white pixels are gaps
    row_threshold = w * 0.01
    is_content = projection > row_threshold

    # find line regions (contiguous blocks of content rows)
    lines = []
    in_line = False
    start = 0

    for i in range(h):
        if is_content[i] and not in_line:
            start = i
            in_line = True
        elif not is_content[i] and in_line:
            lines.append((start, i))
            in_line = False
    if in_line:
        lines.append((start, h))

    # filter out tiny strips (noise)
    min_height = h * 0.05
    lines = [(y1, y2) for y1, y2 in lines if (y2 - y1) > min_height]

    if len(lines) <= 1:
        # single line or couldn't split - return whole image
        return [binary_img]

    # crop each line with a bit of padding
    pad = 5
    result = []
    for y1, y2 in lines:
        y1 = max(0, y1 - pad)
        y2 = min(h, y2 + pad)
        result.append(binary_img[y1:y2, :])

    return result
    import sys
    import os
    from preprocess import preprocess

    if len(sys.argv) < 2:
        print('usage: python segment.py <image_path>')
        sys.exit(1)

    binary = preprocess(sys.argv[1])
    chars, boxes = segment(binary)
    print(f'found {len(chars)} characters')

    out_dir = os.path.join(os.path.dirname(sys.argv[1]), '..', 'segments')
    os.makedirs(out_dir, exist_ok=True)
    for i, char_img in enumerate(chars):
        path = os.path.join(out_dir, f'char_{i}.png')
        cv2.imwrite(path, char_img)
        print(f'  saved {path}')
