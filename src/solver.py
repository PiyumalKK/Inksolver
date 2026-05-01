import re
import sympy as sp


def detect_equals(predictions, boxes):
    """
    the CNN cant recognize '=' since it wasnt in HASYv2.
    two ways it can show up:
    1) segmentation keeps the two bars separate -> two '-' predictions stacked vertically
    2) segmentation merges them -> one wide box predicted as '-' or 'div'
    """
    if not predictions:
        return predictions, boxes

    new_preds = []
    new_boxes = []
    skip_next = False

    for i in range(len(predictions)):
        if skip_next:
            skip_next = False
            continue

        label, conf = predictions[i]
        x, y, w, h = boxes[i]

        # case 1: two separate '-' bars stacked vertically
        if label == '-' and i + 1 < len(predictions):
            next_label, next_conf = predictions[i + 1]
            nx, ny, nw, nh = boxes[i + 1]

            x_center = x + w / 2
            nx_center = nx + nw / 2
            x_diff = abs(x_center - nx_center)
            avg_w = (w + nw) / 2

            if next_label == '-' and x_diff < avg_w * 0.6:
                merged_x = min(x, nx)
                merged_y = min(y, ny)
                merged_w = max(x + w, nx + nw) - merged_x
                merged_h = max(y + h, ny + nh) - merged_y
                new_preds.append(('=', min(conf, next_conf)))
                new_boxes.append((merged_x, merged_y, merged_w, merged_h))
                skip_next = True
                continue

        # case 2: merged into one wide box - CNN sees it as '-' or 'div'
        # '=' is wide and short, aspect ratio > 1.5
        if label in ('-', 'div') and h > 0 and w / h > 1.5:
            new_preds.append(('=', conf))
            new_boxes.append(boxes[i])
            continue

        new_preds.append((label, conf))
        new_boxes.append(boxes[i])

    return new_preds, new_boxes


def resolve_ambiguity(predictions):
    """
    handle x/X/times confusion.
    if x, X, or times appears between two operands -> treat as multiplication.
    otherwise -> treat as variable x.
    """
    labels = [p[0] for p in predictions]
    confs = [p[1] for p in predictions]
    resolved = []

    for i, label in enumerate(labels):
        if label in ('X', 'times', 'x'):
            prev = labels[i - 1] if i > 0 else None
            nxt = labels[i + 1] if i < len(labels) - 1 else None

            prev_is_operand = prev is not None and (prev.isdigit() or prev in ('x', 'y', ')'))
            next_is_operand = nxt is not None and (nxt.isdigit() or nxt in ('x', 'y', '('))

            if label == 'times':
                # times is always multiplication
                resolved.append(('*', confs[i]))
            elif prev_is_operand and next_is_operand and label == 'X':
                # X between two operands = multiply
                resolved.append(('*', confs[i]))
            else:
                # treat as variable x
                resolved.append(('x', confs[i]))
        elif label == 'div':
            resolved.append(('/', confs[i]))
        else:
            resolved.append((label, confs[i]))

    return resolved


def build_equation(predictions):
    """
    take resolved predictions and build equation string.
    adds implicit multiplication where needed (e.g. 2x -> 2*x).
    """
    labels = [p[0] for p in predictions]
    parts = []

    for i, sym in enumerate(labels):
        parts.append(sym)

        # implicit multiplication
        if i < len(labels) - 1:
            nxt = labels[i + 1]
            # digit followed by variable
            if sym.isdigit() and nxt in ('x', 'y'):
                parts.append('*')
            # variable followed by digit (like x2 -> x*2... rare but handle it)
            elif sym in ('x', 'y') and nxt.isdigit():
                parts.append('*')
            # closing paren followed by digit or variable
            elif sym == ')' and (nxt.isdigit() or nxt in ('x', 'y', '(')):
                parts.append('*')
            # digit or variable followed by opening paren
            elif (sym.isdigit() or sym in ('x', 'y')) and nxt == '(':
                parts.append('*')

    return ''.join(parts)


def solve_equation(eq_str):
    """
    solve using sympy.
    if there's '=' -> solve the equation.
    if no '=' -> just evaluate the arithmetic.
    """
    x, y = sp.symbols('x y')

    try:
        if '=' in eq_str:
            left, right = eq_str.split('=', 1)
            lhs = sp.sympify(left)
            rhs = sp.sympify(right)
            expr = lhs - rhs
        else:
            # no equals sign -> just evaluate
            expr = sp.sympify(eq_str)
            variables = expr.free_symbols
            if not variables:
                # pure arithmetic
                result = float(expr)
                if result == int(result):
                    result = int(result)
                return {
                    'type': 'arithmetic',
                    'expression': eq_str,
                    'result': result,
                }

        variables = expr.free_symbols

        if len(variables) == 0:
            # equation like 3+4=7 -> check if it's true
            val = float(expr)
            return {
                'type': 'verification',
                'expression': eq_str,
                'result': val == 0,
            }
        elif len(variables) == 1:
            var = list(variables)[0]
            solutions = sp.solve(expr, var)
            sols = []
            for s in solutions:
                v = float(s)
                if v == int(v):
                    v = int(v)
                sols.append(v)
            return {
                'type': 'equation',
                'expression': eq_str,
                'variable': str(var),
                'solutions': sols,
            }
        else:
            # multiple variables, just simplify
            simplified = sp.simplify(expr)
            return {
                'type': 'multi_variable',
                'expression': eq_str,
                'simplified': str(simplified),
                'variables': [str(v) for v in variables],
            }

    except Exception as e:
        return {
            'type': 'error',
            'expression': eq_str,
            'error': str(e),
        }


def solve_from_predictions(predictions, boxes):
    """
    full pipeline: predictions + boxes -> detect '=' -> resolve ambiguity -> build -> solve.
    """
    preds, bxs = detect_equals(predictions, boxes)
    resolved = resolve_ambiguity(preds)
    eq_str = build_equation(resolved)
    result = solve_equation(eq_str)
    result['symbols'] = [p[0] for p in resolved]
    return result


if __name__ == '__main__':
    import sys
    import os
    sys.path.insert(0, os.path.dirname(__file__))
    from preprocess import preprocess
    from segment import segment
    from model import load_model, predict_batch

    if len(sys.argv) < 2:
        print('usage: python solver.py <image_path>')
        sys.exit(1)

    load_model()
    binary = preprocess(sys.argv[1])
    chars, boxes = segment(binary)
    predictions = predict_batch(chars)

    print(f'raw predictions: {[p[0] for p in predictions]}')

    result = solve_from_predictions(predictions, boxes)
    print(f'equation: {result.get("expression", "?")}')
    print(f'symbols: {result["symbols"]}')

    if result['type'] == 'arithmetic':
        print(f'result: {result["result"]}')
    elif result['type'] == 'equation':
        print(f'{result["variable"]} = {result["solutions"]}')
    elif result['type'] == 'verification':
        print(f'equation is {"TRUE" if result["result"] else "FALSE"}')
    elif result['type'] == 'error':
        print(f'error: {result["error"]}')
    else:
        print(f'simplified: {result.get("simplified", "?")}')
