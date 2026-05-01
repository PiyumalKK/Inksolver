import re
import sympy as sp


def detect_equals(predictions, boxes):
    """
    fallback '=' detection for when segmentation splits the two bars.
    the CNN can recognize '=' directly, but sometimes segmentation keeps
    the two bars separate -> two '-' predictions stacked vertically.
    """
    if not predictions:
        return predictions, boxes

    # check if model already found '=' — if so, skip wide-bar heuristic
    has_equals = any(p[0] == '=' for p in predictions)

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

        # case 2: wide bar heuristic — only if model didn't find '=' already
        if not has_equals and label in ('-', 'div') and h > 0:
            aspect = w / h
            if 1.5 < aspect < 4.0:
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

            if prev_is_operand and next_is_operand:
                # between two operands = multiplication
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


def solve_system(equations):
    """
    solve a system of equations (e.g. two equations, two unknowns).
    equations is a list of equation strings like ['2*x+y=10', 'x-y=2']
    """
    x, y = sp.symbols('x y')

    try:
        exprs = []
        for eq_str in equations:
            if '=' not in eq_str:
                return {
                    'type': 'error',
                    'expressions': equations,
                    'error': f'no equals sign in: {eq_str}',
                }
            left, right = eq_str.split('=', 1)
            exprs.append(sp.sympify(left) - sp.sympify(right))

        # collect all variables
        all_vars = set()
        for expr in exprs:
            all_vars.update(expr.free_symbols)

        solutions = sp.solve(exprs, list(all_vars))

        if isinstance(solutions, dict):
            # single solution
            result = {}
            for var, val in solutions.items():
                v = float(val)
                if v == int(v):
                    v = int(v)
                result[str(var)] = v
            return {
                'type': 'system',
                'expressions': equations,
                'solutions': result,
            }
        elif isinstance(solutions, list) and solutions:
            return {
                'type': 'system',
                'expressions': equations,
                'solutions': {str(v): float(s) for v, s in zip(all_vars, solutions[0])},
            }
        else:
            return {
                'type': 'error',
                'expressions': equations,
                'error': 'no solution found',
            }

    except Exception as e:
        return {
            'type': 'error',
            'expressions': equations,
            'error': str(e),
        }


if __name__ == '__main__':
    import sys
    import os
    sys.path.insert(0, os.path.dirname(__file__))
    from preprocess import preprocess
    from segment import segment, split_lines
    from model import load_model, predict_batch

    if len(sys.argv) < 2:
        print('usage: python solver.py <image_path>')
        sys.exit(1)

    load_model()
    binary = preprocess(sys.argv[1])
    lines = split_lines(binary)

    if len(lines) == 1:
        # single equation
        chars, boxes = segment(lines[0])
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

    else:
        # multiple lines = system of equations
        print(f'detected {len(lines)} equations')
        eq_strings = []

        for i, line_img in enumerate(lines):
            chars, boxes = segment(line_img)
            predictions = predict_batch(chars)
            preds, bxs = detect_equals(predictions, boxes)
            resolved = resolve_ambiguity(preds)
            eq_str = build_equation(resolved)
            eq_strings.append(eq_str)
            print(f'  line {i+1}: {[p[0] for p in predictions]} -> {eq_str}')

        result = solve_system(eq_strings)
        print()
        if result['type'] == 'system':
            print('system of equations:')
            for eq in eq_strings:
                print(f'  {eq}')
            print('solution:')
            for var, val in result['solutions'].items():
                print(f'  {var} = {val}')
        elif result['type'] == 'error':
            print(f'error: {result["error"]}')
