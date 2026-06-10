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
    handle x/X/times confusion and Y/y variable normalization.
    if x, X, or times appears between two operands -> treat as multiplication.
    if Y appears between two operands -> treat as multiplication.
    otherwise -> treat as variable x or y respectively.
    """
    labels = [p[0] for p in predictions]
    confs = [p[1] for p in predictions]
    resolved = []

    for i, label in enumerate(labels):
        if label in ('X', 'times', 'x'):
            prev = labels[i - 1] if i > 0 else None
            nxt = labels[i + 1] if i < len(labels) - 1 else None

            prev_is_operand = prev is not None and (prev.isdigit() or prev in ('x', 'y', 'z', ')'))
            next_is_operand = nxt is not None and (nxt.isdigit() or nxt in ('x', 'y', 'z', '('))

            if prev_is_operand and next_is_operand:
                # between two operands = multiplication
                resolved.append(('*', confs[i]))
            else:
                # treat as variable x
                resolved.append(('x', confs[i]))
        elif label == 'Y':
            # uppercase Y from CNN — treat as variable y or multiplication
            prev = labels[i - 1] if i > 0 else None
            nxt = labels[i + 1] if i < len(labels) - 1 else None

            prev_is_operand = prev is not None and (prev.isdigit() or prev in ('x', 'y', 'z', ')'))
            next_is_operand = nxt is not None and (nxt.isdigit() or nxt in ('x', 'y', 'z', '('))

            if prev_is_operand and next_is_operand:
                resolved.append(('*', confs[i]))
            else:
                # normalise to lowercase y so SymPy & build_equation can handle it
                resolved.append(('y', confs[i]))
        elif label == 'Z':
            # uppercase Z — normalise to lowercase z
            resolved.append(('z', confs[i]))
        elif label == 'div':
            resolved.append(('/', confs[i]))
        elif label == 'o':
            # Letter 'o' is almost always a digit '0' in math equations
            resolved.append(('0', confs[i]))
        elif label in ('ascii_124', 'l'):
            # Pipe symbol '|' or lowercase 'l' is almost always digit '1'
            resolved.append(('1', confs[i]))
        else:
            resolved.append((label, confs[i]))

    return resolved


def _dry_run_parse(eq_str):
    """Dry-run parse the equation string using SymPy to check syntax validity."""
    import sympy as sp
    eq_str_fixed = _fix_leading_zeros(eq_str)
    
    # Check simple syntax errors before SymPy
    if not eq_str_fixed or not eq_str_fixed.strip():
        return False, None
    if re.search(r'[=+\-*/]{3,}', eq_str_fixed):
        return False, None
    if re.match(r'^[=+*/]', eq_str_fixed):
        return False, None
    if re.search(r'[=+\-*/]$', eq_str_fixed):
        return False, None

    try:
        if '=' in eq_str_fixed:
            left, right = eq_str_fixed.split('=', 1)
            if not left.strip() or not right.strip():
                return False, None
            lhs = sp.sympify(left)
            rhs = sp.sympify(right)
            expr = lhs - rhs
        else:
            expr = sp.sympify(eq_str_fixed)
        return True, expr
    except Exception:
        return False, None


def resolve_ambiguity_top_k(preds_top_k, boxes, confidence_threshold=0.85):
    """
    Finds the mathematically most plausible equation using top-k predictions
    and equation context.
    """
    import itertools
    import sympy as sp

    # 1. Determine which positions are ambiguous and should be branched
    branch_positions = []
    # Known highly ambiguous characters
    ambiguous_set = {'o', '0', 'l', '1', 'ascii_124', 's', '5', 'z', '2', 'g', 'q', '9', 't', '+', 'x', 'X', 'times', 'b', '6', 'B', '8'}

    for i, options in enumerate(preds_top_k):
        if not options:
            continue
        top_label, top_conf = options[0]
        # Branch if confidence is low, or if the top class is known to be ambiguous
        # and there is a reasonable second choice (confidence > 0.1)
        should_branch = (top_conf < confidence_threshold) or (top_label in ambiguous_set and len(options) > 1 and options[1][1] > 0.1)
        if should_branch:
            branch_positions.append(i)

    # To avoid combinatorial explosion, limit the branching to the 4 most uncertain positions
    if len(branch_positions) > 4:
        # Sort by top-1 confidence (lowest first)
        branch_positions = sorted(branch_positions, key=lambda idx: preds_top_k[idx][0][1])[:4]

    # 2. Build the candidate pool for each position
    candidate_pools = []
    for i, options in enumerate(preds_top_k):
        if i in branch_positions:
            # Consider all top-k options for this position
            candidate_pools.append(options)
        else:
            # Only consider the top option
            candidate_pools.append([options[0]])

    # 3. Generate all combinations of predictions
    best_candidate_resolved = None
    best_eq_str = None
    best_score = -999.0
    best_avg_conf = 0.0

    # Let's generate candidates and evaluate them
    for candidate in itertools.product(*candidate_pools):
        # A candidate is a list/tuple of (label, conf) for each character
        # A. Detect equals and resolve standard ambiguity
        preds_eq, boxes_eq = detect_equals(list(candidate), boxes)
        resolved = resolve_ambiguity(preds_eq)
        eq_str = build_equation(resolved)

        # B. Check if it's syntactically valid in SymPy
        is_valid, parsed_expr = _dry_run_parse(eq_str)
        
        # C. Compute score
        avg_conf = sum(p[1] for p in candidate) / len(candidate)
        
        # Base score is the average confidence
        score = avg_conf

        if not is_valid:
            # If not valid, we heavily penalize it but don't completely discard it
            # (in case NO candidate is syntactically valid, we still want to pick the best-guess prediction)
            score -= 10.0
        else:
            # Context-aware bonuses/penalties for valid equations
            free_symbols = {str(sym) for sym in parsed_expr.free_symbols} if parsed_expr else set()
            
            # 1. Variable 'o' or 'l' penalty (extremely likely to be '0' or '1')
            if 'o' in free_symbols:
                score -= 0.5
            if 'l' in free_symbols:
                score -= 0.5

            # 2. Consistent variables bonus
            # If the equation uses common variables like x or y, it's preferred
            for common_var in ['x', 'y']:
                if common_var in free_symbols:
                    score += 0.1

            # 3. Number vs variable implicit multiplication check
            # e.g., if we chose 'o' as a variable, we have terms like '2*o'
            # If we chose '0', it becomes a single number '20'.
            if '*o' in eq_str or 'o*' in eq_str:
                score -= 0.3
            if '*l' in eq_str or 'l*' in eq_str:
                score -= 0.3

            # 4. If it contains '0' (digit zero), give a small bonus because digits are much more common than letter 'o' in equations
            # especially if it forms a multi-digit number
            digit_sequences = re.findall(r'\d+', eq_str)
            for seq in digit_sequences:
                if len(seq) > 1:
                    score += 0.05 * len(seq)

        # Track the best candidate
        if score > best_score:
            best_score = score
            best_candidate_resolved = resolved
            best_eq_str = eq_str
            best_avg_conf = avg_conf

    # Fallback to top-1 if somehow no candidate was selected (should not happen)
    if best_candidate_resolved is None:
        default_candidate = [options[0] for options in preds_top_k]
        preds_eq, boxes_eq = detect_equals(default_candidate, boxes)
        best_candidate_resolved = resolve_ambiguity(preds_eq)
        best_eq_str = build_equation(best_candidate_resolved)
        best_avg_conf = sum(p[1] for p in default_candidate) / len(default_candidate)

    return best_candidate_resolved, best_eq_str, best_avg_conf


def build_equation(predictions):
    """
    take resolved predictions and build equation string.
    adds implicit multiplication where needed (e.g. 2x -> 2*x, 3y -> 3*y).
    """
    labels = [p[0] for p in predictions]
    parts = []
    _vars = ('x', 'y', 'z')  # supported variables

    for i, sym in enumerate(labels):
        parts.append(sym)

        # implicit multiplication
        if i < len(labels) - 1:
            nxt = labels[i + 1]
            # digit followed by variable (e.g. 3x, 3y, 3z)
            if sym.isdigit() and nxt in _vars:
                parts.append('*')
            # variable followed by digit (like x2 -> x*2... rare but handle it)
            elif sym in _vars and nxt.isdigit():
                parts.append('*')
            # closing paren followed by digit or variable
            elif sym == ')' and (nxt.isdigit() or nxt in _vars or nxt == '('):
                parts.append('*')
            # digit or variable followed by opening paren
            elif (sym.isdigit() or sym in _vars) and nxt == '(':
                parts.append('*')

    return ''.join(parts)


def _fix_leading_zeros(s):
    """Remove leading zeros from integer literals (e.g. '02' -> '2') so SymPy can parse them."""
    return re.sub(r'\b0+(\d)', r'\1', s)


def solve_equation(eq_str):
    """
    solve using sympy.
    if there's '=' -> solve the equation.
    if no '=' -> just evaluate the arithmetic.
    """
    eq_str = _fix_leading_zeros(eq_str)
    x, y, z = sp.symbols('x y z')

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
    x, y, z = sp.symbols('x y z')

    try:
        exprs = []
        for eq_str in equations:
            eq_str = _fix_leading_zeros(eq_str)
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
