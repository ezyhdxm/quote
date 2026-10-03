"""Check documented logical blocks without importing or running research code."""
# SETUP LOGIC: Standard-library-only source inspection; no model/data imports.
import argparse
import ast
import io
import json
from pathlib import Path
import re
import tokenize

# CONFIGURATION LOGIC: Non-core categories close a preceding core block.
MARKER = re.compile(r'^\s*#\s*(CORE LOGIC: STEP ([1-9][0-9]*)\b|[A-Z][A-Z /_-]* LOGIC\b)')
IGNORED = {tokenize.COMMENT, tokenize.NL, tokenize.NEWLINE, tokenize.INDENT,
           tokenize.DEDENT, tokenize.ENDMARKER, tokenize.ENCODING}

# SETUP LOGIC: Function declarations and docstrings are not business computation.
def code_lines(source):
    """Physical lines containing Python code; ignore standalone documentation."""
    # CORE LOGIC: STEP 1 — Exclude literal docstring text from executable lines.
    # Input: source='def f():\n    """explain"""\n    return 2\n'.
    # Output: documentation={2}; the function declaration and return remain code.
    # Trick: AST identifies real string expressions; a '#' inside a string is not a comment.
    tree = ast.parse(source)
    documentation = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Expr) and isinstance(node.value, ast.Constant) and isinstance(node.value.value, str):
            documentation.update(range(node.lineno, node.end_lineno + 1))
    # CORE LOGIC: STEP 2 — Count physical code lines, including multi-line expressions.
    # Input: source='x = (\n  1 +\n  2\n)\n# note\n'.
    # Output: {1, 2, 3, 4}; the comment line 5 is absent.
    # Trick: a multi-line token contributes every occupied line, not just its start.
    lines = set()
    for token in tokenize.generate_tokens(io.StringIO(source).readline):
        if token.type not in IGNORED:
            lines.update(range(token.start[0], token.end[0] + 1))
    return lines - documentation

# SETUP LOGIC: Inspect one source string; errors are returned for the CLI/test caller.
def check_source(source, label='<source>'):
    # CORE LOGIC: STEP 1 — Locate annotation boundaries and executable code.
    # Input: lines=['# CORE LOGIC: STEP 1', '# Input: x=2', '# Output: y=3', 'y=x+1'].
    # Output: one marker at physical line 1; code={4}; no syntax error.
    lines = source.splitlines()
    markers = [(i + 1, MARKER.match(line)) for i, line in enumerate(lines) if MARKER.match(line)]
    errors, blocks = [], []
    try:
        executable = code_lines(source)
    except (SyntaxError, tokenize.TokenError) as error:
        return [f'{label}: syntax/token error: {error}'], []
    # CORE LOGIC: STEP 2 — Require a classification before the first code line.
    # Input: source='x=1\n' gives executable={1}, markers=[].
    # Output: ['<source>: code is missing a leading LOGIC classification'].
    if executable and (not markers or min(executable) < markers[0][0]):
        errors.append(f'{label}: code is missing a leading LOGIC classification')
    # CORE LOGIC: STEP 3 — Bound each marked core span by the next logic marker.
    # Input: CORE marker=10, next PLOTTING marker=15, code lines={13,14,16}.
    # Output: core code=[13,14]; plotting line 16 is excluded from its length.
    # Trick: count physical lines, not semicolon-separated statements or AST nodes.
    for index, (start, match) in enumerate(markers):
        if match.group(2) is None:
            continue
        end = markers[index + 1][0] if index + 1 < len(markers) else len(lines) + 1
        code = sorted(line for line in executable if start < line < end)
        first = code[0] if code else end
        example = '\n'.join(lines[start:first - 1])
        blocks.append(dict(file=label, line=start, step=int(match.group(2)), code_lines=len(code)))
        # CORE LOGIC: STEP 4 — Report oversized blocks and missing examples.
        # Input: one block has 11 code lines and only '# Input: x=2'.
        # Output: two errors: 'CORE block has 11 code lines (>10)' and 'missing Output example'.
        # Trick: presence/size are automated; arithmetic truth and useful examples need human review.
        if len(code) > 10:
            errors.append(f'{label}:{start}: CORE block has {len(code)} code lines (>10)')
        for field in ['Input', 'Output']:
            if not re.search(r'^\s*#\s*' + field + r':\s*\S', example, flags=re.M):
                errors.append(f'{label}:{start}: missing {field} example before core code')
    # VALIDATION LOGIC: Return diagnostics; never rewrite a source file to pass lint.
    return errors, blocks

# FILE IO LOGIC: Read each notebook cell as source without executing it.
def inspect_path(path):
    if path.suffix == '.ipynb':
        notebook = json.loads(path.read_text(encoding='utf-8'))
        sources = [(''.join(cell['source']), f'{path}:cell{index}') for index, cell in enumerate(notebook['cells']) if cell['cell_type'] == 'code']
    else:
        sources = [(path.read_text(encoding='utf-8'), str(path))]
    errors, blocks = [], []
    for source, label in sources:
        local_errors, local_blocks = check_source(source, label)
        errors.extend(local_errors); blocks.extend(local_blocks)
    return errors, blocks

# CLI LOGIC: Default scope is all current quality-research runtime modules/notebooks.
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('paths', nargs='*')
    parser.add_argument('--json', action='store_true')
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    paths = [Path(p) for p in args.paths] if args.paths else sorted(root.glob('quote_quality_*.py')) + sorted(root.glob('quote_quality_*.ipynb'))
    errors, blocks = [], []
    for path in paths:
        e, b = inspect_path(path); errors.extend(e); blocks.extend(b)
    if args.json:
        print(json.dumps(dict(files=len(paths), core_blocks=len(blocks), max_core_lines=max((b['code_lines'] for b in blocks), default=0), errors=errors), indent=2))
    elif errors:
        print('\n'.join(errors))
    else:
        print(f'{len(paths)} files; {len(blocks)} core blocks; every core block <=10 code lines with Input/Output comments.')
    return 1 if errors else 0

# CLI LOGIC: Importing the checker is safe; only an explicit invocation runs it.
if __name__ == '__main__':
    raise SystemExit(main())
