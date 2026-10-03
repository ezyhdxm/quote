"""Enforce small, example-bearing core blocks without running notebooks."""
# TEST SETUP LOGIC: Standard-library source fixtures; no training or data access.
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

checker_path = Path(__file__).resolve().parents[1] / 'tools/check_logic_comments.py'
spec = importlib.util.spec_from_file_location('logic_comment_checker', checker_path)
checker = importlib.util.module_from_spec(spec)
spec.loader.exec_module(checker)

# TEST LOGIC: Check distinct boundary failures and actual repository entry points.
class LogicCommentChecks(unittest.TestCase):
    def test_ten_lines_pass_eleven_fail(self):
        header = '# CORE LOGIC: STEP 1\n# Input: x=0\n# Output: x=9\n'
        ten = header + 'x=0\n' + 'x+=1\n' * 9
        self.assertEqual(checker.check_source(ten)[0], [])
        self.assertTrue(any('11 code lines' in error for error in checker.check_source(ten + 'x+=1\n')[0]))

    def test_multiline_code_counts_all_physical_lines(self):
        source = '# CORE LOGIC: STEP 1\n# Input: x=1\n# Output: y=2\ny=(\n  x\n  + 1\n)\n'
        errors, blocks = checker.check_source(source)
        self.assertFalse(errors)
        self.assertEqual(blocks[0]['code_lines'], 4)

    def test_missing_output_and_misplaced_example_fail(self):
        source = '# CORE LOGIC: STEP 1\n# Input: x=1\ny=x+1\n# Output: y=2\n'
        self.assertTrue(any('missing Output' in error for error in checker.check_source(source)[0]))
        self.assertTrue(checker.check_source('x=1\n')[0])

    def test_noncore_marker_closes_block_and_docstrings_do_not_count(self):
        source = '# CORE LOGIC: STEP 1\n# Input: None\n# Output: 1\ndef f():\n    """documentation\n    more lines\n    """\n    return 1\n# PLOTTING LOGIC: display only\n' + 'x=0\n' * 20
        errors, blocks = checker.check_source(source)
        self.assertEqual(errors, [])
        self.assertEqual(blocks[0]['code_lines'], 2)

    def test_notebook_cells_are_independent(self):
        notebook = {'cells': [{'cell_type': 'markdown', 'source': ['description']},
                              {'cell_type': 'code', 'source': ['# UI LOGIC: first\nx=1\n']},
                              {'cell_type': 'code', 'source': ['y=2\n']}]}
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'example.ipynb'
            path.write_text(json.dumps(notebook))
            errors, _ = checker.inspect_path(path)
        self.assertEqual(len(errors), 1)
        self.assertIn(':cell2:', errors[0])

    def test_all_current_quality_runtime_sources_meet_standard(self):
        root = Path(__file__).resolve().parents[1]
        paths = sorted(root.glob('quote_quality_*.py')) + sorted(root.glob('quote_quality_*.ipynb'))
        paths.append(checker_path)
        for path in paths:
            with self.subTest(file=path.name):
                errors, _ = checker.inspect_path(path)
                self.assertEqual(errors, [])

# TEST RUNNER LOGIC: Explicit invocation only.
if __name__ == '__main__':
    unittest.main()
