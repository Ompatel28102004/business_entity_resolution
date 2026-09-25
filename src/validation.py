"""
Local submission validation.

The official validator shipped with the challenge resources
(``student_resource/utils/validate_submission.py``, stdlib-only) is used
directly via subprocess -- this module does not reimplement its rules, it
just wires it into the pipeline with the right paths so it's a one-liner to
run after ``inference.py`` produces the two output files. See
``output.py`` for the additional fail-fast checks applied at write time.
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

from . import config

OFFICIAL_VALIDATOR = config.REPO_DIR / "student_resource" / "utils" / "validate_submission.py"


def run_official_validator(
    matching_path: Path = config.OUTPUT_DIR / "matching_results.tsv",
    candidate_path: Path = config.OUTPUT_DIR / "candidate_pairs.tsv",
    test_dir: Path = config.RAW_TEST_DIR,
    check_ids: bool = False,
) -> int:
    """Run the official ``validate_submission.py`` and stream its output.

    Returns the process's exit code (0 == PASS, matching the script's own
    convention). Raises ``FileNotFoundError`` if the official validator is
    not present in this checkout (it should be, under
    ``student_resource/utils/``).
    """
    if not OFFICIAL_VALIDATOR.exists():
        raise FileNotFoundError(
            f"Official validator not found at {OFFICIAL_VALIDATOR}. "
            "It ships with the challenge resources under student_resource/utils/."
        )
    cmd = [
        sys.executable,
        str(OFFICIAL_VALIDATOR),
        "--matching", str(matching_path),
        "--candidate", str(candidate_path),
        "--test-dir", str(test_dir),
    ]
    if check_ids:
        cmd.append("--check-ids")
    proc = subprocess.run(cmd, cwd=str(OFFICIAL_VALIDATOR.parent.parent))
    return proc.returncode


if __name__ == "__main__":
    rc = run_official_validator()
    sys.exit(rc)
