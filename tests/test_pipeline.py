"""
End-to-end integration and smoke test runner via pytest.
"""

import pytest
from scripts.run_smoke_test import run_smoke_test


def test_smoke_pipeline_execution():
    success = run_smoke_test()
    assert success is True
