import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import paddle
import pytest


@pytest.fixture(autouse=True)
def _reset_paddle_graph_mode():
    """Keep tests independent of paddle's global graph mode.

    Several quantization test modules call ``paddle.enable_static()`` at import
    time, which otherwise leaks static-graph mode into every test collected
    afterwards and breaks the dynamic-graph tests. Reset to paddle's default
    dynamic mode around each test; tests that need static mode enable it in
    their own ``setUp``.
    """
    paddle.disable_static()
    yield
    paddle.disable_static()
