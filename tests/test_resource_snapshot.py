from unittest.mock import patch

import psutil
import pytest

from informity.diagnostics.resource_snapshot import capture_resource_snapshot


def test_capture_resource_snapshot_returns_capture_error_for_psutil_error() -> None:
    with patch('informity.diagnostics.resource_snapshot.psutil.Process', side_effect=psutil.AccessDenied()):
        payload = capture_resource_snapshot()
    assert isinstance(payload.get('capture_error'), str)


def test_capture_resource_snapshot_does_not_swallow_unexpected_runtime_error() -> None:
    with (
        patch('informity.diagnostics.resource_snapshot.psutil.Process', side_effect=RuntimeError('boom')),
        pytest.raises(RuntimeError, match='boom'),
    ):
        capture_resource_snapshot()
