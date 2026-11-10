from protocol_risk_analyzer.analysis import _estimate_eta_seconds


def test_estimate_eta_seconds_returns_zero_when_no_progress() -> None:
    assert _estimate_eta_seconds(elapsed_seconds=10, processed_chunks=0, total_chunks=5) == 0.0


def test_estimate_eta_seconds_scales_with_remaining_chunks() -> None:
    eta = _estimate_eta_seconds(elapsed_seconds=20, processed_chunks=4, total_chunks=10)
    assert eta == 30.0
