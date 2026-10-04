from unittest.mock import Mock

import pytest
import torch

from trifuse.benchmark.suite import BenchmarkSuite
from trifuse.models import KernelMetrics


@pytest.mark.parametrize(
    "entrypoint", [BenchmarkSuite.benchmark_kernel, BenchmarkSuite.compare_with_pytorch]
)
def test_incorrect_output_is_rejected_before_timing(monkeypatch, entrypoint):
    measure = Mock()
    monkeypatch.setattr("trifuse.benchmark.suite.measure_metrics", measure)
    suite = BenchmarkSuite()
    x = torch.ones(4)

    with pytest.raises(ValueError, match="incorrect: correctness verification failed"):
        entrypoint(suite, lambda x: x * 2, lambda x: x, "incorrect", (4,), x)

    measure.assert_not_called()
    assert suite.report.results == []
    assert suite.report.comparisons == []


@pytest.mark.parametrize(
    "entrypoint", [BenchmarkSuite.benchmark_kernel, BenchmarkSuite.compare_with_pytorch]
)
def test_matching_output_can_be_timed(monkeypatch, entrypoint):
    measure = Mock(return_value=KernelMetrics(1.0, 0.0, 0.0, 0.0))
    monkeypatch.setattr("trifuse.benchmark.suite.measure_metrics", measure)
    suite = BenchmarkSuite()
    x = torch.ones(4)

    result = entrypoint(suite, lambda x: x * 2, lambda x: x * 2, "correct", (4,), x)

    assert result.correctness
    assert measure.call_count == (1 if entrypoint is BenchmarkSuite.benchmark_kernel else 2)


def test_gated_mlp_suite_selects_two_projection_profile(monkeypatch):
    randn = torch.randn
    monkeypatch.setattr(
        torch, "randn", lambda *args, **kwargs: randn(*args, **{**kwargs, "device": "cpu"})
    )
    suite = BenchmarkSuite()
    benchmark = Mock()
    monkeypatch.setattr(suite, "benchmark_kernel", benchmark)

    suite.benchmark_gated_mlp([2], [4], [8], [16])

    profile = benchmark.call_args.kwargs["performance"]
    assert profile.kind == "gated_mlp"
    assert profile.dims == (8, 16, 8)
    assert profile.bytes_per_element == 2
