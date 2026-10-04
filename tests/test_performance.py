import pytest

from trifuse import performance
from trifuse.performance import MIN_LATENCY_MS


def test_latency_only_preserves_latency_and_zeroes_derived_fields():
    metrics = performance.latency_only().metrics(0.25)

    assert metrics.latency_ms == 0.25
    assert metrics.throughput_tflops == 0.0
    assert metrics.bandwidth_gbps == 0.0
    assert metrics.bandwidth_utilization == 0.0


def test_elementwise_profile_normalizes_zero_latency_and_computes_bandwidth():
    metrics = performance.elementwise(numel=256).metrics(0.0)

    profile = performance.elementwise(numel=256)
    expected_bandwidth = (256 * profile.bytes_per_element * 2) / (MIN_LATENCY_MS * 1e6)
    assert metrics.latency_ms == MIN_LATENCY_MS
    assert metrics.throughput_tflops == 0.0
    assert metrics.bandwidth_gbps == pytest.approx(expected_bandwidth)
    assert metrics.bandwidth_utilization == pytest.approx(
        (expected_bandwidth / profile.peak_bandwidth_gbps) * 100
    )


def test_gemm_profile_computes_throughput_and_bandwidth_and_utilization():
    profile = performance.gemm(M=8, N=16, K=32)
    metrics = profile.metrics(0.5)

    expected_tflops = (2 * 8 * 16 * 32) / (0.5 * 1e9)
    expected_bandwidth = ((8 * 32 + 32 * 16 + 8 * 16) * 2) / (0.5 * 1e6)
    expected_util = (expected_bandwidth / profile.peak_bandwidth_gbps) * 100

    assert metrics.latency_ms == 0.5
    assert metrics.throughput_tflops == pytest.approx(expected_tflops)
    assert metrics.bandwidth_gbps == pytest.approx(expected_bandwidth)
    assert metrics.bandwidth_utilization == pytest.approx(expected_util)


def test_invalid_profile_inputs_raise_value_error():
    with pytest.raises(ValueError):
        performance.elementwise(numel=0)

    with pytest.raises(ValueError):
        performance.gemm(M=8, N=0, K=16)


@pytest.mark.parametrize("bytes_per_element", [2, 4])
def test_gated_mlp_counts_two_projections_and_shared_input(bytes_per_element):
    profile = performance.gated_mlp(M=128, N=11264, K=4096, bytes_per_element=bytes_per_element)
    metrics = profile.metrics(1.0)

    assert metrics.throughput_tflops == pytest.approx(4 * 128 * 11264 * 4096 / 1e9)
    assert metrics.bandwidth_gbps == pytest.approx(
        (128 * 4096 + 2 * 4096 * 11264 + 128 * 11264) * bytes_per_element / 1e6
    )


def test_gated_mlp_rejects_invalid_dims():
    with pytest.raises(ValueError, match="N must be a positive int"):
        performance.gated_mlp(M=8, N=0, K=16)


@pytest.mark.parametrize("latency", [-1.0, float("nan"), float("inf"), float("-inf")])
def test_invalid_latency_raises_value_error(latency):
    with pytest.raises(ValueError):
        performance.elementwise(numel=256).metrics(latency)


@pytest.mark.parametrize("val", [True, False])
def test_latency_rejects_bool(val):
    # Should reject True and False as latency values
    with pytest.raises(ValueError):
        performance.elementwise(numel=256).metrics(val)


def test_elementwise_rejects_non_int_numel():
    with pytest.raises(ValueError):
        performance.elementwise(numel=256.0)


@pytest.mark.parametrize(
    "M,N,K",
    [
        (8.0, 16, 32),
        (8, 16.0, 32),
        (8, 16, 32.0),
    ],
)
def test_gemm_rejects_non_int_dims(M, N, K):
    with pytest.raises(ValueError):
        performance.gemm(M=M, N=N, K=K)


@pytest.mark.parametrize(
    "profile_fn,kwargs",
    [
        (performance.elementwise, {"numel": 256, "bytes_per_element": 2.0}),
        (performance.gemm, {"M": 8, "N": 16, "K": 32, "bytes_per_element": 2.0}),
    ],
)
def test_bytes_per_element_rejects_non_int(profile_fn, kwargs):
    with pytest.raises(ValueError):
        profile_fn(**kwargs)
