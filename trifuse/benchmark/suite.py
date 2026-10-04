"""Benchmark suite for Triton operators."""

from abc import ABC, abstractmethod
from typing import Any, Callable, Dict, List, Literal, Tuple

import torch

from trifuse import performance as perf_module
from trifuse.benchmark.correctness import CorrectnessVerifier
from trifuse.benchmark.report import BenchmarkResult, ComparisonResult, PerformanceReport
from trifuse.performance import PerformanceProfile, measure_metrics


class KernelBenchmark(ABC):
    """Abstract base class for kernel-specific benchmarks.

    Each kernel family can implement this interface to provide its own
    benchmark configuration. This allows BenchmarkSuite to be extended
    without modification when new kernels are added.

    Example:
        >>> class RMSNormRoPEBenchmark(KernelBenchmark):
        ...     @property
        ...     def name(self) -> str:
        ...         return "fused_rmsnorm_rope"
        ...
        ...     def create_inputs(self, problem_size):
        ...         batch, seq_len, hidden_dim = problem_size
        ...         return {
        ...             "x": torch.randn(batch, seq_len, hidden_dim, ...),
        ...             ...
        ...         }
        ...
        ...     def kernel_fn(self, inputs):
        ...         return fused_rmsnorm_rope(**inputs)
        ...
        ...     def reference_fn(self, inputs):
        ...         return reference_fused_rmsnorm_rope(**inputs)
        ...
        ...     def performance_profile(self, problem_size):
        ...         batch, seq_len, hidden_dim = problem_size
        ...         return elementwise(numel=batch * seq_len * hidden_dim)
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Return the kernel name for reporting."""
        pass

    @abstractmethod
    def create_inputs(self, problem_size: Tuple[int, ...]) -> Dict[str, Any]:
        """Create input tensors for the given problem size.

        Args:
            problem_size: Problem dimensions

        Returns:
            Dictionary of input tensors
        """
        pass

    @abstractmethod
    def kernel_fn(self, inputs: Dict[str, Any]) -> torch.Tensor:
        """Execute the Triton kernel.

        Args:
            inputs: Dictionary of input tensors from create_inputs()

        Returns:
            Output tensor
        """
        pass

    @abstractmethod
    def reference_fn(self, inputs: Dict[str, Any]) -> torch.Tensor:
        """Execute the reference implementation.

        Args:
            inputs: Dictionary of input tensors from create_inputs()

        Returns:
            Output tensor
        """
        pass

    @abstractmethod
    def performance_profile(self, problem_size: Tuple[int, ...]) -> PerformanceProfile | None:
        """Create a performance profile for the given problem size.

        Args:
            problem_size: Problem dimensions

        Returns:
            PerformanceProfile or None for latency-only metrics
        """
        pass

    def get_problem_sizes(self) -> List[Tuple[int, ...]]:
        """Return a list of problem sizes to benchmark.

        Override this method to customize the problem sizes for this kernel.

        Returns:
            List of problem size tuples
        """
        return [
            (2, 128, 4096),
            (4, 256, 4096),
            (8, 512, 4096),
        ]


class BenchmarkSuite:
    """Comprehensive benchmark suite for Triton operators.

    Provides functionality to:
    - Benchmark Triton kernels
    - Compare against PyTorch native operations
    - Compare against cuBLAS/cuDNN baselines
    - Verify numerical correctness
    - Generate performance reports

    Args:
        warmup_runs: Number of warmup iterations
        benchmark_runs: Number of benchmark iterations
        rtol: Relative tolerance for correctness
        atol: Absolute tolerance for correctness
    """

    def __init__(
        self,
        warmup_runs: int = 10,
        benchmark_runs: int = 100,
        rtol: float = 1e-2,
        atol: float = 1e-2,
    ):
        self.warmup_runs = warmup_runs
        self.benchmark_runs = benchmark_runs
        self.verifier = CorrectnessVerifier(rtol=rtol, atol=atol)
        self.report = PerformanceReport()

        # Set device metadata
        self.report.set_metadata(
            "device",
            str(torch.cuda.get_device_name(0)) if torch.cuda.is_available() else "cpu",
        )
        if torch.cuda.is_available():
            self.report.set_metadata("cuda_version", torch.version.cuda)
        self.report.set_metadata("pytorch_version", torch.__version__)

    def benchmark_kernel(
        self,
        kernel_fn: Callable,
        reference_fn: Callable,
        kernel_name: str,
        problem_size: Tuple[int, ...],
        *args,
        config: Dict[str, Any] = None,
        performance: PerformanceProfile | None = None,
        **kwargs,
    ) -> BenchmarkResult:
        """Benchmark a kernel and verify correctness.

        Args:
            kernel_fn: Triton kernel function
            reference_fn: Reference implementation for correctness
            kernel_name: Name for reporting
            problem_size: Problem dimensions
            *args: Arguments for both functions
            config: Kernel configuration
            performance: Optional PerformanceProfile for derived metrics
            **kwargs: Keyword arguments for both functions

        Returns:
            BenchmarkResult with timing and correctness info
        """
        # Get outputs
        triton_output = kernel_fn(*args, **kwargs)
        reference_output = reference_fn(*args, **kwargs)

        # Verify correctness
        is_correct = self.verifier.verify_allclose(triton_output, reference_output)
        if not is_correct:
            raise ValueError(f"{kernel_name}: correctness verification failed; timing rejected")

        metrics = measure_metrics(
            lambda: kernel_fn(*args, **kwargs),
            warmup_runs=self.warmup_runs,
            benchmark_runs=self.benchmark_runs,
            profile=performance,
        )

        result = BenchmarkResult(
            kernel_name=kernel_name,
            problem_size=problem_size,
            config=config or {},
            metrics=metrics,
            correctness=is_correct,
        )

        self.report.add_result(result)
        return result

    def compare_with_pytorch(
        self,
        triton_fn: Callable,
        pytorch_fn: Callable,
        kernel_name: str,
        problem_size: Tuple[int, ...],
        *args,
        performance: PerformanceProfile | None = None,
        **kwargs,
    ) -> ComparisonResult:
        """Compare Triton kernel with PyTorch native operation.

        Args:
            triton_fn: Triton kernel function
            pytorch_fn: PyTorch native function
            kernel_name: Name for reporting
            problem_size: Problem dimensions
            *args: Arguments for both functions
            performance: Optional PerformanceProfile for derived metrics
            **kwargs: Keyword arguments

        Returns:
            ComparisonResult with speedup info
        """
        # Get outputs
        triton_output = triton_fn(*args, **kwargs)
        pytorch_output = pytorch_fn(*args, **kwargs)

        # Verify correctness
        is_correct = self.verifier.verify_allclose(triton_output, pytorch_output)
        if not is_correct:
            raise ValueError(f"{kernel_name}: correctness verification failed; comparison rejected")

        triton_metrics = measure_metrics(
            lambda: triton_fn(*args, **kwargs),
            warmup_runs=self.warmup_runs,
            benchmark_runs=self.benchmark_runs,
            profile=performance,
        )
        pytorch_metrics = measure_metrics(
            lambda: pytorch_fn(*args, **kwargs),
            warmup_runs=self.warmup_runs,
            benchmark_runs=self.benchmark_runs,
            profile=performance,
        )

        speedup = (
            pytorch_metrics.latency_ms / triton_metrics.latency_ms
            if triton_metrics.latency_ms > 0
            else float("inf")
            if pytorch_metrics.latency_ms > 0
            else 0
        )

        comparison = ComparisonResult(
            kernel_name=kernel_name,
            problem_size=problem_size,
            triton_metrics=triton_metrics,
            baseline_metrics=pytorch_metrics,
            speedup=speedup,
            correctness=is_correct,
        )

        self.report.add_comparison(comparison)
        return comparison

    def benchmark_rmsnorm_rope(
        self,
        batch_sizes: List[int],
        seq_lens: List[int],
        hidden_dims: List[int],
        head_dim: int = 64,
    ) -> List[BenchmarkResult]:
        """Benchmark RMSNorm + RoPE across different sizes.

        Args:
            batch_sizes: List of batch sizes to test
            seq_lens: List of sequence lengths to test
            hidden_dims: List of hidden dimensions to test
            head_dim: Head dimension for RoPE

        Returns:
            List of benchmark results
        """
        from trifuse.kernels.rmsnorm_rope import fused_rmsnorm_rope
        from trifuse.reference import fused_rmsnorm_rope as fused_rmsnorm_rope_reference

        results = []

        for batch in batch_sizes:
            for seq_len in seq_lens:
                for hidden_dim in hidden_dims:
                    # Create inputs
                    x = torch.randn(batch, seq_len, hidden_dim, device="cuda", dtype=torch.float16)
                    weight = torch.randn(hidden_dim, device="cuda", dtype=torch.float16)
                    cos = torch.randn(seq_len, head_dim, device="cuda", dtype=torch.float16)
                    sin = torch.randn(seq_len, head_dim, device="cuda", dtype=torch.float16)

                    problem_size = (batch, seq_len, hidden_dim)
                    numel = batch * seq_len * hidden_dim
                    result = self.benchmark_kernel(
                        fused_rmsnorm_rope,
                        lambda *a, **kw: fused_rmsnorm_rope_reference(*a, **kw, backend="cuda"),
                        "fused_rmsnorm_rope",
                        problem_size,
                        x,
                        weight,
                        cos,
                        sin,
                        performance=perf_module.elementwise(numel=numel),
                    )
                    results.append(result)

        return results

    def benchmark_gated_mlp(
        self,
        batch_sizes: List[int],
        seq_lens: List[int],
        hidden_dims: List[int],
        intermediate_dims: List[int],
        activations: List[Literal["silu", "gelu"]] | None = None,
    ) -> List[BenchmarkResult]:
        """Benchmark Gated MLP across different sizes.

        Args:
            batch_sizes: List of batch sizes
            seq_lens: List of sequence lengths
            hidden_dims: List of hidden dimensions
            intermediate_dims: List of intermediate dimensions
            activations: List of activation functions

        Returns:
            List of benchmark results
        """
        if activations is None:
            activations = ["silu"]
        from trifuse.kernels.gated_mlp import fused_gated_mlp
        from trifuse.reference import gated_mlp as gated_mlp_reference

        results = []

        for batch in batch_sizes:
            for seq_len in seq_lens:
                for hidden_dim in hidden_dims:
                    for inter_dim in intermediate_dims:
                        for activation in activations:
                            # Create inputs
                            # 权重缩放到 0.1（与 tests/test_gated_mlp.py 一致）：无界
                            # randn 会让 gate*up 中间值达到 ±5e4，fp16 输出量化误差
                            # 可达数十，导致 correctness 误判失败。
                            x = torch.randn(
                                batch, seq_len, hidden_dim, device="cuda", dtype=torch.float16
                            )
                            gate_w = (
                                torch.randn(
                                    inter_dim, hidden_dim, device="cuda", dtype=torch.float16
                                )
                                * 0.1
                            )
                            up_w = (
                                torch.randn(
                                    inter_dim, hidden_dim, device="cuda", dtype=torch.float16
                                )
                                * 0.1
                            )

                            problem_size = (batch, seq_len, hidden_dim, inter_dim)

                            # Create wrapper functions to avoid activation parameter conflict
                            def triton_fn(x, gate_w, up_w):
                                return fused_gated_mlp(x, gate_w, up_w, activation=activation)

                            def ref_fn(x, gate_w, up_w):
                                # 必须用 cuda 后端：cpu 后端会在 NumPy 里跑 fp32 大矩阵乘
                                # （几千 × 上万 × 上万），单次 >30s，整个 benchmark 不可用。
                                return gated_mlp_reference(
                                    x, gate_w, up_w, activation=activation, backend="cuda"
                                )

                            result = self.benchmark_kernel(
                                triton_fn,
                                ref_fn,
                                f"fused_gated_mlp_{activation}",
                                problem_size,
                                x,
                                gate_w,
                                up_w,
                                performance=perf_module.gated_mlp(
                                    M=batch * seq_len,
                                    N=inter_dim,
                                    K=hidden_dim,
                                    bytes_per_element=x.element_size(),
                                ),
                            )
                            results.append(result)

        return results

    def benchmark_kernel_family(
        self,
        kernel_benchmark: KernelBenchmark,
        problem_sizes: List[Tuple[int, ...]] | None = None,
    ) -> List[BenchmarkResult]:
        """Benchmark a kernel family using the KernelBenchmark interface.

        This method provides an extensible way to benchmark new kernels
        without modifying BenchmarkSuite.

        Args:
            kernel_benchmark: KernelBenchmark implementation for the kernel family
            problem_sizes: Optional list of problem sizes. If None, uses kernel's defaults.

        Returns:
            List of benchmark results
        """
        sizes = problem_sizes or kernel_benchmark.get_problem_sizes()
        results = []

        for problem_size in sizes:
            # Create inputs
            inputs = kernel_benchmark.create_inputs(problem_size)

            # Get performance profile
            profile = kernel_benchmark.performance_profile(problem_size)

            # Run benchmark
            result = self.benchmark_kernel(
                lambda: kernel_benchmark.kernel_fn(inputs),
                lambda: kernel_benchmark.reference_fn(inputs),
                kernel_benchmark.name,
                problem_size,
                performance=profile,
            )
            results.append(result)

        return results

    def generate_report(self, format: str = "text") -> str:
        """Generate benchmark report.

        Args:
            format: "text" or "json"

        Returns:
            Formatted report string
        """
        if format == "json":
            return self.report.generate_json_report()
        return self.report.generate_text_report()

    def save_report(self, filepath: str, format: str = "text") -> None:
        """Save benchmark report to file.

        Args:
            filepath: Output file path
            format: "text" or "json"
        """
        self.report.save(filepath, format)
