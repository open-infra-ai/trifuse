# trifuse

> 📚 Portfolio map: https://github.com/open-infra-ai/open-infra-ai

> **面向 Transformer 推理的可验证 Triton 融合算子与 `torch.library` 集成。**

[![CI](https://github.com/open-infra-ai/trifuse/actions/workflows/ci.yml/badge.svg)](https://github.com/open-infra-ai/trifuse/actions/workflows/ci.yml)

> 状态：**stable**。当前源码包与 GitHub 最新 Release/Tag 均为 `2.0.1`；
> 新功能暂停，继续维护正确性、兼容性与可复现验证。

面向 AI Infra 学习的精简 Triton 算子仓库。只保留三条可以用独立参考实现验证的 Transformer 推理路径：

- `fused_rmsnorm_rope`：融合 RMSNorm 与 RoPE
- `fused_gated_mlp`：标准 SwiGLU/GeGLU，公式为 `activation(gate_proj(x)) * up_proj(x)`
- `flash_attention`：带在线 softmax 的 FlashAttention 前向，支持 causal mask

> ℹ️ **定位**：Triton FlashAttention 是 [cuflash](https://github.com/open-infra-ai/cuflash)
> 的独立参考实现，用于验证 CUDA C++ 版本的正确性。完整 FlashAttention 前后向 +
> 优化叙事见 cuflash（本仓库只保留前向参考实现）。

仓库同时保留 NumPy/PyTorch 参考实现、输入契约、差分测试、benchmark 与 autotuner 基础设施。旧版所谓“FP8 E4M3”实际是存储在 `uint8` 中的均匀线性量化，并不编码 E4M3 指数和尾数，因此已整条删除，避免把 INT8 路径误当作 FP8 教材。

## 安装

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e '.[dev]'
```

运行 Triton kernel 需要 NVIDIA GPU、CUDA 版 PyTorch 和 Triton。CPU 环境仍可运行参考实现、输入契约、格式和打包检查。

## 快速示例

```python
import torch
from trifuse import flash_attention, fused_gated_mlp, fused_rmsnorm_rope

x = torch.randn(2, 128, 4096, device="cuda", dtype=torch.float16)
gate_weight = torch.randn(11008, 4096, device="cuda", dtype=torch.float16)
up_weight = torch.randn_like(gate_weight)

# 标准 SwiGLU: silu(gate_proj(x)) * up_proj(x)
mlp_output = fused_gated_mlp(x, gate_weight, up_weight, activation="silu")

q = torch.randn(2, 32, 128, 128, device="cuda", dtype=torch.float16)
k = torch.randn_like(q)
v = torch.randn_like(q)
attention_output = flash_attention(q, k, v, causal=True)
```

RMSNorm + RoPE 的完整调用示例见 [`examples/rmsnorm_rope_example.py`](examples/rmsnorm_rope_example.py)，Gated MLP 示例见 [`examples/gated_mlp_example.py`](examples/gated_mlp_example.py)。

## torch.library 自定义算子

`import trifuse` 会把三个公开 kernel 注册进 `torch.ops.trifuse.*` 命名空间
（与 vLLM/SGLang 等推理框架用 `torch.library` 注册自定义 op 的方式一致，便于接入
`torch.compile` / `torch.export` 图）：

```python
import torch
import trifuse  # 触发注册

# trifuse::sgemm(Tensor a, Tensor b) -> Tensor
c = torch.ops.trifuse.sgemm(a, b)

# trifuse::fused_rmsnorm_rope(Tensor x, Tensor weight, Tensor cos, Tensor sin, float eps) -> Tensor
out = torch.ops.trifuse.fused_rmsnorm_rope(x, weight, cos, sin)

# trifuse::fused_gated_mlp(Tensor x, Tensor gate_weight, Tensor up_weight, str activation) -> Tensor
out = torch.ops.trifuse.fused_gated_mlp(x, gate_weight, up_weight, activation="silu")
```

注册策略（`trifuse/ops.py`，需 torch>=2.4）：

统一使用 `torch.library.custom_op + register_fake`：`custom_op` 提供 eager 执行，
`register_fake` 提供 shape 推断，使 op 对 `torch.compile` / `torch.export` 作为
opaque 自定义算子可编译、可导出。

> 说明：`torch.library.triton_op`（2.13+）会把实现体暴露给 Inductor，要求用
> `torch._library.triton.wrap_triton` 注册 kernel；本仓库直接启动 Triton kernel，
> 实测在 torch 2.13 下 triton_op 路径 torch.compile 报 “Cannot access data
> pointer of Tensor”，故不采用。

所有 op 只接受 CUDA 张量，CPU 输入直接抛 `NotImplementedError`；op 内部只调用
`trifuse.kernels.*` 的公开函数，不复制 kernel 逻辑。

### 与 vLLM / SGLang custom op 的对应关系

推理框架用同一套 `torch.library` 机制暴露自定义 kernel：

- vLLM 的 `_custom_ops.py` 用 `torch.library.custom_op` 注册 `vllm::*` 命名空间
  （如 `vllm::attention.forward`），并用 `torch.library.register_fake` 提供 meta 实现；
- SGLang 通过 `torch.library` 暴露 `sglang::*` 算子，同样以 `register_fake` 支持
  torch.compile / 图捕获；
- 本仓库用同一模式：`trifuse::sgemm` / `trifuse::fused_rmsnorm_rope` /
  `trifuse::fused_gated_mlp`。差别仅在实现层：vLLM/SGLang 的生产 kernel 走
  FlashInfer/CUDA Graph 等，这里用 Triton kernel 做最小可验证实现。

## 验证

```bash
ruff format --check .
ruff check .
mypy trifuse --ignore-missing-imports
pytest -q
python -m build
```

测试分为两层：

- CPU 可运行：NumPy/PyTorch 参考模型、标准 SwiGLU 契约、FlashAttention/SDPA 对照、输入失败路径、benchmark/autotuner 工具。
- 必须有 CUDA：Triton kernel 与参考实现的数值差分测试；无 GPU 时明确 skip，不报告为已通过。

2026-08-23 验证结果：CPU-only 模式（`CUDA_VISIBLE_DEVICES=''`）为 **57 passed /
66 skipped**；RTX 3060 Laptop（PyTorch 2.13.0、Triton 3.7.1）为
**123/123 passed**。CI 使用 Hypothesis `ci` profile；本地默认使用较快的 `dev`
profile。

## 性能测量与引用口径

本仓尚无能绑定完整环境、精确 commit 和逐次原始样本的正式性能结果包。
历史 README 延迟表不作为当前性能证据；数值正确性测试也不构成速度优势。

`fused_gated_mlp` 计算 `activation(x @ gate_w.T) * (x @ up_w.T)`，输出
`[batch, seq, intermediate]`，只有两次 GEMM，**不包含 down projection**。
令 `M=batch×seq`、`K=hidden`、`N=intermediate`，GEMM FLOPs 为 `4MNK`
（不计激活和逐元素乘法）；逻辑 tensor bytes 为
`(MK + 2KN + MN) × element_size`，共享输入只计一次。该 bytes 模型不是实测
DRAM 流量，导出的带宽和利用率不是硬件计数器；利用率默认峰值参数也不是本机峰值。

`BenchmarkSuite` 在数值验证失败时抛出 `ValueError`，不计时、不写结果或 speedup。
`measure_latency` 返回预热后、两端 CUDA 同步的循环总墙钟时间除以调用次数，即
每调用**均值**，包含 Python/launch 开销；它不是逐样本中位数或纯 GPU kernel 时间。

复现（仓库内）：

```bash
# 需要 CUDA GPU + torch/triton
python -m tests.benchmarks.bench_gated_mlp
python -m tests.benchmarks.bench_rmsnorm_rope
# 或用 BenchmarkSuite 定制配置：
#   from trifuse.benchmark import BenchmarkSuite
#   BenchmarkSuite(warmup_runs=3, benchmark_runs=20).benchmark_gated_mlp(...)
```

发布性能结论前须保存 GPU/driver、Torch/Triton、commit/dirty 状态、输入与 dtype、
参考实现、容差、warmup/repeats 和原始计时样本，并报告失败与回退 shape。
gelu 使用精确 erf 定义；fp32 输入路径禁用 TF32 截断。数值正确性由
`tests/test_gated_mlp.py` 的差分测试覆盖，指标模型和拒绝计时门禁有 CPU 回归测试。

## 项目边界

这个仓库练习 Triton kernel 与验证方法，不承担以下职责：

- CUDA C++ 的系统学习路径：[`cuda-foundations`](https://github.com/open-infra-ai/cuda-foundations)
- FlashAttention 前后向的 CUDA C++ 深挖：[`cuflash`](https://github.com/open-infra-ai/cuflash)
- 完整模型加载与 token 生成：[`tiny-llm`](https://github.com/open-infra-ai/tiny-llm)
- Paged KV 与 continuous batching 控制面：[`paged-serving`](https://github.com/open-infra-ai/paged-serving)

新 kernel 只有在具备独立参考实现、边界测试和真实 GPU 验证计划时才进入主分支。未在当前硬件上测量的性能数字不会写入 README。

## License

MIT，详见 [LICENSE](LICENSE)。
