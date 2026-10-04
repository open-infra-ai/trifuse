# Agent Note: 范围冻结为三条可独立验证的算子路径

Status: implemented

## Problem

Triton 算子仓可以无限膨胀。本仓存在的意义不是算子数量，而是展示
「每条路径都能被独立参考实现与差分测试验证」的方法；范围不收窄，
验证纪律就会被稀释。

## Decision

仓内只保留三条面向 Transformer 推理的融合算子路径，经 `torch.library`
注册为 `torch.ops.trifuse.*`：

- `fused_rmsnorm_rope`：融合 RMSNorm 与 RoPE
- `fused_gated_mlp`：标准 SwiGLU/GeGLU，`activation(gate_proj(x)) * up_proj(x)`
- `flash_attention`：在线 softmax 前向，支持 causal mask——同时充当
  [cuflash](https://github.com/open-infra-ai/cuflash) 的独立参考实现

每条都保留 NumPy/PyTorch 参考实现、输入契约、差分测试、benchmark 与
autotuner 基建。删除双语文档站、OpenSpec 等外围脚手架，收敛为
核心代码 + 参考 + 测试 + 中文 README。

## Alternatives considered

- **保留更多算子做大而全的 kernel 库** — 覆盖面最强；但没有独立参考的
  算子没法差分验证，违背本仓的存在理由。
- **完全并入 cuflash** — 少一个仓；但 Triton 参考实现的价值恰在与
  CUDA C++ 版互为独立实现，合仓会让「独立对照」变味。

## Consequences

- **收益**：57 passed / 66 skipped（CPU-only）、123/123（GPU）的验证面
  小而全；面试叙事里「Triton vs CUDA 取舍」有对照实物。
- **代价**：不做新算子扩展；需求超出三路径时另立项目而非加进本仓。

## Verification

性能计量和错误输出门禁由
[两投影与正确性计时笔记](../testing/2026-10-04-benchmark-correctness-and-workload.md)
补充；此处的历史测试数量不作为当前数量或性能结论。

`pyproject.toml` 与 `trifuse/` 源码仅含三个算子入口；
`torch.ops.trifuse::*` 命名空间在测试中可枚举。
