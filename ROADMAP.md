# trifuse 路线图

> 定位：Triton kernel 与验证方法的**精简练习仓**，状态 **stable**（作品完成，只修正确性 bug 与文档；`phase-2-e` tag 记录 2026-08 面试就绪快照）。
> 三条算子路径（fused_rmsnorm_rope / fused_gated_mlp / flash_attention）已具备
> 独立参考实现与差分测试；另有 Triton SGEMM + `torch.library` 注册。
> FlashAttention 前向是 [cuflash](https://github.com/open-infra-ai/cuflash)
> 的参考实现，不是本仓的优化旗舰。

## 面试前建议（低成本）

- [x] 在可用 GPU 上跑一次完整 benchmark，把真实数字（带硬件型号）写入 README
      —— RTX 3060 Laptop，`fused_gated_mlp` ≈ 3.45 ms、`fused_rmsnorm_rope` ≈ 0.10 ms
      （commit `ebf6c32+`，见 README「性能基准」）
- [x] Triton SGEMM + `torch.library` 注册三个自定义 op（`torch.ops.trifuse.*`）
- [ ] 准备「Triton 版 FlashAttention vs CUDA C++ 版（cuflash）」的对比讲述：
      两者的 block 设计、在线 softmax 实现与验证方法异同（Phase 3 讲述稿）

## 可选扩展（只在有余力时；stable 状态下不做）

- [ ] 新增融合算子候选：fused softmax + mask、INT8/FP8 反量化融合（必须带独立参考实现与差分测试才进主分支）
- [ ] 与 tiny-llm 的 kernel 选型呼应：说明何时选 Triton、何时选 CUDA C++

## 明确不做

- 不引入无参考实现的 kernel
- 不写入未在真实硬件测量的性能数字（沿用仓库既有原则）
