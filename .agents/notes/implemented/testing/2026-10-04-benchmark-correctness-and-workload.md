# Agent Note: 性能结果以正确输出和真实计算图为前提

Status: implemented

## Problem

Gated MLP 的计时对象只做 gate/up 两次投影，单 GEMM 指标和三 GEMM 文档均与实现
不一致。正确性验证失败仍返回带指标的结果，会让错误输出参与速度比较。

## Decision

指标模型按两投影计算，共享输入只计一次逻辑流量，不包含 down projection；两个
benchmark 入口在正确性失败时拒绝计时与入报告。README 只解释可验证测量口径，
不保留缺少原始样本的速度声明。算子功能保持不变；本笔记只补充
[既有算子范围笔记](../architecture/2026-08-06-three-verifiable-ops-scope.md)中的验证纪律。

## Alternatives considered

将单 GEMM TFLOPS 乘二改动最小，但同时乘 bytes 会重复计算共享输入，而且无法表达
两投影语义，采用独立 profile。

保留失败结果并标记 correctness=false 有利于收集诊断，但现有报表照样显示速度和
speedup。拒绝计时使错误输出不能产生可引用性能数据；诊断保留在验证器输出中。

## Verification

`CUDA_VISIBLE_DEVICES='' .venv/bin/python -m pytest -q`：65 passed、66 skipped；
新增 8 个 CPU case 覆盖两投影 FLOPs/bytes、错误输出拒绝计时、正确输出计时与 suite
profile 选择。ruff check/format、`python -m mypy` 与 sdist/wheel 构建通过。

## Consequences

错误输出无法生成速度结论，计量对象可追溯到真实计算图；调用方依赖
correctness=false 返回值时需处理 ValueError。CPU 测试不证明 CUDA kernel 正确性；
本次 CUDA 数值矩阵与性能重采集未运行，逐次 raw/provenance 仍需独立补齐。
