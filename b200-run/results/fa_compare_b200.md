| system | model | SMs | 4000 steps (s) | sd | tok/s | final loss | vs FA 70.1 s |
|---|---|---|---|---|---|---|---|
| megatron_te | S | 116 (MPS) | 62.4 | 0.23 | 262685 | 2.1536 | 0.89x |
| megatron_te | S | full 148 | 57.3 | 0.00 | 286128 | 2.1437 | 0.82x |
| pytorch_eager | S | 116 (MPS) | 52.5 | 0.02 | 312172 | 2.1516 | 0.75x |
| pytorch_eager | S | full 148 | 45.4 | 0.00 | 360776 | 2.1479 | 0.65x |

| vLLM decode (batch 1) | model | SMs | tok/s | vs FA 710 |
|---|---|---|---|---|
| vllm | S | 24 (MPS) | 684 | 0.96x |
| vllm | S | full | 682 | 0.96x |

FA published (B200): baseline 70.1 s / 4000 steps, ~230000 tok/s, vLLM 710 tok/s; launches 514562.
Launch counts (this machine): see launch_counts.demo_equivalent in baseline_v2_b200_20261102.json
