# ALICE ML SaaS

GPU-free LLM inference service powered by ALICE-ML's 1.58bit Ternary engine.
Run 70B models locally on Apple Silicon or Raspberry Pi 5 — no GPU required.

## License

AGPL-3.0

## Why GPU-Free?

ALICE-ML の 1.58bit Ternary 量子化により、従来 GPU (VRAM 80GB+) が必要だった
70B パラメータの LLM を、CPU のメモリ帯域だけで実用速度で推論できる。

| 量子化 | 70B モデルサイズ | 必要VRAM/RAM |
|--------|-----------------|-------------|
| FP16 | ~140 GB | A100 x2 |
| INT4 (GPTQ) | ~35 GB | A100 x1 |
| **1.58bit Ternary** | **~9.3 GB** | **Mac Mini 32GB** |

### Apple Silicon 性能見込み

| デバイス | メモリ帯域 | 70B 1.58bit | 体感 |
|---------|-----------|------------|------|
| Mac Mini M4 (32GB) | 120 GB/s | 12-15 tok/s | 普通に会話できる |
| Mac Mini M4 Pro (48-64GB) | 273 GB/s | 30-50 tok/s | サクサク |
| Mac Studio M4 Ultra (192GB) | 800 GB/s | 80-100 tok/s | ほぼ即答 |
| Raspberry Pi 5 (8GB) | 17 GB/s | 5-8 tok/s (24B) | ポツポツ会話 |

## Architecture

```
Frontend :3000  -->  API Gateway :8080  -->  ML Engine :8081
                                                │
                                    ┌───────────┴───────────┐
                                    │   ALICE Inference      │
                                    │   Stack                │
                                    ├───────────────────────┤
                                    │ ALICE-ML    1.58bit    │
                                    │   ternary_matvec       │
                                    │   speculative_decode   │
                                    │   moe_router           │
                                    │   weight_streaming     │
                                    │   arena (zero-alloc)   │
                                    ├───────────────────────┤
                                    │ ALICE-SIMD  fast_exp   │
                                    │   rmsnorm, softmax     │
                                    ├───────────────────────┤
                                    │ ALICE-Cache KV-cache   │
                                    │   batch_get/put        │
                                    │   tiered (Hot/Warm)    │
                                    │   oracle (prefetch)    │
                                    └───────────────────────┘
```

| Layer | Port | Technology |
|-------|------|-----------|
| Frontend | 3000 | Next.js 14, Tailwind CSS |
| API Gateway | 8080 | Rust, Axum |
| ML Engine | 8081 | Rust, Axum, ALICE-ML/SIMD/Cache |

## Endpoints

| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/api/v1/infer` | テキスト生成（1.58bit Ternary推論） |
| `POST` | `/api/v1/quantize` | FP32重みを1.58bit Ternaryに量子化 |
| `GET` | `/api/v1/capabilities` | 推論エンジンの機能一覧 |
| `GET` | `/health` | ヘルスチェック |

### POST /api/v1/infer

```json
{
  "input": "What is the meaning of life?",
  "max_tokens": 256,
  "temperature": 0.7,
  "top_k": 40
}
```

Response:
```json
{
  "output": "The meaning of life is...",
  "tokens_generated": 128,
  "latency_ms": 8500.0,
  "tokens_per_second": 15.06,
  "model": {
    "name": "bitnet-70b-1.58bit",
    "quantization": "ternary_1.58bit",
    "parameters": "70B",
    "memory_mb": 9523.2
  }
}
```

### POST /api/v1/quantize

```json
{
  "weights_base64": "AAAA...",
  "rows": 4096,
  "cols": 4096
}
```

Response:
```json
{
  "quantized_size": 4194304,
  "original_size": 67108864,
  "compression_ratio": 16.0,
  "stats": {
    "positive_ratio": 0.33,
    "negative_ratio": 0.33,
    "zero_ratio": 0.34,
    "scale": 1.0
  }
}
```

### GET /api/v1/capabilities

```json
{
  "quantization": ["ternary_1.58bit", "int8"],
  "inference_modes": ["dense", "speculative_decoding", "moe_top_k", "weight_streaming"],
  "supported_models": ["custom_ternary", "bitnet_b1.58"],
  "platform": {
    "arch": "aarch64",
    "simd": "NEON 128-bit (4-wide f32)",
    "cores": 10,
    "memory_bandwidth": "120-273 GB/s (Apple Silicon) / 17 GB/s (RasPi 5)"
  }
}
```

## Quick Start

```bash
# Core Engine のみ起動
cd services/core-engine
cargo run --release

# ヘルスチェック
curl http://localhost:8081/health

# 推論リクエスト
curl -X POST http://localhost:8081/api/v1/infer \
  -H "Content-Type: application/json" \
  -d '{"input": "Hello", "max_tokens": 64}'
```

## ALICE Crate Dependencies

| Crate | Role | Feature Flag |
|-------|------|-------------|
| `alice-ml` | 1.58bit推論、Speculative Decoding、MoE | `alice-core` |
| `alice-simd` | fast_exp, RMSNorm, Softmax | `alice-core` |
| `alice-cache` | KVキャッシュ、Tiered Storage | (planned) |

## Author

Moroya Sakamoto
