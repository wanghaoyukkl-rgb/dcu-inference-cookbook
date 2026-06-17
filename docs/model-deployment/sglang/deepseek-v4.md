# DeepSeek-V4 on SGLang

## 模型简介

DeepSeek-V4 是 DeepSeek 系列面向高吞吐对话、复杂推理与代码生成场景的大语言模型。基于 SGLang 在 DCU 平台可通过 IFB 或 PD 分离模式部署，并兼容 OpenAI API。

## 模型列表

| 模型权重 | 量化方式 | SGLang 版本 | 推荐硬件 | 卡数 | 部署方式 | 启动命令 |
| -------- | -------- | ----------- | -------- | ---- | -------- | -------- |
| [hygon/Deepseek-v4-Flash-FP8-channel](https://www.modelscope.cn/models/hygon/Deepseek-v4-Flash-FP8-channel-IFB) | FP8 W8A8 | 0.5.10 | BW1100 | 8 | IFB | [**`>_`**](#deepseek-v4-flash-fp8-channel-ifb-bw1100-8x-sglang-0510) |
|  | FP8 W8A8 | 0.5.10 | BW1100 | 16 | 1P1D | [**`>_`**](#deepseek-v4-flash-fp8-channel-1p1d-bw1100-16x-sglang-0510) |

## 启动命令

### Deepseek-v4-Flash-FP8-channel IFB BW1100 8x SGLang 0.5.10
#### Prefill

```bash
export NCCL_SOCKET_IFNAME=xxx
export GLOO_SOCKET_IFNAME=xxx
export NCCL_MIN_NCHANNELS=16
export NCCL_MAX_NCHANNELS=16
export SGLANG_OPT_USE_FUSED_STORE_CACHE=false
export SGLANG_OPT_USE_FUSED_HASH_TOPK=true
export SGLANG_OPT_SWIGLU_CLAMP_FUSION=false
export SGLANG_TOPK_TRANSFORM_512_TORCH=false
export SGLANG_OPT_USE_JIT_KERNEL_FUSED_TOPK=true
export SGLANG_JIT_DEEPGEMM_PRECOMPILE=0
export SGLANG_ENABLE_SPEC_V2=1
export USE_DCU_CUSTOM_ALLREDUCE=1
export SGLANG_USE_LIGHTOP=1
export SGLANG_USE_FP8_W8A8_MOE=1
export SGLANG_USE_DEEPGEMM_MOE=1
export ROCSHMEM_DISABLE_HDP_FLUSH=1
export ROCSHMEM_GDA_NUM_QPS_DEFAULT_CTX=288
export ROCSHMEM_HEAP_SIZE=3173741824
export SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK=128
export SGLANG_ROCM_USE_AITER_MOE=1
export SGLANG_USE_OPT_CAT=1
export SGLANG_USE_FUSED_MLA_CAT=1
export SGLANG_USE_LIGHTOP_GROUP_FP8_QUANT=1
export SGLANG_USE_FUSED_DPSKV4_SILU_MUL_FP8_QUANT=1
export SGLANG_USE_LINEAR_BF16_FP32_USE_BLASLT=1
export SGLANG_APPLY_CONFIG_BACKUP=none
export SGLANG_ROCM_USE_AITER_TILELANG_MHC=1
export SGLANG_DSV4_SPLIT_PREFILL_DECODE_MLA=1
export SGLANG_OPT_FLASHMLA_SPARSE_PREFILL=1
export SGLANG_USE_DPSKV4_LIGHTOP_QUANT_K_CACHE=1
export SGLANG_USE_DPSKV4_LIGHTOP_RMSNORM=1
export SGLANG_USE_FUSED_DPSKV4_QNORM_ROPE_KV_ROPE_QUANT=1
export SGLANG_USE_LIGHTOP_EP_MOE_ALIGN=1
export SGLANG_USE_LIGHTOP_EP_SCATTER=1
export SGLANG_USE_LIGHTOP_EP_GATHER=1
export SGLANG_USE_LIGHTOP_TOPK_IDS_POSTPROCESS=1

sglang serve \
 --tp-size 8 \
 --dist-timeout 10000 \
 --watchdog-timeout 3600 \
 --disable-radix-cache \
 --port 30000 \
 --host $(hostname -I | awk '{print $1}') \
 --model-path hygon/Deepseek-v4-Flash-FP8-channel \
 --model-loader-extra-config '{"enable_multithread_load": "true","num_threads": 64}' \
 --trust-remote-code \
 --chunked-prefill-size 32768 \
 --mem-fraction-static 0.8 \
 --disable-flashinfer-autotune \
 --skip-server-warmup \
 --enable-nsa-prefill-context-parallel \
 --nsa-prefill-cp-mode round-robin-split \
 --moe-a2a-backend deepep \
 --deepep-mode auto \
 --deepep-config deepep-config.json \
 --cuda-graph-max-bs 128 \
 --max-total-tokens 1048576
```
#### deepep-config.json
```json
{
  "normal_dispatch": {
    "num_sms": 48,
    "num_max_nvl_chunked_send_tokens": 6,
    "num_max_nvl_chunked_recv_tokens": 256,
    "num_max_rdma_chunked_send_tokens": 6,
    "num_max_rdma_chunked_recv_tokens": 128
  },
  "normal_combine": {
    "num_sms": 48,
    "num_max_nvl_chunked_send_tokens": 4,
    "num_max_nvl_chunked_recv_tokens": 256,
    "num_max_rdma_chunked_send_tokens": 6,
    "num_max_rdma_chunked_recv_tokens": 128
  }
}
```

#### Decode
```bash
export SGLANG_DSV4_SPLIT_PREFILL_DECODE_MLA=1
export SGLANG_ENABLE_SPEC_V2=1
export SGLANG_ROCM_USE_AITER_MOE=1 
export SGLANG_USE_OPT_CAT=1 
export SGLANG_USE_FUSED_MLA_CAT=1 
export SGLANG_USE_LIGHTOP_GROUP_FP8_QUANT=1 
export SGLANG_USE_FUSED_DPSKV4_SILU_MUL_FP8_QUANT=1 
export SGLANG_USE_LINEAR_BF16_FP32_USE_BLASLT=1
export SGLANG_ROCM_USE_AITER_TILELANG_MHC=1
export SGLANG_USE_DPSKV4_LIGHTOP_QUANT_K_CACHE=1
export SGLANG_USE_DPSKV4_LIGHTOP_RMSNORM=1
export SGLANG_USE_FUSED_DPSKV4_QNORM_ROPE_KV_ROPE_QUANT=1
export SGLANG_USE_LIGHTOP_EP_MOE_ALIGN=1
export SGLANG_USE_LIGHTOP_EP_SCATTER=1
export SGLANG_USE_LIGHTOP_EP_GATHER=1
export SGLANG_USE_LIGHTOP_TOPK_IDS_POSTPROCESS=1 
export SGL_CHUNKED_PREFIX_CACHE_THRESHOLD=0
export SGLANG_DISAGGREGATION_BOOTSTRAP_TIMEOUT=1200
export GLIBC_TUNABLES=glibc.rtld.optional_static_tls=0x40000
export SGLANG_SET_CPU_AFFINITY=1
export HIP_KERNEL_BATCH_CEILING=100
export GPU_MAX_HW_QUEUES=3
export HIP_H2D_DISABLE_COPY_BUFFER=0 
export HIP_D2H_DISABLE_COPY_BUFFER=0 
export HIP_H2D_DIRECT_COPY_THRESHOLD=32768 
export HIP_H2D_HSAAPI_COPY_THRESHOLD=32768 
export HIP_D2H_DIRECT_COPY_THRESHOLD=512 
export HIP_D2H_HSAAPI_COPY_THRESHOLD=512 
export USE_DCU_CUSTOM_ALLREDUCE=1
export HIP_KERNEL_EVENT_SYSTENFENCE=1 
export SGLANG_USE_LIGHTOP=1 
export SGLANG_OPT_USE_FUSED_STORE_CACHE=false
export SGLANG_OPT_USE_FUSED_HASH_TOPK=true
export SGLANG_OPT_SWIGLU_CLAMP_FUSION=false
export SGLANG_TOPK_TRANSFORM_512_TORCH=false
export SGLANG_OPT_USE_JIT_KERNEL_FUSED_TOPK=true
export SGLANG_JIT_DEEPGEMM_PRECOMPILE=0
export SGLANG_APPLY_CONFIG_BACKUP=none

sglang serve \
  --trust-remote-code \
  --model-path hygon/Deepseek-v4-Flash-FP8-channel \
  --tp 8 \
  --chunked-prefill-size 2048 \
  --disable-flashinfer-autotune \
  --speculative-algorithm EAGLE \
  --speculative-num-steps 1 \
  --speculative-eagle-topk 1 \
  --speculative-num-draft-tokens 2 \
  --dp-size 8 \
  --enable-dp-attention --moe-dense-tp-size 1  --enable-dp-lm-head \
  --host $(hostname -I | awk '{print $1}') \
  --port 30000 \
```

### Deepseek-v4-Flash-FP8-channel 1P1D BW1100 16x SGLang 0.5.10

网卡配置参考：[IB 网卡](../../troubleshooting/common-issues.md#ib网卡)。

#### P node 0

```bash
export SGLANG_HEALTH_CHECK_TIMEOUT=180
export NCCL_SOCKET_IFNAME=xxx
export GLOO_SOCKET_IFNAME=xxx
export MC_ENABLE_DEST_DEVICE_AFFINITY=1 
export UCX_NET_DEVICES=mlx5_2:1,mlx5_3:1,mlx5_4:1,mlx5_5:1,mlx5_6:1,mlx5_7:1,mlx5_0:1,mlx5_1:1
export NCCL_IB_HCA=mlx5_2:1,mlx5_3:1,mlx5_4:1,mlx5_5:1,mlx5_6:1,mlx5_7:1,mlx5_0:1,mlx5_1:1
export NCCL_MIN_NCHANNELS=16
export NCCL_MAX_NCHANNELS=16
export SGLANG_ENABLE_SPEC_V2=1
export SGLANG_DSV4_SPLIT_PREFILL_DECODE_MLA=1
export SGLANG_OPT_FLASHMLA_SPARSE_PREFILL=1
export SGLANG_OPT_USE_FUSED_STORE_CACHE=false
export SGLANG_OPT_USE_FUSED_HASH_TOPK=true
export SGLANG_OPT_SWIGLU_CLAMP_FUSION=false
export SGLANG_TOPK_TRANSFORM_512_TORCH=false
export SGLANG_OPT_USE_JIT_KERNEL_FUSED_TOPK=true
export SGLANG_JIT_DEEPGEMM_PRECOMPILE=0
export USE_DCU_CUSTOM_ALLREDUCE=1
export HIP_KERNEL_EVENT_SYSTENFENCE=1 
export SGLANG_USE_LIGHTOP=1
export SGLANG_USE_FP8_W8A8_MOE=1
export SGLANG_USE_DEEPGEMM_MOE=1
export SGLANG_ROCM_USE_AITER_MOE=1
export ROCSHMEM_DISABLE_HDP_FLUSH=1
export ROCSHMEM_GDA_NUM_QPS_DEFAULT_CTX=288
export ROCSHMEM_HEAP_SIZE=3173741824
export SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK=128
export SGLANG_USE_OPT_CAT=1
export SGLANG_USE_FUSED_MLA_CAT=1
export SGLANG_USE_LIGHTOP_GROUP_FP8_QUANT=1
export SGLANG_USE_FUSED_DPSKV4_SILU_MUL_FP8_QUANT=1
export SGLANG_USE_LINEAR_BF16_FP32_USE_BLASLT=1
export SGLANG_APPLY_CONFIG_BACKUP=none
export SGLANG_ROCM_USE_AITER_TILELANG_MHC=1
export SGLANG_USE_DPSKV4_LIGHTOP_QUANT_K_CACHE=1
export SGLANG_USE_DPSKV4_LIGHTOP_RMSNORM=1
export SGLANG_USE_FUSED_DPSKV4_QNORM_ROPE_KV_ROPE_QUANT=1
export SGLANG_USE_LIGHTOP_EP_MOE_ALIGN=1
export SGLANG_USE_LIGHTOP_EP_SCATTER=1
export SGLANG_USE_LIGHTOP_EP_GATHER=1
export SGLANG_USE_LIGHTOP_TOPK_IDS_POSTPROCESS=1

option+=" --disaggregation-ib-device mlx5_0,mlx5_1,mlx5_2,mlx5_3,mlx5_4,mlx5_5,mlx5_6,mlx5_7 "
option+=" --disaggregation-mode prefill "

sglang serve ${option} \
 --tp-size 8 \
 --dist-timeout 10000 \
 --watchdog-timeout 3600 \
 --model-path  hygon/Deepseek-v4-Flash-FP8-channel \
 --trust-remote-code \
 --chunked-prefill-size 32768 \
 --mem-fraction-static 0.8 \
 --disable-flashinfer-autotune \
 --disable-cuda-graph \
 --enable-nsa-prefill-context-parallel \
 --nsa-prefill-cp-mode round-robin-split \
 --moe-a2a-backend deepep \
 --deepep-mode auto \
 --deepep-config deepep-config.json \
 --max-total-tokens 1048576 \
 --nnodes 1 \
 --node-rank 0 \
 --host $(hostname -I | awk '{print $1}') --port 30000  \
 --dist-init-addr $(hostname -I | awk '{print $1}'):5123 --nnodes $nodes --node-rank $rank \
```

#### deepep-config.json
```json
{
  "normal_dispatch": {
    "num_sms": 48,
    "num_max_nvl_chunked_send_tokens": 6,
    "num_max_nvl_chunked_recv_tokens": 256,
    "num_max_rdma_chunked_send_tokens": 6,
    "num_max_rdma_chunked_recv_tokens": 128
  },
  "normal_combine": {
    "num_sms": 48,
    "num_max_nvl_chunked_send_tokens": 4,
    "num_max_nvl_chunked_recv_tokens": 256,
    "num_max_rdma_chunked_send_tokens": 6,
    "num_max_rdma_chunked_recv_tokens": 128
  }
}
```

#### D node 0

```bash
export SGLANG_HEALTH_CHECK_TIMEOUT=180
export NCCL_SOCKET_IFNAME=xxx
export GLOO_SOCKET_IFNAME=xxx
export MC_ENABLE_DEST_DEVICE_AFFINITY=1 
export UCX_NET_DEVICES=mlx5_2:1,mlx5_3:1,mlx5_4:1,mlx5_5:1,mlx5_6:1,mlx5_7:1,mlx5_0:1,mlx5_1:1
export NCCL_IB_HCA=mlx5_2:1,mlx5_3:1,mlx5_4:1,mlx5_5:1,mlx5_6:1,mlx5_7:1,mlx5_0:1,mlx5_1:1
export NCCL_MIN_NCHANNELS=16
export NCCL_MAX_NCHANNELS=16
export SGLANG_DSV4_SPLIT_PREFILL_DECODE_MLA=1
export SGLANG_ENABLE_SPEC_V2=1
export SGLANG_ROCM_USE_AITER_MOE=1 
export SGLANG_USE_OPT_CAT=1 
export SGLANG_USE_FUSED_MLA_CAT=1 
export SGLANG_USE_LIGHTOP_GROUP_FP8_QUANT=1 
export SGLANG_USE_FUSED_DPSKV4_SILU_MUL_FP8_QUANT=1 
export SGLANG_USE_LINEAR_BF16_FP32_USE_BLASLT=1
export SGLANG_ROCM_USE_AITER_TILELANG_MHC=1
export SGLANG_USE_DPSKV4_LIGHTOP_QUANT_K_CACHE=1
export SGLANG_USE_DPSKV4_LIGHTOP_RMSNORM=1
export SGLANG_USE_FUSED_DPSKV4_QNORM_ROPE_KV_ROPE_QUANT=1
export SGLANG_USE_LIGHTOP_EP_MOE_ALIGN=1
export SGLANG_USE_LIGHTOP_EP_SCATTER=1
export SGLANG_USE_LIGHTOP_EP_GATHER=1
export SGLANG_USE_LIGHTOP_TOPK_IDS_POSTPROCESS=1 
export SGL_CHUNKED_PREFIX_CACHE_THRESHOLD=0
export SGLANG_DISAGGREGATION_BOOTSTRAP_TIMEOUT=1200
export GLIBC_TUNABLES=glibc.rtld.optional_static_tls=0x40000
export SGLANG_SET_CPU_AFFINITY=1
export HIP_KERNEL_BATCH_CEILING=100
export GPU_MAX_HW_QUEUES=3
export HIP_H2D_DISABLE_COPY_BUFFER=0 
export HIP_D2H_DISABLE_COPY_BUFFER=0 
export HIP_H2D_DIRECT_COPY_THRESHOLD=32768 
export HIP_H2D_HSAAPI_COPY_THRESHOLD=32768 
export HIP_D2H_DIRECT_COPY_THRESHOLD=512 
export HIP_D2H_HSAAPI_COPY_THRESHOLD=512 
export USE_DCU_CUSTOM_ALLREDUCE=1
export HIP_KERNEL_EVENT_SYSTENFENCE=1 
export SGLANG_USE_LIGHTOP=1 
export SGLANG_OPT_USE_FUSED_STORE_CACHE=false
export SGLANG_OPT_USE_FUSED_HASH_TOPK=true
export SGLANG_OPT_SWIGLU_CLAMP_FUSION=false
export SGLANG_TOPK_TRANSFORM_512_TORCH=false
export SGLANG_OPT_USE_JIT_KERNEL_FUSED_TOPK=true
export SGLANG_JIT_DEEPGEMM_PRECOMPILE=0
export SGLANG_APPLY_CONFIG_BACKUP=none

option+=" --disaggregation-ib-device mlx5_0,mlx5_1,mlx5_2,mlx5_3,mlx5_4,mlx5_5,mlx5_6,mlx5_7 "
option+=" --disaggregation-mode decode "

sglang serve ${option} \
  --trust-remote-code \
  --model-path hygon/Deepseek-v4-Flash-FP8-channel \
  --tp-size 8 \
  --chunked-prefill-size 32768 \
  --mem-fraction-static 0.8 \
  --disable-flashinfer-autotune \
  --dp-size 8 \
  --enable-dp-attention --moe-dense-tp-size 1  --enable-dp-lm-head \
  --speculative-algorithm EAGLE \
  --speculative-num-steps 1 \
  --speculative-eagle-topk 1 \
  --speculative-num-draft-tokens 2 \
  --nnodes 1 \
  --node-rank 0 \
  --host $(hostname -I | awk '{print $1}') --port 30000  \
  --dist-init-addr $(hostname -I | awk '{print $1}'):5123 --nnodes $nodes --node-rank $rank \
```

#### Router

```bash
python3 -m sglang_router.launch_router \
  --pd-disaggregation \
  --prefill http://<P_node_ip>:30000 \
  --decode http://<D_node_ip>:30000 \
  --port 10015
```

## API 调用

### IFB

```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:30000/v1", api_key="not-needed")

response = client.chat.completions.create(
    model="hygon/Deepseek-v4-Flash-FP8-channel",
    messages=[{"role": "user", "content": "你好，请介绍一下自己。"}],
    max_tokens=2048,
)
```

```bash
curl http://localhost:30000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model": "hygon/Deepseek-v4-Flash-FP8-channel-IFB", "messages": [{"role": "user", "content": "你好，请介绍一下自己。"}], "max_tokens": 128}'
```

### PD 分离

PD 分离模式下，客户端请求发送到 SGLang Router，而非直接发送到 P/D 节点。Router 示例端口为 `10015`。

```python
client = OpenAI(base_url="http://<router_ip>:10015/v1", api_key="not-needed")
```

```bash
curl http://<router_ip>:10015/v1/chat/completions ...
```
