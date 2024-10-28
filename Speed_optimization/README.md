# 7. 大模型训练推理优化的方法或工具

## 提速方法：
- 1.将模型导入显存，借助Cuda提升推理速度。
- 2.使用TensorRT/Onnx等框架，提升推理速度。

## 实现原理：
- 由于LLM模型都是多层transformer block构成，当前层block计算完毕后，在最终结果仍未给出时，如果把新的请求给到当前层进行计算，不会对上一个请求的计算结果产生干扰。

### 测试结果：
    cd Speed_optimization\Model_Transform\LLM\

    docker-compose up -d

    docker exec -it trt_llm /bin/bash 

    # Qwen
    python3 convert_checkpoint.py --model_dir /root/qwen  --output_dir ./tllm_qwen_fp16_gpu --dtype float16

    trtllm-build --checkpoint_dir ./tllm_qwen_fp16_gpu --output_dir ./trt_engines/qwen-7b/fp16_gpu --gemm_plugin float16

    python3 run.py \
    --input_text "你好，你是谁" \
    --max_output_len 50 \
    --tokenizer_dir /root/qwen \
    --engine_dir trt_engines/qwen-7b/fp16_1_gpu