import os
os.environ.update({ #关闭所有日志
    #vLLM
    #"VLLM_LOGGING_LEVEL": "ERROR",
    #HuggingFace
    "TRANSFORMERS_VERBOSITY": "error",
    "HF_HUB_DISABLE_PROGRESS_BARS": "1",
    "TOKENIZERS_PARALLELISM": "false",
    "VLLM_OFFLOAD_WEIGHTS_BEFORE_QUANT":"1",
    "TORCH_LLM_ADDREDUCE":"1",
    #oneCCL
    "CCL_LOG_LEVEL": "error",
    "CCL_ATL_TRANSPORT": "ofi",
})
from vllm import LLM, SamplingParams
import torch
import random
#INPUT_LENS=[1024, 2048, 4096, 8192, 10240]#, 80_000],,, 80K 会OOM
#INPUT_LENS=[4000, 40000]
INPUT_LENS=[1024] #2048, 3854, 4096.... will failed for profiling
MAX_OUTPUT=512
MML=8192 #INPUT_LENS[-1] + MAX_OUTPUT# max model len, 须满足MML > inputTokenLen + outputTokenLen
GMU=0.8  # gpu memory utilization
MNS=1
MNBT=MML*MNS# max number batched tokens, 
LOOPS=5
sampling_params = SamplingParams(temperature=0.8, top_p=0.95, ignore_eos=True, max_tokens=MAX_OUTPUT)
ttft_params     = SamplingParams(temperature=0.8, top_p=0.95, ignore_eos=True, max_tokens=1)
def build_prompt_token_ids(tokenizer, target_len):
    vocab_size = getattr(tokenizer, "vocab_size", None) or len(tokenizer)
    special_ids = set(getattr(tokenizer, "all_special_ids", []) or [])
    ids = []
    while len(ids) < target_len:
        t = random.randint(0, vocab_size - 1)
        if t not in special_ids:
            ids.append(t)
    return ids
def main():
    random.seed(0)
    llm = LLM(model="/home/chaos/prjs/models/Qwen3.8-27B-AWQ-INT4",
              gpu_memory_utilization=GMU,
              enforce_eager=True, #禁用XPU/NV Graph
              max_model_len=MML, 
              dtype="float16",
              quantization="compressed-tensors", 
              trust_remote_code=True,
              tensor_parallel_size=1, #GPU个数
              block_size=64, 
              max_num_batched_tokens=MNBT,  #每批次最大的token总数, 影响chunk prefill
              max_num_seqs=MNS,         #最大并发数
              disable_log_stats=True,  #禁用日志
              enable_prefix_caching=False,
              profiler_config={
                  "profiler": "torch",
                  "torch_profiler_dir":"/home/chaos/tmp/torchProfile_qwen38_27b_awq_int4",
                  "torch_profiler_record_shapes":True,
                  "active_iterations":5,
                  "torch_profiler_with_memory":True,
                  })
    tokenizer = llm.get_tokenizer()
    def _avg(xs): return sum(xs) / len(xs) if xs else float("nan")
    summary = []

    for in_len in INPUT_LENS:
        token_ids = build_prompt_token_ids(tokenizer, in_len)
        print(f"\033[36mInput length={in_len}\033[0m")
        llm.generate({"prompt_token_ids": token_ids}, sampling_params, use_tqdm=False)
        print("\033[41mStart torchProfiler collection\033[0m") ####ignore the LLM obj init
        torch.xpu.synchronize();
        llm.start_profile()
        for _ in range(LOOPS):
            outs = llm.generate({"prompt_token_ids": token_ids}, sampling_params, use_tqdm=False)
        torch.xpu.synchronize();
        llm.stop_profile();
    print("\033[41mStopped torchProfiler collection\033[0m")
if __name__ == "__main__":
    main()
