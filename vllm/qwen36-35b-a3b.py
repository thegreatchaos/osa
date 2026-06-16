import os
os.environ.update({ #关闭所有日志
    #vLLM
    "VLLM_LOGGING_LEVEL": "ERROR",
    #HuggingFace
    "TRANSFORMERS_VERBOSITY": "error",
    "HF_HUB_DISABLE_PROGRESS_BARS": "1",
    "TOKENIZERS_PARALLELISM": "false",
    #oneCCL
    "CCL_LOG_LEVEL": "error",
    "CCL_ATL_TRANSPORT": "ofi",
})
from vllm import LLM, SamplingParams
from ittapi import collection_control as cc
import ittapi.compat  as itt
import time
import random
INPUT_LENS=[10_000, 20_000, 40_000, 60_000, 70_000]#, 80_000],,, 80K 会OOM
#INPUT_LENS=[4000, 40000]
#INPUT_LENS=[1024] #2048, 3854, 4096.... will failed for profiling
MAX_OUTPUT=512
MML=INPUT_LENS[-1] + MAX_OUTPUT# max model len, 须满足MML > inputTokenLen + outputTokenLen
GMU=0.8  # gpu memory utilization
MNBT=8192# max number batched tokens, 
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
    llm = LLM(model="/home/chaos/prjs/models/Qwen3.6-35B-A3B",
              gpu_memory_utilization=GMU,
              enforce_eager=True, #禁用XPU/NV Graph
              max_model_len=MML, 
              quantization="fp8", 
              trust_remote_code=True,
              tensor_parallel_size=1, #GPU个数
              block_size=64, 
              max_num_batched_tokens=MNBT,  #每批次最大的token总数, 影响chunk prefill
              max_num_seqs=1,         #最大并发数
              disable_log_stats=True,  #禁用日志
              enable_prefix_caching=False)
    infer = itt.domain_create("infer.vLLM.Qwen3.6-35B-A3B")
    tokenizer = llm.get_tokenizer()
    def _avg(xs): return sum(xs) / len(xs) if xs else float("nan")
    summary = []

    for in_len in INPUT_LENS:
        token_ids = build_prompt_token_ids(tokenizer, in_len)
        print(f"\033[36mInput length={in_len}\033[0m")

        llm.generate({"prompt_token_ids": token_ids}, sampling_params, use_tqdm=False)
        print("\033[41mStart Inference collection\033[0m") ####ignore the LLM obj init
        cc.resume()
        itt.task_begin(infer, f"Infer-{in_len}")
        ttfts, tpots, ntoks = [], [], []
        loop_t0 = time.perf_counter()
        for _ in range(LOOPS):
            t0 = time.perf_counter()
            llm.generate({"prompt_token_ids": token_ids}, ttft_params, use_tqdm=False)
            ttft = time.perf_counter() - t0

            t1 = time.perf_counter()
            outs = llm.generate({"prompt_token_ids": token_ids}, sampling_params, use_tqdm=False)
            total = time.perf_counter() - t1

            n = len(outs[0].outputs[0].token_ids)
            ttfts.append(ttft)
            ntoks.append(n)
            if n > 1:
                tpots.append((total - ttft) / (n - 1))
        total_wall = time.perf_counter() - loop_t0
        itt.task_end(infer)
        # +LOOPS: each loop also generates 1 token via ttft_params; count it to match total_wall.
        #tps = 1. / tpot(sum(ntoks) + LOOPS) / total_wall
        ttft_avg = _avg(ttfts) * 1000
        tpot_avg = _avg(tpots) * 1000
        tps = 1000.0 / tpot_avg
        print(f"\033[32m\t[in={in_len}] TTFT avg={ttft_avg:.2f}ms  "
              f"min={min(ttfts)*1000:.2f}ms  max={max(ttfts)*1000:.2f}ms\033[0m")
        if tpots:
            print(f"\033[32m\t[in={in_len}] TPOT avg={tpot_avg:.2f}ms  "
                  f"min={min(tpots)*1000:.2f}ms  max={max(tpots)*1000:.2f}ms\033[0m")
        else:
            print(f"\033[32m\t[in={in_len}] TPOT n/a (no decode steps observed)\033[0m")
        print(f"\033[32m\t[in={in_len}] TPS  {tps:.2f} tok/s  "
              f"({sum(ntoks) + LOOPS} tokens / {total_wall:.2f}s, {LOOPS} requests)\033[0m")
        summary.append((in_len, ttft_avg, tpot_avg, tps))
        cc.pause()
    cc.detach()
    print("\n\033[33mSummary::\033[0m")
    print(f"\033[33m\t{'InputLen':>10} {'TTFT(ms)':>12} {'TPOT(ms)':>12} {'TPS(tok/s)':>12}\033[0m")
    for in_len, ttft_avg, tpot_avg, tps in summary:
        print(f"\033[33m\t{in_len:>10} {ttft_avg:>12.2f} {tpot_avg:>12.2f} {tps:>12.2f}\033[0m")
    print("\033[41mStopped Inference collection\033[0m")
if __name__ == "__main__":
    main()
