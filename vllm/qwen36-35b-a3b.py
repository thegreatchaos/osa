# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from vllm import LLM, SamplingParams
from time import sleep
from ittapi import collection_control as cc
import ittapi.compat  as itt
import time
INPUT_LENS=[10_000, 20_000, 30_000, 40_000, 50_000, 60_000, 70_000]#, 80_000],,, 80K 会OOM
MAX_OUTPUT=128
MML=max(INPUT_LENS) + MAX_OUTPUT + 64 # max model len
GMU=0.9  # gpu memory utilization
MNBT=max(INPUT_LENS) + MAX_OUTPUT + 64 # max number batched tokens
LOOPS=5
prompts = "The Zen of Python, by Tim Peters Beautiful is better than ugly.  Explicit is better than implicit.  Simple is better than complex.  Complex is better than complicated.  Flat is better than nested.  Sparse is better than dense.  Readability counts.  Special cases aren't special enough to break the rules.  Although practicality beats purity.  Errors should never pass silently.  Unless explicitly silenced.  In the face of ambiguity, refuse the temptation to guess.  There should be one-- and preferably only one --obvious way to do it.  Although that way may not be obvious at first unless you're Dutch.  Now is better than never.  Although never is often better than *right* now.  If the implementation is hard to explain, it's a bad idea.  If the implementation is easy to explain, it may be a good idea.  Namespaces are one honking great idea -- let's do more of those!"

sampling_params = SamplingParams(temperature=0.8, top_p=0.95, ignore_eos=True, max_tokens=MAX_OUTPUT)


def build_prompt_token_ids(tokenizer, target_len):
    base_ids = tokenizer.encode(prompts, add_special_tokens=False)
    if not base_ids:
        raise RuntimeError("Tokenizer returned empty ids for base prompt")
    repeats = (target_len + len(base_ids) - 1) // len(base_ids)
    ids = (base_ids * repeats)[:target_len]
    return ids


def _request_stats(req_output):
    """Return (ttft, decode_time, n_out_tokens) for one RequestOutput.

    Note: in vLLM v1, `arrival_time` is wall-clock while `first_token_ts` /
    `last_token_ts` are monotonic — they cannot be subtracted directly.
    Use `first_token_latency` (engine-computed TTFT) and the same-clock
    `last_token_ts - first_token_ts` for decode time.
    """
    m = req_output.metrics
    n_out = len(req_output.outputs[0].token_ids)
    first_t = getattr(m, "first_token_ts", None) or getattr(m, "first_token_time", None)
    last_t  = getattr(m, "last_token_ts",  None) or getattr(m, "finished_time",   None)
    ttft = getattr(m, "first_token_latency", None)
    if ttft is None:
        arrival = getattr(m, "arrival_time", None)
        ttft = first_t - arrival
    decode_time = last_t - first_t
    return ttft, decode_time, n_out


def main():
    print("\033[41mStart Inference collection\033[0m")
    cc.resume();

    inst = itt.domain_create("inst.vLLM.Qwen3.6-35B-A3B");
    itt.task_begin(inst, "Initialization")
    llm = LLM(model="/home/chaos/prjs/models/Qwen3.6-35B-A3B",revision=None, gpu_memory_utilization=GMU,enforce_eager=True,max_model_len=MML, quantization="fp8", block_size=64, max_num_batched_tokens=MNBT, max_num_seqs=1, disable_log_stats=False, enable_prefix_caching=False)
    itt.task_end(inst)

    infer = itt.domain_create("infer.vLLM.Qwen3.6-35B-A3B")
    tokenizer = llm.get_tokenizer()

    def _avg(xs): return sum(xs) / len(xs) if xs else float("nan")

    summary = []
    for in_len in INPUT_LENS:
        token_ids = build_prompt_token_ids(tokenizer, in_len)
        actual_in_len = len(token_ids)
        print(f"\033[36m--- Input length target={in_len}, actual={actual_in_len} ---\033[0m")

        llm.generate({"prompt_token_ids": token_ids}, sampling_params)

        itt.task_begin(infer, f"Infer-{in_len}")
        ttfts, tpots, ntoks = [], [], []
        loop_t0 = time.perf_counter()
        for _ in range(LOOPS):
            outs = llm.generate({"prompt_token_ids": token_ids}, sampling_params)
            for o in outs:
                ttft, decode_time, n = _request_stats(o)
                ttfts.append(ttft)
                ntoks.append(n)
                if n > 1:
                    tpots.append(decode_time / (n - 1))
        total_wall = time.perf_counter() - loop_t0
        itt.task_end(infer)

        tps = sum(ntoks) / total_wall
        ttft_avg = _avg(ttfts) * 1000
        tpot_avg = _avg(tpots) * 1000
        print(f"\033[32m[in={actual_in_len}] TTFT avg={ttft_avg:.2f}ms  "
              f"min={min(ttfts)*1000:.2f}ms  max={max(ttfts)*1000:.2f}ms\033[0m")
        print(f"\033[32m[in={actual_in_len}] TPOT avg={tpot_avg:.2f}ms  "
              f"min={min(tpots)*1000:.2f}ms  max={max(tpots)*1000:.2f}ms\033[0m")
        print(f"\033[32m[in={actual_in_len}] TPS  {tps:.2f} tok/s  "
              f"({sum(ntoks)} tokens / {total_wall:.2f}s, {LOOPS} requests)\033[0m")
        summary.append((actual_in_len, ttft_avg, tpot_avg, tps))

    print("\n\033[33m=== Summary ===\033[0m")
    print(f"\033[33m{'InputLen':>10} {'TTFT(ms)':>12} {'TPOT(ms)':>12} {'TPS(tok/s)':>12}\033[0m")
    for in_len, ttft_avg, tpot_avg, tps in summary:
        print(f"\033[33m{in_len:>10} {ttft_avg:>12.2f} {tpot_avg:>12.2f} {tps:>12.2f}\033[0m")
    cc.pause()
    print("\033[41mStopped Inference collection\033[0m")
    cc.detach()
if __name__ == "__main__":
    main()
