"""Toggle-equivalence check for the ESIMD MoE grouped GEMM.

The ESIMD kernel replaces the two CUTLASS grouped GEMMs in the decode MoE path.
It is *not* bit-identical to CUTLASS -- the K reduction order differs, so bf16
outputs differ by ~1e-3 relative. That is expected. What must hold is:

  * per-step logprobs agree to within bf16 noise,
  * the greedy token stream stays identical for a long prefix,
  * generations stay coherent.

Because the disable flag and the cached per-layer plan are both resolved once
per process, the two arms must be two separate processes:

    cd /home/chaos/prjs/osa/scalerDev
    .venv/bin/python verify_esimd_moe.py dump --out /tmp/base.json \
        --env DISABLE_ESIMD_MOE_GEMM=1
    .venv/bin/python verify_esimd_moe.py dump --out /tmp/esimd.json
    .venv/bin/python verify_esimd_moe.py compare /tmp/base.json /tmp/esimd.json

``--env`` is applied before vllm is imported, so it also works for
``ESIMD_MOE_GEMM_M_MAX=8192``, which forces prefill through the ESIMD kernel
too. That turns ``--prompt-logprobs`` into a very dense discriminator (one
comparison per prompt position instead of one per generated token); it is slow,
so use it for a correctness sweep, not for perf.

Note the M<=8 gate: with the default ESIMD_MOE_GEMM_M_MAX the kernel only runs
during decode, so only generated tokens exercise it.
"""

import argparse
import json
import os
import sys

# Real text, not random token ids: random-token prompts leave the model with
# near-tied logits, where legitimate bf16 noise flips argmax and the token
# comparison becomes pure noise.
PROMPTS = [
    "The capital of France is",
    "Write a Python function that reverses a linked list.\n\ndef",
    "Explain in two sentences why the sky is blue.",
    "List the first ten prime numbers:",
    "Q: If a train travels 60 km in 45 minutes, what is its speed in km/h?\nA:",
    "Translate to Chinese: The weather is nice today.",
    "def quicksort(arr):\n    if len(arr) <= 1:\n        return arr\n",
    "Summarize: The Industrial Revolution began in Britain in the late 18th "
    "century and spread to Europe and North America.",
]


def dump(args) -> None:
    from vllm import LLM, SamplingParams

    llm = LLM(
        model=args.model,
        gpu_memory_utilization=args.gmu,
        enforce_eager=True,
        max_model_len=args.max_model_len,
        quantization="fp8",
        trust_remote_code=True,
        tensor_parallel_size=args.tp,
        block_size=64,
        max_num_seqs=args.max_num_seqs,
        disable_log_stats=True,
        enable_prefix_caching=False,
    )

    params = SamplingParams(
        temperature=0.0,          # greedy: the only way two arms are comparable
        top_p=1.0,
        seed=0,
        ignore_eos=True,          # fixed step count keeps the arms aligned
        max_tokens=args.max_tokens,
        logprobs=args.logprobs,
        prompt_logprobs=1 if args.prompt_logprobs else None,
    )

    outs = llm.generate(PROMPTS[: args.num_prompts], params, use_tqdm=True)

    records = []
    for prompt, out in zip(PROMPTS[: args.num_prompts], outs):
        comp = out.outputs[0]
        steps = []
        for step in comp.logprobs or []:
            # {token_id: Logprob} -> sorted by logprob so ranks are comparable
            items = sorted(step.items(), key=lambda kv: -kv[1].logprob)
            steps.append([[int(tid), float(lp.logprob)] for tid, lp in items])
        prompt_lp = None
        if args.prompt_logprobs and out.prompt_logprobs:
            prompt_lp = []
            for pos in out.prompt_logprobs:
                if pos is None:  # first position has no prediction
                    prompt_lp.append(None)
                    continue
                tid, lp = max(pos.items(), key=lambda kv: kv[1].logprob)
                prompt_lp.append([int(tid), float(lp.logprob)])
        records.append({
            "prompt": prompt,
            "token_ids": [int(t) for t in comp.token_ids],
            "text": comp.text,
            "steps": steps,
            "prompt_logprobs": prompt_lp,
        })

    payload = {
        "model": args.model,
        "env": {k: os.environ.get(k) for k in (
            "DISABLE_ESIMD_MOE_GEMM", "ESIMD_MOE_GEMM_M_MAX")},
        "max_tokens": args.max_tokens,
        "records": records,
    }
    with open(args.out, "w") as fh:
        json.dump(payload, fh)
    print(f"wrote {args.out}  ({len(records)} prompts, env={payload['env']})")


def _cmp_steps(a, b):
    """(max |dlogprob|, mean |dlogprob|, top1_agree, compared) over common ids."""
    worst = 0.0
    total = 0.0
    count = 0
    top1 = 0
    steps = 0
    for sa, sb in zip(a, b):
        if not sa or not sb:
            continue
        steps += 1
        if sa[0][0] == sb[0][0]:
            top1 += 1
        db = dict(sb)
        for tid, lp in sa:
            if tid in db:
                d = abs(lp - db[tid])
                worst = max(worst, d)
                total += d
                count += 1
    mean = total / count if count else float("nan")
    return worst, mean, (top1, steps), count


def compare(args) -> None:
    with open(args.a) as fh:
        A = json.load(fh)
    with open(args.b) as fh:
        B = json.load(fh)

    print(f"A: {args.a}  env={A['env']}")
    print(f"B: {args.b}  env={B['env']}")
    if A["env"] == B["env"]:
        print("WARNING: both arms ran with the same env -- this compares "
              "nothing. One arm needs DISABLE_ESIMD_MOE_GEMM=1.")
    print()

    hdr = (f"{'#':>2} {'ident_prefix':>12} {'/steps':>7} {'top1':>9} "
           f"{'max|dlp|':>9} {'mean|dlp|':>10}")
    print(hdr)
    print("-" * len(hdr))

    all_worst = 0.0
    full_match = 0
    for i, (ra, rb) in enumerate(zip(A["records"], B["records"])):
        ta, tb = ra["token_ids"], rb["token_ids"]
        prefix = 0
        for x, y in zip(ta, tb):
            if x != y:
                break
            prefix += 1
        worst, mean, (t1, steps), _ = _cmp_steps(ra["steps"], rb["steps"])
        all_worst = max(all_worst, worst)
        if prefix == len(ta) == len(tb):
            full_match += 1
        print(f"{i:>2} {prefix:>12} {len(ta):>7} {t1:>4}/{steps:<4} "
              f"{worst:>9.2e} {mean:>10.2e}")

        if args.verbose and prefix < len(ta):
            print(f"     A: ...{ra['text'][:160]!r}")
            print(f"     B: ...{rb['text'][:160]!r}")

        if ra["prompt_logprobs"] and rb["prompt_logprobs"]:
            pw = pm = 0.0
            n = 0
            for pa, pb in zip(ra["prompt_logprobs"], rb["prompt_logprobs"]):
                if pa is None or pb is None or pa[0] != pb[0]:
                    continue
                d = abs(pa[1] - pb[1])
                pw = max(pw, d)
                pm += d
                n += 1
            if n:
                print(f"     prompt_logprobs: n={n} max={pw:.2e} "
                      f"mean={pm / n:.2e}")
                all_worst = max(all_worst, pw)

    total = min(len(A["records"]), len(B["records"]))
    print()
    print(f"identical generations: {full_match}/{total}")
    print(f"worst |dlogprob| over all compared entries: {all_worst:.3e}")
    print()
    # bf16 has ~3 decimal digits; a logit assembled from a bf16 MoE output
    # should not move by more than O(1e-2) in logprob space from a reduction
    # order change alone. Orders of magnitude above that means a real bug.
    if all_worst > 0.5:
        print("VERDICT: FAIL -- logprobs moved far beyond bf16 noise.")
        sys.exit(1)
    elif all_worst > 0.05:
        print("VERDICT: SUSPICIOUS -- larger than expected; inspect the "
              "per-op mean_rel from bench_moe_grouped_gemm_fp8_pert_kn.py "
              "--cutlass before trusting this.")
        sys.exit(1)
    else:
        print("VERDICT: PASS -- within bf16 reduction-order noise.")


def main() -> None:
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)

    d = sub.add_parser("dump", help="run generations and dump logprobs")
    d.add_argument("--out", required=True)
    d.add_argument("--env", action="append", default=[], metavar="K=V",
                   help="set env var before importing vllm (repeatable)")
    d.add_argument("--model", default="/home/chaos/prjs/models/Qwen3.6-35B-A3B")
    d.add_argument("--max-tokens", type=int, default=64)
    d.add_argument("--logprobs", type=int, default=8)
    d.add_argument("--num-prompts", type=int, default=len(PROMPTS))
    d.add_argument("--prompt-logprobs", action="store_true",
                   help="also dump prefill logprobs; only meaningful with "
                        "ESIMD_MOE_GEMM_M_MAX raised above the prompt length")
    d.add_argument("--max-model-len", type=int, default=4096)
    d.add_argument("--max-num-seqs", type=int, default=1)
    d.add_argument("--gmu", type=float, default=0.8)
    d.add_argument("--tp", type=int, default=1)
    d.set_defaults(func=dump)

    c = sub.add_parser("compare", help="diff two dumps")
    c.add_argument("a")
    c.add_argument("b")
    c.add_argument("-v", "--verbose", action="store_true")
    c.set_defaults(func=compare)

    args = p.parse_args()
    if args.cmd == "dump":
        for kv in args.env:
            k, _, v = kv.partition("=")
            os.environ[k] = v
        os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
        os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    args.func(args)


if __name__ == "__main__":
    main()
