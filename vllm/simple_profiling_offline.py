# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import os
import time
import torch
from vllm import LLM, SamplingParams

#os.environ["VLLM_DISABLE_COMPILE_CACHE"] = "1"
#os.environ["VLLM_XPU_ENABLE_XPU_GRAPH"] = "1"

# # For IPEX
os.environ["VLLM_OFFLOAD_WEIGHTS_BEFORE_QUANT"] = "1"
os.environ["TORCH_LLM_ALLREDUCE"] = "1"

# # enable torch profiler, can also be set on cmd line
# os.environ["VLLM_TORCH_PROFILER_DIR"] = "./vllm_profile"

one_k_prompt = "Mr. and Mrs. Dursley, of number four, Privet Drive, were proud to say that they were perfectly normal, thank you very much. They were the last people you'd expect to be involved in anything strange or mysterious, because they just didn't hold with such nonsense. Mr. Dursley was the director of a firm called Grunnings, which made drills. He was a big, beefy man with hardly any neck, although he did have a very large mustache. Mrs. Dursley was thin and blonde and had nearly twice the usual amount of neck, which came in very useful as she spent so much of her time craning over garden fences, spying on the neighbors. The Dursleys had a small son called Dudley and in their opinion there was no finer boy anywhere. The Dursleys had everything they wanted, but they also had a secret, and their greatest fear was that somebody would discover it. They didn't think they could bear it if anyone found out about the Potters. Mrs. Potter was Mrs. Dursley's sister, but they hadn't met for several years; in fact, Mrs. Dursley pretended she didn't have a sister, because her sister and her good-for-nothing husband were as unDursleyish as it was possible to be. The Dursleys shuddered to think what the neighbors would say if the Potters arrived in the street. The Dursleys knew that the Potters had a small son, too, but they had never even seen him. This boy was another good reason for keeping the Potters away; they didn't want Dudley mixing with a child like that.When Mr. and Mrs. Dursley woke up on the dull, gray Tuesday our story starts, there was nothing about the cloudy sky outside to suggest that strange and mysterious things would soon be happening all over the country. Mr. Dursley hummed as he picked out his most boring tie for work, and Mrs. Dursley gossiped away happily as she wrestled a screaming Dudley into his high chair. None of them noticed a large, tawny owl flutter past the window. At half past eight, Mr. Dursley picked up his briefcase, pecked Mrs. Dursley on the cheek, and tried to kiss Dudley good-bye but missed, because Dudley was now having a tantrum and throwing his cereal at the walls. “Little tyke,” chortled Mr. Dursley as he left the house. He got into his car and backed out of number four's drive. It was on the corner of the street that he noticed the first sign of something peculiar — a cat reading a map. For a second, Mr. Dursley didn't realize what he had seen — then he jerked his head around to look again. There was a tabby cat standing on the corner of Privet Drive, but there wasn't a map in sight. What could he have been thinking of? It must have been a trick of the light. Mr. Dursley blinked and stared at the cat. It stared back. As Mr. Dursley drove around the corner and up the road, he watched the cat in his mirror. It was now reading the sign that said Privet Drive — no, looking at the sign; cats couldn't read maps or signs. Mr. Dursley gave himself a little shake and put the cat out of his mind. As he drove toward town he thought of nothing except a large order of drills he was hoping to get that day. But on the edge of town, drills were driven out of his mind by something else. As he sat in the usual morning traffic jam, he couldn't help noticing that there seemed to be a lot of strangely dressed people about. People in cloaks. Mr. Dursley couldn't bear people who dressed in funny clothes — the getups you saw on young people! He supposed this was some stupid new fashion. He drummed his fingers on the steering wheel and his eyes fell on a huddle of these weirdos standing quite close by. They were whispering excitedly together. Mr. Dursley was enraged to see that a couple of them weren't young at all; why, that man had to be older than he was, and wearing an emerald-green cloak! The nerve of him! But then it struck Mr. Dursley that this was probably some silly stunt — these people were obviously collecting for something... yes, that would be it. The traffic moved on and a few minutes later, Mr. Dursley arrived in the Grunnings parking lot, his mind back on drills. Mr. Dursley always sat with his back to the window in his office on the ninth floor. If he hadn't, he might have found it harder to concentrate on drills that morning."
prompt_len_in_k = 4
decode_len_in_k = .25

# Sample prompts.
prompts = [
    one_k_prompt * prompt_len_in_k,
]

# Create a sampling params object.
sampling_params = SamplingParams(
    temperature=0,
    max_tokens=int(10),
    min_tokens=int(10),
)

enable_profile = True


def main():
    model_name = "/home/chaos/prjs/models/Qwen3.8-27B-AWQ-INT4"

    # Create an LLM.
    llm = LLM(
        model=model_name,
        max_num_batched_tokens=8192,
        dtype="float16",
        quantization="compressed-tensors",
        enable_prefix_caching=False,
        max_model_len=8192,
        block_size=64,
        gpu_memory_utilization=0.9,
        enforce_eager=True,
        # XPU + IPEX sym_int4 path hits an unsupported TorchDynamo graph break
        # when fullgraph capture calls torch.xpu.has_xmx() via the pybind layer.
        # Disabling graph capture keeps the offline profile usable on this stack.
        profiler_config={
            "profiler": "torch",
            "torch_profiler_dir": "/home/chaos/tmp/qwen38-27b-awq-int4-vllm_profile",
            "active_iterations": 5,
            "torch_profiler_with_memory": True,
            "torch_profiler_record_shapes": True,
        },
    )

    print("Model loaded.")

    if enable_profile:
        # warmup
        outputs = llm.generate(prompts, sampling_params)
        torch.xpu.synchronize()
        llm.start_profile()

    # profile
    #with torch.autograd.profiler.emit_itt(record_shapes=False):
    outputs = llm.generate(prompts, sampling_params)
    torch.xpu.synchronize()
    if enable_profile:
        llm.stop_profile()
    
    # Print the generated outputs
    for output in outputs:
        prompt = output.prompt
        generated_text = output.outputs[0].text
        print(f"------------------------------------------------------")
        print(f"Generated text: {generated_text!r}")


if __name__ == "__main__":
    main()
