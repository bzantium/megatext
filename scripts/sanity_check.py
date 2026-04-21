"""Quick sanity check for a converted HF checkpoint.

Usage:
  python custom/sanity_check.py --model checkpoints/pretrain-qwen3-swa-8b/1000
  python custom/sanity_check.py --model checkpoints/pretrain-qwen3-swa-8b/1000 \
      --prompt "The capital of France is" --max-new-tokens 128
"""

from __future__ import annotations

import argparse
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

DEFAULT_PROMPT = "대한민국의 서울은 단연코 세계에서 인정받는"


def main() -> None:
    parser = argparse.ArgumentParser(description="Sanity-check a converted HF model.")
    parser.add_argument("--model", required=True, help="Path to the HF model directory.")
    parser.add_argument("--prompt", default=DEFAULT_PROMPT, help="Input prompt for generation.")
    parser.add_argument("--max-new-tokens", type=int, default=64, help="Maximum new tokens to generate.")
    parser.add_argument("--temperature", type=float, default=None, help="Sampling temperature (default: greedy).")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Loading model from {args.model} to {device} ...")
    t0 = time.time()
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        dtype=torch.bfloat16,
        device_map={"": device},
        trust_remote_code=True,
    )
    print(f"Model loaded in {time.time() - t0:.1f}s")

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)

    inputs = tokenizer(args.prompt, return_tensors="pt").to(device)
    print(f"Prompt ({inputs['input_ids'].shape[-1]} tokens): {args.prompt}")

    do_sample = args.temperature is not None
    gen_kwargs = dict(
        max_new_tokens=args.max_new_tokens,
        do_sample=do_sample,
    )
    if do_sample:
        gen_kwargs["temperature"] = args.temperature

    t1 = time.time()
    with torch.no_grad():
        out = model.generate(**inputs, **gen_kwargs)
    elapsed = time.time() - t1
    new_tokens = out.shape[-1] - inputs["input_ids"].shape[-1]

    print(f"\nGenerated {new_tokens} tokens in {elapsed:.1f}s ({new_tokens / elapsed:.1f} tok/s)")
    print("=" * 60)
    print(tokenizer.decode(out[0], skip_special_tokens=True))
    print("=" * 60)


if __name__ == "__main__":
    main()
