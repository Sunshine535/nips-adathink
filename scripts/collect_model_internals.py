#!/usr/bin/env python3
"""
Complete inference script with model internals capture.
Generates data for SOTA adaptive controller.
"""
import torch
import argparse
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
import json
from pathlib import Path

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default='Qwen/Qwen3.5-27B')
    parser.add_argument('--dataset', default='gsm8k')
    parser.add_argument('--budgets', nargs='+', type=int, default=[128, 256, 512])
    parser.add_argument('--n_samples', type=int, default=100)
    parser.add_argument('--output', default='results/internals/')
    args = parser.parse_args()

    Path(args.output).mkdir(parents=True, exist_ok=True)

    # Load model
    print(f"Loading {args.model}...")
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.bfloat16,
        device_map='auto'
    )
    tokenizer = AutoTokenizer.from_pretrained(args.model)

    # Load dataset
    dataset = load_dataset('openai/gsm8k', 'main', split='test')

    results = []
    for idx, sample in enumerate(dataset.select(range(args.n_samples))):
        question = sample['question']
        answer = sample['answer']

        sample_results = {'idx': idx, 'question': question, 'answer': answer}

        for budget in args.budgets:
            outputs = model.generate(
                **tokenizer(question, return_tensors='pt').to(model.device),
                max_new_tokens=budget,
                output_hidden_states=True,
                output_scores=True,
                return_dict_in_generate=True,
                do_sample=False
            )

            # Extract internals
            logits = torch.stack(outputs.scores)
            probs = torch.softmax(logits, dim=-1)

            sample_results[f'b{budget}_entropy'] = -(probs * torch.log(probs + 1e-10)).sum(-1).mean().item()
            sample_results[f'b{budget}_max_prob'] = probs.max(-1)[0].mean().item()
            sample_results[f'b{budget}_output'] = tokenizer.decode(outputs.sequences[0])

        results.append(sample_results)

        if (idx + 1) % 10 == 0:
            print(f"Processed {idx + 1}/{args.n_samples}")

    # Save
    output_file = Path(args.output) / f'internals_{args.model.split("/")[-1]}.json'
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"Saved to {output_file}")

if __name__ == '__main__':
    main()
