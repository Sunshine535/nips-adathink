#!/usr/bin/env python3
"""Train CART answer transducer: LoRA adapter for prefix→answer task.

Takes truncated reasoning prefixes from build_cart_training_data.py output,
trains a LoRA adapter on the base model to generate answers from partial reasoning.

Two modes:
  - question_only: input = question only (ablation baseline)
  - prefix_conditioned: input = question + reasoning prefix (main transducer)

Usage:
    python3 scripts/train_cart_transducer.py \
        --config configs/cart/debug_overfit.yaml \
        --train_data results/cart/train_prefixes.jsonl \
        --output_dir checkpoints/cart/debug_overfit \
        --max_steps 100 --overfit_one_batch
"""
import argparse, json, logging, os, sys, time, yaml

import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)


class CARTPrefixDataset(Dataset):
    """Dataset of (question, prefix, gold_answer) for transducer training."""

    def __init__(self, jsonl_path, tokenizer, max_input_len=2048,
                 max_answer_len=128, trace_conditioned=True):
        self.records = []
        with open(jsonl_path) as f:
            for line in f:
                self.records.append(json.loads(line))
        self.tokenizer = tokenizer
        self.max_input_len = max_input_len
        self.max_answer_len = max_answer_len
        self.trace_conditioned = trace_conditioned
        log.info(f"Loaded {len(self.records)} training records")

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        r = self.records[idx]
        question = r["question"]
        prefix = r["reasoning_prefix"] if self.trace_conditioned else ""
        gold = r["gold_answer"]

        if prefix:
            system = "You are an expert mathematician. Given the partial reasoning below, extract or compute the final answer."
            user_text = f"Question: {question}\n\nPartial reasoning: {prefix}\n\nFinal answer:"
        else:
            system = "You are an expert mathematician. Solve this problem."
            user_text = f"Question: {question}\n\nFinal answer:"

        messages = [{"role": "system", "content": system},
                    {"role": "user", "content": user_text}]
        try:
            prompt = self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True,
                enable_thinking=False)
        except TypeError:
            prompt = self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True)

        full_text = prompt + f" {gold}"

        encoded = self.tokenizer(full_text, truncation=True,
                                 max_length=self.max_input_len + self.max_answer_len,
                                 return_tensors="pt")
        input_ids = encoded["input_ids"].squeeze(0)
        attention_mask = encoded["attention_mask"].squeeze(0)

        prompt_encoded = self.tokenizer(prompt, truncation=True,
                                        max_length=self.max_input_len,
                                        return_tensors="pt")
        prompt_len = prompt_encoded["input_ids"].shape[1]

        labels = input_ids.clone()
        labels[:prompt_len] = -100

        return {"input_ids": input_ids, "attention_mask": attention_mask,
                "labels": labels, "prompt_len": prompt_len}


def collate_fn(batch):
    max_len = max(b["input_ids"].shape[0] for b in batch)
    input_ids = torch.zeros(len(batch), max_len, dtype=torch.long)
    attention_mask = torch.zeros(len(batch), max_len, dtype=torch.long)
    labels = torch.full((len(batch), max_len), -100, dtype=torch.long)

    for i, b in enumerate(batch):
        L = b["input_ids"].shape[0]
        input_ids[i, :L] = b["input_ids"]
        attention_mask[i, :L] = b["attention_mask"]
        labels[i, :L] = b["labels"]

    return {"input_ids": input_ids, "attention_mask": attention_mask,
            "labels": labels}


def train(model, tokenizer, dataset, cfg, output_dir, max_steps, overfit_one_batch, eval_generate_every=0):
    os.makedirs(output_dir, exist_ok=True)

    loader = DataLoader(dataset, batch_size=cfg.get("batch_size", 4),
                        shuffle=not overfit_one_batch, collate_fn=collate_fn)

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=cfg.get("learning_rate", 2e-4))

    device = next(model.parameters()).device
    model.train()

    step = 0
    losses = []
    batch_iter = iter(loader)
    if overfit_one_batch:
        fixed_batch = next(batch_iter)
        fixed_batch = {k: v.to(device) for k, v in fixed_batch.items()}

    while step < max_steps:
        if overfit_one_batch:
            batch = fixed_batch
        else:
            try:
                batch = next(batch_iter)
            except StopIteration:
                batch_iter = iter(loader)
                batch = next(batch_iter)
            batch = {k: v.to(device) for k, v in batch.items()}

        outputs = model(**batch)
        loss = outputs.loss
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        losses.append(loss.item())
        step += 1

        if step % 10 == 0 or step == 1:
            log.info(f"Step {step}/{max_steps}: loss={loss.item():.4f}")

        # Generation EM check during overfit
        if eval_generate_every > 0 and step % eval_generate_every == 0 and overfit_one_batch:
            model.eval()
            with torch.no_grad():
                gen_out = model.generate(
                    input_ids=fixed_batch["input_ids"][:1, :fixed_batch["input_ids"].shape[1]//2],
                    max_new_tokens=128, do_sample=False,
                    pad_token_id=tokenizer.eos_token_id)
                gen_text = tokenizer.decode(gen_out[0], skip_special_tokens=True)
                log.info(f"  [gen@{step}] output: {gen_text[-200:]}")
            model.train()

    # Save adapter
    if hasattr(model, "save_pretrained"):
        model.save_pretrained(output_dir)
        tokenizer.save_pretrained(output_dir)
    else:
        torch.save(model.state_dict(), os.path.join(output_dir, "model.pt"))

    # Save loss log
    with open(os.path.join(output_dir, "training_log.json"), "w") as f:
        json.dump({"losses": losses, "max_steps": max_steps,
                   "overfit_one_batch": overfit_one_batch,
                   "final_loss": losses[-1] if losses else None}, f, indent=2)

    log.info(f"Training complete. Final loss: {losses[-1]:.4f}")
    log.info(f"Saved to {output_dir}")
    return losses


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/cart/debug_overfit.yaml")
    p.add_argument("--train_data", required=True)
    p.add_argument("--output_dir", default="checkpoints/cart/debug_overfit")
    p.add_argument("--max_steps", type=int, default=100)
    p.add_argument("--overfit_one_batch", action="store_true")
    p.add_argument("--eval_generate_every", type=int, default=0,
                   help="Evaluate generation EM every N steps during overfit (0=disabled)")
    p.add_argument("--model", default=None)
    args = p.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    model_name = args.model or cfg.get("model", "Qwen/Qwen3-8B")
    training_cfg = cfg.get("training", {})
    trace_conditioned = cfg.get("trace_conditioned", True)

    log.info(f"Loading {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch.bfloat16, device_map="auto",
        trust_remote_code=True)

    # Apply LoRA
    try:
        from peft import LoraConfig, get_peft_model
        lora_cfg = LoraConfig(
            r=training_cfg.get("lora_r", 16),
            lora_alpha=training_cfg.get("lora_alpha", 32),
            target_modules=training_cfg.get("lora_target_modules",
                                            ["q_proj", "k_proj", "v_proj", "o_proj"]),
            bias="none",
            task_type="CAUSAL_LM",
        )
        model = get_peft_model(model, lora_cfg)
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total = sum(p.numel() for p in model.parameters())
        log.info(f"LoRA applied: {trainable:,} / {total:,} trainable ({trainable/total*100:.2f}%)")
    except ImportError:
        log.warning("peft not installed — training full model (NOT recommended)")

    dataset = CARTPrefixDataset(
        args.train_data, tokenizer,
        max_answer_len=training_cfg.get("max_answer_tokens", 128),
        trace_conditioned=trace_conditioned)

    losses = train(model, tokenizer, dataset, training_cfg,
                   args.output_dir, args.max_steps,
                   args.overfit_one_batch or training_cfg.get("overfit_one_batch", False),
                   eval_generate_every=args.eval_generate_every)

    # Report
    report = {
        "model": model_name,
        "train_data": args.train_data,
        "trace_conditioned": trace_conditioned,
        "max_steps": args.max_steps,
        "overfit_one_batch": args.overfit_one_batch,
        "final_loss": losses[-1] if losses else None,
        "loss_at_10": losses[9] if len(losses) >= 10 else None,
        "loss_at_50": losses[49] if len(losses) >= 50 else None,
    }
    log.info(f"\n=== CART TRAINING REPORT ===")
    log.info(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
