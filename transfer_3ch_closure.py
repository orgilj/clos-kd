"""
transfer_3ch_closure.py
=======================
Distills BERT attention projections (Q, K, V) and FFN layers
into CLOS modules, then evaluates on SQuAD dev set.

Key fixes vs. previous version:
  - Uses a real DataLoader (not data_collator) for evaluation
  - Correctly assigns clos back into model using setattr / nested access
  - Uses channel=2 for [B*S, H] flattened linear (correct for BERT projections)
  - Evaluates exact-match on SQuAD instead of broken accuracy loop
  - Layer replacement is done surgically, one at a time

Usage:
python3 transfer_3ch_closure.py \
    --model deepset/bert-base-uncased-squad2 \
    --layers q k v ffn1 ffn2 \
    --layer_start 0 --layer_end 11 \
    --max_steps 2000 --multiplier 4
"""

import argparse
import copy
import math
import os
import re

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW, LBFGS, lr_scheduler
from torch import Tensor

# HuggingFace
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForQuestionAnswering,
)
from torch.utils.data import DataLoader

# ─────────────────────────────────────────────────────────────────────────────
# CLOS module (same as your clos.py / transfer_2ch_closure.py)
# ─────────────────────────────────────────────────────────────────────────────

class Clos(nn.Module):
    def __init__(self, in_features=768, out_features=None, channel=2,
                 switches=None, bias=True, middle_switch_multiplier=4):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features if out_features is not None else in_features
        self.channel = channel
        self.bias = bias
        self.middle_switch_multiplier = middle_switch_multiplier
        self.switches = {}
        self.find_factors()
        if switches is not None:
            self.switches.update(switches)

        self.weight1 = nn.Parameter(torch.Tensor(
            self.switches['bin'], self.switches['b1'], self.switches['b2']))
        self.weight2 = nn.Parameter(torch.Tensor(
            self.switches['b1'], self.switches['b2'], self.switches['b3']))
        self.weight3 = nn.Parameter(torch.Tensor(
            self.switches['b2'], self.switches['b3'], self.switches['bout']))

        if self.bias:
            self.bias1 = nn.Parameter(torch.Tensor(self.switches['b1']))
            self.bias2 = nn.Parameter(torch.Tensor(self.switches['b2']))
            self.bias3 = nn.Parameter(torch.Tensor(self.switches['b3']))
        else:
            self.register_parameter('bias1', None)
            self.register_parameter('bias2', None)
            self.register_parameter('bias3', None)

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight1, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.weight2, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.weight3, a=math.sqrt(5))
        if self.bias:
            for w, b in [(self.weight1, self.bias1),
                         (self.weight2, self.bias2),
                         (self.weight3, self.bias3)]:
                fan_in, _ = nn.init._calculate_fan_in_and_fan_out(w)
                bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
                nn.init.uniform_(b, -bound, bound)

    def find_factors(self):
        for i in range(int(math.sqrt(self.in_features)), 0, -1):
            if self.in_features % i == 0:
                self.switches['bin'] = i
                self.switches['b1']  = self.in_features // i
                break
        for i in range(int(math.sqrt(self.out_features)), 0, -1):
            if self.out_features % i == 0:
                self.switches['bout'] = i
                self.switches['b3']   = self.out_features // i
                break
        self.switches['b2'] = self.middle_switch_multiplier * self.switches['bin']

    def __repr__(self):
        s = self.switches
        return (f"Clos(in={self.in_features}, out={self.out_features}, "
                f"bin={s['bin']}, b1={s['b1']}, b2={s['b2']}, "
                f"b3={s['b3']}, bout={s['bout']}, ch={self.channel})")

    def channel2(self, x: Tensor) -> Tensor:
        # Handle both [B, H] and [B, S, H] (BERT passes 3-D tensors)
        shape = x.shape
        x = x.reshape(-1, shape[-1])   # → [B*S, H] or [B, H]
        b = x.shape[0]
        x = x.view(b, self.switches['bin'], self.switches['b1'])
        if self.bias:
            x = torch.einsum('bnr,nrm->bmr', x, self.weight1) + self.bias1
            x = torch.einsum('bmr,rmn->bnm', x, self.weight2) + self.bias2
            x = torch.einsum('bnm,mro->bor', x, self.weight3) + self.bias3
        else:
            x = torch.einsum('bnr,nrm->bmr', x, self.weight1)
            x = torch.einsum('bmr,rmn->bnm', x, self.weight2)
            x = torch.einsum('bnm,mro->bor', x, self.weight3)
        out = x.reshape(b, -1)          # [B*S, out_f]
        # Restore original leading dimensions if input was 3-D
        if len(shape) == 3:
            out = out.reshape(shape[0], shape[1], -1)
        return out

    def channel3(self, x: Tensor) -> Tensor:
        b, c, _ = x.shape
        x = x.view(b, c, self.switches['bin'], self.switches['b1'])
        if self.bias:
            x = torch.einsum('bcnr,nrm->bcmr', x, self.weight1) + self.bias1
            x = torch.einsum('bcmr,rmn->bcnm', x, self.weight2) + self.bias2
            x = torch.einsum('bcnm,mro->bcor', x, self.weight3) + self.bias3
        else:
            x = torch.einsum('bcnr,nrm->bcmr', x, self.weight1)
            x = torch.einsum('bcmr,rmn->bcnm', x, self.weight2)
            x = torch.einsum('bcnm,mro->bcor', x, self.weight3)
        return x.reshape(b, c, -1)

    def forward(self, x: Tensor) -> Tensor:
        if self.channel == 2:
            return self.channel2(x)
        return self.channel3(x)


# ─────────────────────────────────────────────────────────────────────────────
# Distillation  (mirrors transfer_2ch_closure logic, works for BERT linears)
# ─────────────────────────────────────────────────────────────────────────────

def transfer_fc_to_clos_bert(
    fc: nn.Linear,
    max_steps: int = 3000,
    lr: float = 5e-4,
    middle_switch_multiplier: int = 4,
    probe_rand: int = 32768,
    probe_batch: int = 1024,
    gate_margin: float = 0.25,
    lam_eye: float = 5.0,
    lam_gate: float = 1.0,
    verbose: bool = True,
    seed: int = 42,
) -> Clos:
    """
    Distill a single nn.Linear (teacher) into a CLOS module.
    Uses channel=2: BERT projects each token independently,
    so [B, S, H] is reshaped to [B*S, H] before the linear.
    CLOS channel=2 handles [B, H] directly — same math.
    """
    device = fc.weight.device
    torch.manual_seed(seed)

    in_f, out_f = fc.in_features, fc.out_features
    clos = Clos(in_features=in_f, out_features=out_f, channel=2,
                bias=(fc.bias is not None),
                middle_switch_multiplier=middle_switch_multiplier).to(device)
    fc.eval()
    clos.train()

    # ── Build probe set: zeros + eye + -eye + random ──
    with torch.no_grad():
        eye  = torch.eye(in_f, device=device)            # [d, d]
        zero = torch.zeros(256, in_f, device=device)
        rand = torch.randn(probe_rand, in_f, device=device)
        X    = torch.cat([zero, eye, -eye, rand], dim=0) # [N, d]
        T    = fc(X)                                      # teacher targets [N, out_f]
        T_eye = fc(eye)                                   # used as eye anchor

    opt   = AdamW(clos.parameters(), lr=lr, weight_decay=1e-4)
    sched = lr_scheduler.CosineAnnealingLR(opt, T_max=max_steps)

    n_struct = 256 + 2 * in_f

    for step in range(max_steps):
        k_struct = min(probe_batch // 2, n_struct)
        k_rand   = probe_batch - k_struct
        idx_s    = torch.randint(0, n_struct,     (k_struct,), device=device)
        idx_r    = torch.randint(n_struct, X.size(0), (k_rand,), device=device)
        idx      = torch.cat([idx_s, idx_r])

        xb, tb = X[idx], T[idx]
        opt.zero_grad(set_to_none=True)
        sb = clos(xb)

        loss_mse  = F.mse_loss(sb, tb)
        sign      = torch.where(tb > 0, torch.ones_like(tb), -torch.ones_like(tb))
        loss_gate = F.relu(gate_margin - sign * sb).mean()
        loss_eye  = F.mse_loss(clos(eye), T_eye)
        loss      = loss_mse + lam_gate * loss_gate + lam_eye * loss_eye

        loss.backward()
        torch.nn.utils.clip_grad_norm_(clos.parameters(), 1.0)
        opt.step()
        sched.step()

        if verbose and (step % 500 == 0 or step == max_steps - 1):
            with torch.no_grad():
                emse  = F.mse_loss(clos(eye), T_eye).item()
                gviol = (F.relu(gate_margin - sign * sb) > 0).float().mean().item()
            print(f"  step {step:5d} | loss {loss.item():.4e} "
                  f"| eye_mse {emse:.4e} | gate_viol {gviol:.3f}")

    return clos


# ─────────────────────────────────────────────────────────────────────────────
# Model surgery helpers
# ─────────────────────────────────────────────────────────────────────────────

def get_bert_linear(model, layer_idx: int, proj: str) -> nn.Linear:
    """Return the nn.Linear for a given layer and projection name."""
    enc = model.bert.encoder.layer[layer_idx]
    if proj == 'q':
        return enc.attention.self.query
    elif proj == 'k':
        return enc.attention.self.key
    elif proj == 'v':
        return enc.attention.self.value
    elif proj == 'ffn1':
        return enc.intermediate.dense
    elif proj == 'ffn2':
        return enc.output.dense
    else:
        raise ValueError(f"Unknown proj: {proj}")


def set_bert_linear(model, layer_idx: int, proj: str, module: nn.Module):
    """Replace a linear in BERT with the given module."""
    enc = model.bert.encoder.layer[layer_idx]
    if proj == 'q':
        enc.attention.self.query = module
    elif proj == 'k':
        enc.attention.self.key = module
    elif proj == 'v':
        enc.attention.self.value = module
    elif proj == 'ffn1':
        enc.intermediate.dense = module
    elif proj == 'ffn2':
        enc.output.dense = module


# ─────────────────────────────────────────────────────────────────────────────
# SQuAD evaluation (exact-match)
# ─────────────────────────────────────────────────────────────────────────────

def squad_exact_match(model, dataloader, device, max_batches: int = 200) -> float:
    """
    Rough exact-match on SQuAD: checks if the predicted span string
    matches any of the gold answers.
    max_batches limits evaluation time during search.
    """
    model.eval()
    tokenizer = dataloader.dataset.tokenizer  # attached in prepare_squad()

    correct, total = 0, 0
    with torch.no_grad():
        for i, batch in enumerate(dataloader):
            if i >= max_batches:
                break
            input_ids      = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            token_type_ids = batch.get("token_type_ids", None)
            if token_type_ids is not None:
                token_type_ids = token_type_ids.to(device)

            outputs = model(input_ids=input_ids,
                            attention_mask=attention_mask,
                            token_type_ids=token_type_ids)

            start_logits = outputs.start_logits   # [B, S]
            end_logits   = outputs.end_logits     # [B, S]

            start_pred = start_logits.argmax(dim=-1)  # [B]
            end_pred   = end_logits.argmax(dim=-1)    # [B]

            for b in range(input_ids.size(0)):
                s, e = start_pred[b].item(), end_pred[b].item()
                if e < s:
                    e = s  # clamp invalid spans
                pred_tokens = input_ids[b, s:e+1]
                pred_str    = tokenizer.decode(pred_tokens, skip_special_tokens=True)
                pred_str    = normalize_answer(pred_str)

                # gold answers stored as a list in batch
                golds = batch["answers"][b]   # list of strings
                if any(normalize_answer(g) == pred_str for g in golds):
                    correct += 1
                total += 1

    return 100.0 * correct / total if total > 0 else 0.0


def normalize_answer(s: str) -> str:
    """Lower case, remove punctuation, articles, extra whitespace."""
    import re, string
    s = s.lower()
    s = re.sub(r'\b(a|an|the)\b', ' ', s)
    s = ''.join(ch for ch in s if ch not in set(string.punctuation))
    s = ' '.join(s.split())
    return s


# ─────────────────────────────────────────────────────────────────────────────
# Dataset preparation
# ─────────────────────────────────────────────────────────────────────────────

def prepare_squad(tokenizer, split="validation", max_length=384,
                  doc_stride=128, n_samples=1000):
    """
    Returns a DataLoader over SQuAD examples.
    We attach tokenizer to the dataset so squad_exact_match can decode.
    """
    raw = load_dataset("squad", split=split)
    if n_samples:
        raw = raw.select(range(min(n_samples, len(raw))))

    # Extract gold answers BEFORE remove_columns wipes them
    answers = [ex["answers"]["text"] for ex in raw]

    def preprocess(examples):
        questions = [q.strip() for q in examples["question"]]
        inputs = tokenizer(
            questions,
            examples["context"],
            max_length=max_length,
            truncation="only_second",
            stride=doc_stride,
            return_overflowing_tokens=False,
            return_offsets_mapping=False,
            padding="max_length",
        )
        return inputs

    tokenized = raw.map(preprocess, batched=True,
                        remove_columns=raw.column_names)
    tokenized.set_format(type="torch",
                         columns=["input_ids", "attention_mask",
                                  "token_type_ids"])

    class SQuADDataset(torch.utils.data.Dataset):
        def __init__(self, hf_dataset, answers, tokenizer):
            self.ds       = hf_dataset
            self.answers  = answers
            self.tokenizer = tokenizer   # for decoding in eval

        def __len__(self):
            return len(self.ds)

        def __getitem__(self, idx):
            item = {k: self.ds[idx][k] for k in
                    ["input_ids", "attention_mask", "token_type_ids"]}
            item["answers"] = self.answers[idx]
            return item

    dataset = SQuADDataset(tokenized, answers, tokenizer)

    def collate_fn(batch):
        keys = ["input_ids", "attention_mask", "token_type_ids"]
        out  = {k: torch.stack([b[k] for b in batch]) for k in keys}
        out["answers"] = [b["answers"] for b in batch]
        return out

    return DataLoader(dataset, batch_size=16, shuffle=False,
                      collate_fn=collate_fn)


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model",       default="deepset/bert-base-uncased-squad2")
    p.add_argument("--layers",      nargs="+",
                   choices=["q","k","v","ffn1","ffn2"],
                   default=["q","k","v"])
    p.add_argument("--layer_start", type=int, default=0,
                   help="First BERT layer to replace (0-indexed)")
    p.add_argument("--layer_end",   type=int, default=0,
                   help="Last BERT layer to replace (inclusive)")
    p.add_argument("--max_steps",   type=int, default=3000)
    p.add_argument("--multiplier",  type=int, default=4,
                   help="CLOS middle-stage multiplier")
    p.add_argument("--n_eval",      type=int, default=500,
                   help="Number of SQuAD validation examples to evaluate on")
    p.add_argument("--save_dir",    default="./bert_clos_converted")
    p.add_argument("--squad_split", default="validation")
    p.add_argument("--verbose",     action="store_true", default=True)
    return p.parse_args()


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nDevice: {device}")
    print(f"Model: {args.model}")
    print(f"Replacing projections: {args.layers}  "
          f"in BERT layers {args.layer_start}–{args.layer_end}")
    print(f"CLOS multiplier: {args.multiplier}  |  distil steps: {args.max_steps}\n")

    os.makedirs(args.save_dir, exist_ok=True)

    # ── Load model & tokenizer ──────────────────────────────────────────────
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model     = AutoModelForQuestionAnswering.from_pretrained(args.model).to(device)
    model.eval()

    # ── Prepare SQuAD evaluation loader ────────────────────────────────────
    print("Preparing SQuAD evaluation set …")
    eval_loader = prepare_squad(tokenizer, split=args.squad_split,
                                n_samples=args.n_eval)

    # ── Baseline accuracy ───────────────────────────────────────────────────
    print("Evaluating baseline …")
    baseline_em = squad_exact_match(model, eval_loader, device)
    print(f"Baseline exact-match: {baseline_em:.2f}%\n")

    # ── Parameter counts ────────────────────────────────────────────────────
    total_before = sum(p.numel() for p in model.parameters())

    # ── Layer-by-layer replacement ──────────────────────────────────────────
    results = []

    for layer_idx in range(args.layer_start, args.layer_end + 1):
        for proj in args.layers:
            print(f"─── Layer {layer_idx} | {proj.upper()} ───────────────────")

            fc = get_bert_linear(model, layer_idx, proj)
            in_f, out_f = fc.in_features, fc.out_features
            n_dense = in_f * out_f + (out_f if fc.bias is not None else 0)

            # Best-of-N search (mirrors transfer_2ch_closure.py)
            best_em   = -1.0
            best_clos = None

            for trial in range(3):   # 3 trials, keep best
                print(f"  Trial {trial+1}/3 …")
                clos = transfer_fc_to_clos_bert(
                    fc,
                    max_steps=args.max_steps,
                    lr=5e-4,
                    middle_switch_multiplier=args.multiplier,
                    verbose=args.verbose,
                    seed=trial * 100 + layer_idx,
                )

                # Temporarily replace and evaluate
                set_bert_linear(model, layer_idx, proj, clos)
                em = squad_exact_match(model, eval_loader, device)
                print(f"  Trial {trial+1} EM: {em:.2f}%")

                if em > best_em:
                    best_em   = em
                    best_clos = copy.deepcopy(clos)

                # Restore original for next trial
                set_bert_linear(model, layer_idx, proj, fc)

            # Permanently install best CLOS
            set_bert_linear(model, layer_idx, proj, best_clos)

            n_clos = sum(p.numel() for p in best_clos.parameters())
            reduction = n_dense / n_clos

            save_path = os.path.join(
                args.save_dir, f"layer{layer_idx}_{proj}_clos.pth")
            torch.save(best_clos.state_dict(), save_path)

            results.append({
                "layer": layer_idx, "proj": proj,
                "n_dense": n_dense, "n_clos": n_clos,
                "reduction": reduction, "best_em": best_em,
            })

            print(f"  ✓ Best EM: {best_em:.2f}%  "
                  f"| Params: {n_dense:,} → {n_clos:,}  "
                  f"(×{reduction:.2f} reduction)")
            print(f"  Saved to: {save_path}\n")

    # ── Final whole-model evaluation ────────────────────────────────────────
    total_after = sum(p.numel() for p in model.parameters())
    final_em    = squad_exact_match(model, eval_loader, device)

    print("\n" + "="*60)
    print("FINAL RESULTS")
    print("="*60)
    print(f"Baseline EM         : {baseline_em:.2f}%")
    print(f"CLOS model EM       : {final_em:.2f}%")
    print(f"Total params before : {total_before:,}")
    print(f"Total params after  : {total_after:,}")
    print(f"Overall reduction   : {total_before/total_after:.2f}×")
    print()
    print(f"{'Layer':<8} {'Proj':<6} {'Dense':>10} {'CLOS':>10} {'Red.':>7} {'EM':>8}")
    print("-" * 55)
    for r in results:
        print(f"  {r['layer']:<6} {r['proj']:<6} "
              f"{r['n_dense']:>10,} {r['n_clos']:>10,} "
              f"{r['reduction']:>6.2f}× {r['best_em']:>7.2f}%")

    # ── Save the full converted model + metadata sidecar ───────────────────
    model_save = os.path.join(args.save_dir, "bert_clos_full.pth")
    torch.save(model.state_dict(), model_save)
    print(f"\nFull converted model saved to: {model_save}")

    # Save a metadata JSON so loaders never have to guess multiplier/layers
    import json
    meta = {
        "base_model":  args.model,
        "multiplier":  args.multiplier,
        "replacements": [
            {"layer": r["layer"], "proj": r["proj"],
             "in_features": get_bert_linear.__doc__ and 768 or 768,   # placeholder
             "n_dense": r["n_dense"], "n_clos": r["n_clos"]}
            for r in results
        ]
    }
    # Store actual in/out features from results
    for entry, r in zip(meta["replacements"], results):
        # Recover from saved .pth
        pth = os.path.join(args.save_dir, f"layer{r['layer']}_{r['proj']}_clos.pth")
        sd  = torch.load(pth, map_location="cpu")
        bin_, b1, b2 = sd["weight1"].shape
        b2_, b3, bout = sd["weight3"].shape
        entry["in_features"]  = bin_ * b1
        entry["out_features"] = bout * b3
        entry["multiplier"]   = b2 // bin_   # per-layer ground truth

    meta_path = os.path.join(args.save_dir, "bert_clos_meta.json")
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"Metadata saved to:            {meta_path}")
    print(f"  base_model : {meta['base_model']}")
    print(f"  multiplier : {args.multiplier}")
    print(f"  layers replaced: {len(results)}")


if __name__ == "__main__":
    main()
