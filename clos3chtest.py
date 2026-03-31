"""
demo_clos_squad.py
==================
Loads a BERT QA model, replaces Q/K/V of layer 0 with saved CLOS weights,
then runs inference on 5 SQuAD samples and prints a side-by-side comparison:

    Dense answer  vs.  CLOS answer  vs.  Gold answer

Usage (after running transfer_3ch_closure.py at least once):

    python3 demo_clos_squad.py \
        --model  deepset/bert-base-uncased-squad2 \
        --clos_dir ./bert_clos_converted \
        --layers q k v \
        --layer  0 \
        --n      5


    python3 demo_clos_squad.py \
        --model  deepset/bert-base-uncased-squad2 \
        --clos_dir ./bert_clos_converted \
        --n 5

If you haven't run the distillation yet, omit --clos_dir and the script
will show Dense-only predictions as a baseline demo.
"""

import argparse
import math
import os
import textwrap

import torch
import torch.nn as nn
from torch import Tensor
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForQuestionAnswering


# ─────────────────────────────────────────────────────────────────────────────
# Minimal CLOS class (must match the one used during distillation)
# ─────────────────────────────────────────────────────────────────────────────

class Clos(nn.Module):
    def __init__(self, in_features=768, out_features=None, channel=2,
                 switches=None, bias=True, middle_switch_multiplier=4):
        super().__init__()
        self.in_features  = in_features
        self.out_features = out_features if out_features is not None else in_features
        self.channel      = channel
        self.bias_flag    = bias
        self.middle_switch_multiplier = middle_switch_multiplier
        self.switches = {}
        self.find_factors()
        if switches is not None:
            self.switches.update(switches)

        self.weight1 = nn.Parameter(torch.Tensor(
            self.switches['bin'], self.switches['b1'], self.switches['b2']))
        self.weight2 = nn.Parameter(torch.Tensor(
            self.switches['b1'],  self.switches['b2'], self.switches['b3']))
        self.weight3 = nn.Parameter(torch.Tensor(
            self.switches['b2'],  self.switches['b3'], self.switches['bout']))

        if bias:
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
        if self.bias1 is not None:
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

    def channel2(self, x: Tensor) -> Tensor:
        shape = x.shape
        x   = x.reshape(-1, shape[-1])
        b   = x.shape[0]
        x   = x.view(b, self.switches['bin'], self.switches['b1'])
        if self.bias1 is not None:
            x = torch.einsum('bnr,nrm->bmr', x, self.weight1) + self.bias1
            x = torch.einsum('bmr,rmn->bnm', x, self.weight2) + self.bias2
            x = torch.einsum('bnm,mro->bor', x, self.weight3) + self.bias3
        else:
            x = torch.einsum('bnr,nrm->bmr', x, self.weight1)
            x = torch.einsum('bmr,rmn->bnm', x, self.weight2)
            x = torch.einsum('bnm,mro->bor', x, self.weight3)
        out = x.reshape(b, -1)
        if len(shape) == 3:
            out = out.reshape(shape[0], shape[1], -1)
        return out

    def forward(self, x: Tensor) -> Tensor:
        return self.channel2(x)


# ─────────────────────────────────────────────────────────────────────────────
# Model surgery helpers
# ─────────────────────────────────────────────────────────────────────────────

def get_bert_linear(model, layer_idx: int, proj: str) -> nn.Linear:
    enc = model.bert.encoder.layer[layer_idx]
    return {"q":    enc.attention.self.query,
            "k":    enc.attention.self.key,
            "v":    enc.attention.self.value,
            "ffn1": enc.intermediate.dense,
            "ffn2": enc.output.dense}[proj]


def set_bert_linear(model, layer_idx: int, proj: str, module: nn.Module):
    enc = model.bert.encoder.layer[layer_idx]
    if   proj == "q":    enc.attention.self.query   = module
    elif proj == "k":    enc.attention.self.key     = module
    elif proj == "v":    enc.attention.self.value   = module
    elif proj == "ffn1": enc.intermediate.dense     = module
    elif proj == "ffn2": enc.output.dense           = module


def load_clos_layer(path: str, in_f: int, out_f: int,
                    device, multiplier: int = 4) -> Clos:
    clos = Clos(in_features=in_f, out_features=out_f,
                channel=2, middle_switch_multiplier=multiplier).to(device)
    clos.load_state_dict(torch.load(path, map_location=device))
    clos.eval()
    return clos


# ─────────────────────────────────────────────────────────────────────────────
# Single-example QA inference
# ─────────────────────────────────────────────────────────────────────────────

def predict(model, tokenizer, question: str, context: str,
            device, max_length: int = 384) -> str:
    model.eval()
    enc = tokenizer(
        question, context,
        max_length=max_length,
        truncation="only_second",
        padding="max_length",
        return_tensors="pt",
    ).to(device)

    with torch.no_grad():
        out = model(**enc)

    start = out.start_logits.argmax(dim=-1).item()
    end   = out.end_logits.argmax(dim=-1).item()
    if end < start:
        end = start

    tokens = enc["input_ids"][0][start : end + 1]
    return tokenizer.decode(tokens, skip_special_tokens=True).strip()


# ─────────────────────────────────────────────────────────────────────────────
# Pretty printer
# ─────────────────────────────────────────────────────────────────────────────

def wrap(text: str, width: int = 72) -> str:
    return "\n    ".join(textwrap.wrap(text, width))


def print_sample(idx, question, context, gold, dense_ans, clos_ans):
    print(f"\n{'─'*72}")
    print(f"Sample {idx+1}")
    print(f"{'─'*72}")
    print(f"Q : {wrap(question)}")
    print(f"\nContext (first 200 chars):\n    {wrap(context[:200])}…")
    print(f"\n  Gold   : {gold}")
    print(f"  Dense  : {dense_ans}")
    print(f"  CLOS   : {clos_ans}")

    # Simple exact-match indicator
    def em(pred, golds):
        import re, string
        def norm(s):
            s = s.lower()
            s = re.sub(r'\b(a|an|the)\b', ' ', s)
            s = ''.join(c for c in s if c not in string.punctuation)
            return ' '.join(s.split())
        return any(norm(pred) == norm(g) for g in golds)

    d_match = "✓" if em(dense_ans, gold) else "✗"
    c_match = "✓" if em(clos_ans,  gold) else "✗"
    print(f"\n  Dense EM: {d_match}   CLOS EM: {c_match}")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model",      default="deepset/bert-base-uncased-squad2")
    p.add_argument("--clos_dir",   default="./bert_clos_converted",
                   help="Directory with layer*_*_clos.pth files from "
                        "transfer_3ch_closure.py  (skip if not yet distilled)")
    p.add_argument("--layers",     nargs="+",
                   choices=["q","k","v","ffn1","ffn2"],
                   default=["q","k","v"])
    p.add_argument("--layer",      type=int, default=0,
                   help="Which BERT layer index to replace")
    p.add_argument("--multiplier", type=int, default=4)
    p.add_argument("--n",          type=int, default=5,
                   help="Number of SQuAD samples to show")
    p.add_argument("--squad_split",default="validation")
    return p.parse_args()


def load_full_clos_model(full_pth: str, base_model_name: str,
                         device, multiplier: int = 4):
    """
    Re-build the CLOS-replaced architecture and load bert_clos_full.pth.

    Strategy:
      1. Load the base BertForQuestionAnswering (defines the architecture).
      2. Inspect which keys in the state_dict have shapes that differ from
         the base model — those are the CLOS layers.
      3. Swap those nn.Linear modules for Clos modules with matching dims.
      4. Load the full state_dict (strict=False so unexpected keys are skipped).
    """
    from transformers import AutoModelForQuestionAnswering
    print(f"  Building base architecture from {base_model_name} …")
    model = AutoModelForQuestionAnswering.from_pretrained(base_model_name).to(device)

    print(f"  Loading state dict from {full_pth} …")
    sd = torch.load(full_pth, map_location=device)

    # ── Detect which linears were replaced by CLOS ────────────────────────
    # CLOS parameters are named  weight1/weight2/weight3/bias1/bias2/bias3
    # Find the unique module prefixes for CLOS tensors
    clos_prefixes = set()
    for k in sd.keys():
        if any(k.endswith(s) for s in
               ("weight1", "weight2", "weight3", "bias1", "bias2", "bias3")):
            clos_prefixes.add(k.rsplit(".", 1)[0])   # e.g. "bert.encoder.layer.0.attention.self.query"

    print(f"  Detected {len(clos_prefixes)} CLOS module(s) in checkpoint:")
    for p in sorted(clos_prefixes):
        print(f"    {p}")

    # ── Swap nn.Linear → Clos for each detected prefix ───────────────────
    def get_module(model, dotpath):
        parts = dotpath.split(".")
        m = model
        for p in parts:
            m = getattr(m, p) if not p.isdigit() else m[int(p)]
        return m

    def set_module(model, dotpath, new_mod):
        parts = dotpath.split(".")
        parent = model
        for p in parts[:-1]:
            parent = getattr(parent, p) if not p.isdigit() else parent[int(p)]
        setattr(parent, parts[-1], new_mod)

    for prefix in sorted(clos_prefixes):
        # Recover in_features / out_features from weight1 shape
        # weight1 shape: [bin, b1, b2]  →  in_features = bin * b1
        w1 = sd[f"{prefix}.weight1"]
        w3 = sd[f"{prefix}.weight3"]
        bin_, b1, b2 = w1.shape
        b2_, b3, bout = w3.shape
        in_f  = bin_ * b1
        out_f = bout * b3
        # middle multiplier = b2 / bin
        mult  = b2 // bin_

        has_bias = f"{prefix}.bias1" in sd

        clos = Clos(in_features=in_f, out_features=out_f,
                    channel=2, bias=has_bias,
                    middle_switch_multiplier=mult).to(device)
        set_module(model, prefix, clos)
        print(f"  ✓ Swapped {prefix}  [{in_f}→{out_f}]  mult={mult}")

    # ── Load weights (strict=False: base model has no weight1/2/3 keys) ──
    missing, unexpected = model.load_state_dict(sd, strict=False)
    truly_missing = [k for k in missing if "weight1" not in k
                     and "weight2" not in k and "weight3" not in k
                     and "bias1"   not in k and "bias2"   not in k
                     and "bias3"   not in k]
    if truly_missing:
        print(f"  [WARN] Truly missing keys: {truly_missing}")
    model.eval()
    return model


def main():
    args   = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device : {device}")

    # ── Decide loading path ─────────────────────────────────────────────────
    full_pth = os.path.join(args.clos_dir, "bert_clos_full.pth")
    use_full = os.path.exists(full_pth)

    tokenizer = AutoTokenizer.from_pretrained(args.model)

    if use_full:
        print(f"Loading full CLOS model from: {full_pth}")
        model = load_full_clos_model(full_pth, args.model, device, args.multiplier)
    else:
        print(f"bert_clos_full.pth not found in {args.clos_dir}")
        print("Loading vanilla dense model instead …")
        model = AutoModelForQuestionAnswering.from_pretrained(args.model).to(device)
        model.eval()

    # ── Load SQuAD samples ──────────────────────────────────────────────────
    print(f"Loading {args.n} SQuAD samples …")
    squad   = load_dataset("squad", split=args.squad_split)
    samples = [squad[i] for i in range(args.n)]

    # ── Dense predictions: reload original model separately ─────────────────
    print("Running Dense (original) predictions for comparison …")
    dense_model = AutoModelForQuestionAnswering.from_pretrained(args.model).to(device)
    dense_model.eval()
    dense_answers = [
        predict(dense_model, tokenizer, s["question"], s["context"], device)
        for s in samples
    ]
    del dense_model

    # ── CLOS predictions ────────────────────────────────────────────────────
    print("Running CLOS predictions …")
    clos_answers = [
        predict(model, tokenizer, s["question"], s["context"], device)
        for s in samples
    ]

    # ── Print side-by-side results ───────────────────────────────────────────
    mode = "bert_clos_full.pth" if use_full else "Dense (no CLOS file found)"
    print(f"{'='*72}")
    print(f"  RESULTS — {args.n} SQuAD samples  |  CLOS source: {mode}")
    print(f"{'='*72}")

    for i, s in enumerate(samples):
        gold = s["answers"]["text"]   # list of gold strings
        print_sample(i, s["question"], s["context"],
                     gold, dense_answers[i], clos_answers[i])

    # ── Summary ─────────────────────────────────────────────────────────────
    import re, string
    def norm(s):
        s = s.lower()
        s = re.sub(r'\b(a|an|the)\b', ' ', s)
        s = ''.join(c for c in s if c not in string.punctuation)
        return ' '.join(s.split())
    def em(pred, golds):
        return any(norm(pred) == norm(g) for g in golds)

    d_em = sum(em(dense_answers[i], samples[i]["answers"]["text"])
               for i in range(args.n))
    c_em = sum(em(clos_answers[i],  samples[i]["answers"]["text"])
               for i in range(args.n))

    print(f"\n{'─'*72}")
    print(f"  Exact-match  |  Dense: {d_em}/{args.n}  |  CLOS: {c_em}/{args.n}")
    print(f"{'─'*72}\n")

    # ── Parameter count ──────────────────────────────────────────────────────
    total = sum(p.numel() for p in model.parameters())
    print(f"  Model params after replacement: {total:,}")


if __name__ == "__main__":
    main()
