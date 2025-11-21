import torch
from datasets import load_from_disk
from jiwer import wer
from clos import Clos, transfer_fc_to_clos
import os
from tqdm import tqdm
import numpy as np
from transformers import (
    AutoTokenizer,
    BertConfig,
    BertLMHeadModel
)

torch.manual_seed(42)
def generate_answer(instruction, extra_input=None, tokenizer=None, model=None, max_new_tokens=200):
    prompt = f"Асуулт: {instruction}"
    if extra_input and extra_input.strip():
        prompt += f"\nНэмэлт мэдээлэл: {extra_input}"
    prompt += "\nХариулт:"

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    # CRITICAL: Use eos_token_id and force min generation
    generated_ids = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        min_new_tokens=10,           # Force at least 10 new tokens
        do_sample=True,
        temperature=0.9,
        top_p=0.99,
        repetition_penalty=1.0,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id,
    )

    full = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
    answer = full.split("Хариулт:")[-1].strip()
    answer = answer.replace(tokenizer.eos_token, "").strip()
    return answer
def flat_accuracy(preds, labels):
    pred_flat = np.argmax(preds, axis=1).flatten()
    labels_flat = labels.flatten()
    return np.sum(pred_flat == labels_flat) / len(labels_flat)
def test_model_accuracy(model, test_loader, tokenizer, device="cuda", return_outputs=False, model_outputs=None):
    model = model.to(device)
    model.eval()
    total_eval_loss = 0
    return_preds = []
    i=0
    for batch in tqdm(test_loader, desc="Тестлэх"):
        input = batch['text'][0].split("Хариулт:")[0].strip()
        output = batch['text'][0].split("Хариулт:")[-1].strip()
        pred_answer = generate_answer(input, tokenizer=tokenizer, model=model)
        if return_outputs:
            return_preds.append((pred_answer))
        if model_outputs is not None:
            model_output = model_outputs[i]
            wer_score = wer(pred_answer, model_output)
        else:
            wer_score = wer(pred_answer, output)
        total_eval_loss += wer_score
        i += 1
    accuracy = 1 - (total_eval_loss / len(test_loader))
    
    if return_outputs:
        return accuracy, return_preds
    else:
        return accuracy
# ===================================================================
# 1. CLOS (KD + replacement)
# ===================================================================
def replace_with_clos_distill(model, layer_idx, submodule_path, save_prefix, test_loader, tokenizer, device="cuda", threshold=0.85, max_trials=300, model_outputs=None):
    """
    Жишээ submodule_path:
        "attention.self.query"
        "attention.self.key"
        "attention.self.value"
        # "attention.output.dense"
        # "intermediate.dense"
        # "output.dense"
    """
    # Get the original Linear layer
    if submodule_path == "attention.self.query":
        fc = model.bert.encoder.layer[layer_idx].attention.self.query
    elif submodule_path == "attention.self.key":
        fc = model.bert.encoder.layer[layer_idx].attention.self.key
    elif submodule_path == "attention.self.value":
        fc = model.bert.encoder.layer[layer_idx].attention.self.value
    # elif submodule_path == "attention.output.dense":
    #     fc = model.bert.encoder.layer[layer_idx].attention.output.dense
    # elif submodule_path == "intermediate.dense":
    #     fc = model.bert.encoder.layer[layer_idx].intermediate.dense
    # elif submodule_path == "output.dense":
    #     fc = model.bert.encoder.layer[layer_idx].output.dense
    else:
        raise ValueError(f"Unknown submodule_path: {submodule_path}")

    in_features = fc.in_features
    out_features = fc.out_features
    print(f"\n=== Layer {layer_idx} | {submodule_path} | {in_features} → {out_features} ===")
    best_acc = -100.0
    best_clos = None

    for trial in range(1, max_trials + 1):
        print(f"Trial {trial}/{max_trials} | Generating CLOS...", end=" ")
        clos: Clos = transfer_fc_to_clos(
            fc,
            channel=3,
            W_lr=0.1,
            B_lr=0.3,         
            max_steps=10000,      # илүү сайн approximation
            verbose=False
        ).to(device)
        model.bert.encoder.layer[layer_idx].__setattr__(submodule_path, clos)
        print(model.bert.encoder.layer[layer_idx].__getattr__(submodule_path))
        acc = test_model_accuracy(model, test_loader, tokenizer, device=device, return_outputs=False, model_outputs=model_outputs)
        print(f"Accuracy after replacement: {acc*100:.3f}%")
        if acc > best_acc:
            best_acc = acc
            best_clos = clos
            torch.save(best_clos.state_dict(), f"{save_prefix}_{submodule_path.replace('.', '_')}_best_clos.pth")
            print(f"NEW BEST! Accuracy: {best_acc*100:.3f}% → saved.")

# ===================================================================
# 2. Main function
# ===================================================================
def main():
    def build_example(example):
        instr = example["instruction"]
        inp   = example["input"]
        out   = example["output"]
        text = f"Асуулт: {instr}"
        if inp and inp.strip():
            text += f"\nНэмэлт мэдээлэл: {inp}"
        text += f"\nХариулт: {out}{tokenizer.eos_token}"  # Add EOS at end
        return {"text": text}

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    MODEL_NAME = "toorgil/mongolian-qa-finetuned"
    OUTPUT_CLOS_DIR = "./bert_clos_converted"
    os.makedirs(OUTPUT_CLOS_DIR, exist_ok=True)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    if tokenizer.eos_token is None:
        tokenizer.eos_token = tokenizer.sep_token
        tokenizer.pad_token = tokenizer.eos_token
    config = BertConfig.from_pretrained(MODEL_NAME, is_decoder=True)
    model = BertLMHeadModel.from_pretrained(MODEL_NAME, config=config).to(device)
    model.eval()
    dataset = load_from_disk("/workspace/qa_dataset_properly_loaded")
    datasets = dataset["train"].shuffle(seed=42).select(range(100))
    print("Building text examples …")
    datasets = datasets.map(build_example, remove_columns=datasets.column_names)
    data_loader = torch.utils.data.DataLoader(
        datasets,
        batch_size=1,
        shuffle=False)
    print(f"Original parameters: {sum(p.numel() for p in model.parameters()):,}")
    orig_acc, model_outputs = test_model_accuracy(model, data_loader, tokenizer, device=device, return_outputs=True)
    print(f"Original model WER accuracy: {(1-orig_acc)*100:.3f}%")
    targets = [
        "attention.self.query",
        "attention.self.key",
        "attention.self.value",
        # "attention.output.dense", # not implemented yet
        # "intermediate.dense",
        # "output.dense",
    ]

    # Бүх 12 layer + бүх submodule
    for layer_idx in range(1,12):
        for submodule in targets:
            save_prefix = os.path.join(OUTPUT_CLOS_DIR, f"layer{layer_idx}")
            replace_with_clos_distill(model, layer_idx, submodule, save_prefix, test_loader=data_loader,tokenizer=tokenizer, device=device, threshold=0.60, max_trials=10, model_outputs=model_outputs)

if __name__ == "__main__":
    main()