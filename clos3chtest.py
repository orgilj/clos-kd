# test_clos_bert.py
import torch
import torch.nn as nn
from datasets import load_from_disk
from transformers import AutoTokenizer, BertConfig, BertLMHeadModel
from clos import Clos
from tqdm import tqdm
import os
import re
from jiwer import wer
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

# ==========================
# 1. Model & Tokenizer ачаалах
# ==========================
MODEL_NAME = "/workspace/mongolian_qa_finetuned_last"
CLOS_DIR = "/workspace/clos/bert_clos_converted"

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
if tokenizer.eos_token is None:
    tokenizer.eos_token = tokenizer.sep_token  # Use [SEP] as EOS
    tokenizer.pad_token = tokenizer.eos_token

config = BertConfig.from_pretrained(MODEL_NAME, is_decoder=True)
model = BertLMHeadModel.from_pretrained(MODEL_NAME, config=config).to(device)
model_orig = BertLMHeadModel.from_pretrained(MODEL_NAME, config=config).to(device)
print(f"Original parameters: {sum(p.numel() for p in model.parameters()):,}")

# ==========================
# 2. Бүх .pth файлуудыг жагсааж, автоматаар солих
# ==========================
clos_files = [f for f in os.listdir(CLOS_DIR) if f.endswith("_best_clos.pth")]
pattern = re.compile(r"layer(\d+)_(attention_self_(query|key|value))_best_clos.pth")

print(f"Found {len(clos_files)} CLOS files. Replacing corresponding layers...")

for filename in clos_files:
    match = pattern.match(filename)
    if not match:
        print(f"Skipping {filename} (does not match pattern)")
        continue

    layer_idx = int(match.group(1))
    submodule_name = match.group(2)  # "attention_self_query" гэх мэт

    path = os.path.join(CLOS_DIR, filename)

    # CLOS модуль үүсгээд state_dict ачаалах
    clos = Clos(in_features=768, out_features=768, channel=3).to(device)
    clos.load_state_dict(torch.load(path, map_location=device))
    clos.eval()

    # Яг зөв submodule солих
    if submodule_name == "attention_self_query":
        model.bert.encoder.layer[layer_idx].attention.self.query = clos
    elif submodule_name == "attention_self_key":
        model.bert.encoder.layer[layer_idx].attention.self.key = clos
    elif submodule_name == "attention_self_value":
        model.bert.encoder.layer[layer_idx].attention.self.value = clos

    print(f"Replaced layer {layer_idx} {submodule_name} ← {filename}")

# ==========================
# 3. Параметрийн хэмнэлт харах
# ==========================
new_params = sum(p.numel() for p in model.parameters())
print(f"\nAfter CLOS replacement:")
print(f"Total parameters : {new_params:,}")
print(f"Compression ratio: {110651649 / new_params:.2f}x")
print("New model:\n",model)
# ==========================
# 4. Inference test (WER шалгах)
# ==========================
@torch.no_grad()
def generate_answer(model, instruction, extra_input=None, max_new_tokens=200):
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
    answer = answer.replace(tokenizer.eos_token, "").strip()  # Clean EOS
    return answer

# ==========================
# 5. Туршилтын QA dataset ачаалах
# ==========================
dataset = load_from_disk("/workspace/bert_unet/qa_dataset_properly_loaded")["train"]
test_samples = dataset.shuffle(seed=42).select(range(1))  # 50 жишээ

print("\n" + "="*60)
print("INFERENCE TEST AFTER CLOS REPLACEMENT")
print("="*60)
model.eval()
model_orig.eval()
total_wer = 0.0
total_wer_orig = 0.0
relative_diff = 0.0
for i, example in enumerate(tqdm(test_samples, desc="Testing")):
    question = example["instruction"]
    context = example["input"]
    reference = example["output"].strip()

    prediction = generate_answer(model, instruction=question, extra_input=context)
    prediction_orig = generate_answer(model_orig, instruction=question, extra_input=context)

    # Simple WER (character-level for Mongolian)
    error = wer(reference, prediction)
    error1 = wer(reference, prediction_orig)
    error2 = wer(prediction_orig, prediction)
    
    total_wer += error
    total_wer_orig += error1
    relative_diff += error2

    if i < 10:  # Эхний 10-г хэвлэ
        print(f"\nQ: {question}")
        if context: print(f"Context: {context}")
        print(f"Ref : {reference}")
        print(f"Pred: {prediction}")
        print(f"Pred (orig model): {prediction_orig}")
        print(f"WER : {error:.3f}")
        print(f"WER (orig model): {error1:.3f}")
        print(f"Relative WER between models: {error2:.3f}")

avg_wer = total_wer / len(test_samples)
avg_wer_orig = total_wer_orig / len(test_samples)
avg_wer_diff = relative_diff / len(test_samples)
print("\n" + "="*60)
print(f"Average WER on 100 samples: {avg_wer:.3f} | Original model: {avg_wer_orig:.3f} | Models differences: {avg_wer_diff:.3f}")
print("="*60)
question = "Шинэ мэдлэг хэрхэн хуримтлуулах вэ? "
context = ""
prediction = generate_answer(model, instruction=question, extra_input=context)
print(f"\nSample prediction after CLOS replacement:\nQ: {question}\nA: {prediction}\n")