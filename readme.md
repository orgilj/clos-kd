# CLOS_KD — [CLOS](https://en.wikipedia.org/wiki/Clos_network) layer Knowledge distillation without Require Big training data and replacement

This repository provides tools to distill pretrained nn.Linear layers into CLOS modules and to swap those modules into existing models. It includes example scripts and utilities for two main experiments:

- MNIST MLP: Replace a hidden linear layer with a CLOS module. In the best run (middle-stage multiplier = 4) test accuracy drops modestly from about 98% to 97% (≈1 percentage point; ~3–5% relative reduction).
- BERT (experimental): Prototype code to distill attention projection layers. No formal accuracy numbers were collected, but generated answers are qualitatively similar to the original model.
- Transformer based all models can be used as reduced, transfered parameters. Use QKV, output and dense Linear layers.
- Next, we will implement transformer layer knowledge distillation.

Using a smaller middle-stage multiplier (for example, 2) can lead to substantial parameter savings — roughly a 5× reduction in some configurations — at the cost of a small drop in performance. In Clos layer in_feature/out_feature proportion should be 4x. For example:(1024, 4096) or (8192, 2048) etc suggested. 

The rest of this README documents the project layout, requirements, and common commands.


Main files:

- [clos.py](clos.py) — CLOS implementation and helper:
  - [`clos.Clos`](clos.py) — CLOS module class.
  - [`clos.transfer_fc_to_clos`](clos.py) — routine to distill an `nn.Linear` into a `Clos`.
- [train_mnist.py](train_mnist.py) — MNIST MLP training & evaluation:
  - [`train_mnist.MNIST_Net`](train_mnist.py) — MLP model used for experiments.
  - [`train_mnist.test`](train_mnist.py) — test/eval helper.
- [transfer_2ch_closure.py](transfer_2ch_closure.py) — example: replace MNIST fc1 with 2-channel CLOS and evaluate.
- [clos2chtest.py](clos2chtest.py) — load best saved CLOS for MNIST and evaluate.
- [transfer_3ch_closure.py](transfer_3ch_closure.py) — attempt to distill attention projection layers (channel=3) of a BERT model using Fourier eye inputs.
  - [`transfer_3ch_closure.generate_answer`](transfer_3ch_closure.py) — inference helper used for WER evaluation.
- [clos3chtest.py](clos3chtest.py) — replace attention submodules from saved CLOS files and run inference WER comparisons.
  - [`clos3chtest.generate_answer`](clos3chtest.py) — inference helper used for WER evaluation.

Quick file links:
- [clos.py](clos.py)
- [train_mnist.py](train_mnist.py)
- [transfer_2ch_closure.py](transfer_2ch_closure.py)
- [clos2chtest.py](clos2chtest.py)
- [transfer_3ch_closure.py](transfer_3ch_closure.py)
- [clos3chtest.py](clos3chtest.py)

Requirements
- Python 3.8+
- PyTorch (GPU recommended)
- torchvision
- transformers
- datasets
- jiwer
- tqdm
Install with:
```bash
pip install torch torchvision transformers datasets jiwer tqdm
```

Quick start / common commands
- Train MNIST MLP:
```bash
python train_mnist.py
# saves mnist_784_256_10.pth
```
- Distill MNIST fc1 → CLOS (try multiple trials):
```bash
python transfer_2ch_closure.py
# saves clos_784_best_test.pth
```
- Test MNIST with saved CLOS:
```bash
python clos2chtest.py
```
- Distill BERT attention projections (prototype / experimental):
```bash
python transfer_3ch_closure.py
# writes ./bert_clos_converted/*.pth (per-layer best)
```
- Replace BERT attention layers from saved CLOS and evaluate WER:
```bash
python clos3chtest.py
```

Notes & gotchas
- For BERT/sequence (channel=3) distillation, the code uses a Fourier-based "eye" via `clos.make_fourier_eye` in [clos.py](clos.py). The training target shapes differ from the MLP case; see [`clos.transfer_fc_to_clos`](clos.py).
- Tokenizer EOS handling is adjusted in [transfer_3ch_closure.py](transfer_3ch_closure.py) and [clos3chtest.py](clos3chtest.py) to ensure generation finishes correctly.
- Compression ratio reported in [clos3chtest.py](clos3chtest.py) is computed as:
$$
\text{compression} \;=\; \frac{110651649}{\text{new\_params}}
$$
where `new_params` is the total parameter count after replacement (printed in that script).

Where to look first
- To understand CLOS internals: open [`clos.Clos`](clos.py).
- For MNIST experiments: [`train_mnist.MNIST_Net`](train_mnist.py) → [`transfer_2ch_closure.py`](transfer_2ch_closure.py) → [`clos2chtest.py`](clos2chtest.py).
- For BERT experiments: [`clos.transfer_fc_to_clos`](clos.py) and [`transfer_3ch_closure.py`](transfer_3ch_closure.py) → [`clos3chtest.py`](clos3chtest.py).

If something fails, check:
- Device/GPU availability (scripts pick `cuda` when available).
- Paths such as the dataset path used in BERT scripts (e.g. `/workspace/qa_dataset_properly_loaded`) and the model names/paths in the scripts.

Output of clos2test.py
```console
Ашиглаж байгаа төхөөрөмж: cuda
Device: cuda
Нийт linear параметр: 615,440
Clone done: MNIST_Net(
  (fc1): Clos(in_features=784, out_features=784, bias=True, bin=28, b1=28, b2=112, b3=28, bout=28, channel=2)
  (fc2): Linear(in_features=784, out_features=256, bias=True)
  (fc3): Linear(in_features=256, out_features=10, bias=True)
  (relu): ReLU()
)
Тестийн нарийвчлал: 96.45%
Нийт clos параметр: 263,592
```