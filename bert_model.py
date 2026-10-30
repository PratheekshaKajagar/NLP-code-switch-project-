"""
bert_model.py
-------------
BERT fine-tuning for token-level language identification (KN / HI / EN).

Default checkpoint: bert-base-multilingual-cased (mBERT) - it has seen
Kannada, Hindi and English.  Alternatives that usually do better on
romanized Indian text:  google/muril-base-cased , ai4bharat/IndicBERTv2-MLM-only
(pass with --model).

Needs internet on first run to download weights:
    pip install torch transformers
    python bert_model.py                     # ~3-6 min on CPU for this dataset
    python bert_model.py --epochs 5 --model google/muril-base-cased

Uses exactly the same train/test split as train_models.py, so results are
directly comparable.  Writes results/metrics_bert.json, confusion matrix,
ROC curve and results/bert_test_probs.npz; re-run train_models.py
afterwards to get BERT inside the combined comparison table/plots.

--smoke-test builds a tiny randomly-initialised BERT (no download) just to
verify the code path end-to-end; its numbers are meaningless.
"""

import argparse
import os

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from evaluation import CLASSES, evaluate
from train_models import load_data

SEED = 42
L2I = {c: i for i, c in enumerate(CLASSES)}


class TokDataset(Dataset):
    def __init__(self, tokens, labels, tok, max_len=16):
        self.enc = tok(list(tokens), padding="max_length", truncation=True,
                       max_length=max_len, return_tensors="pt")
        self.y = torch.tensor([L2I[l] for l in labels])

    def __len__(self):
        return len(self.y)

    def __getitem__(self, i):
        return {k: v[i] for k, v in self.enc.items()}, self.y[i]


def predict(model, loader, device):
    model.eval()
    probs = []
    with torch.no_grad():
        for batch, _ in loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            probs.append(torch.softmax(model(**batch).logits, dim=-1).cpu().numpy())
    return np.concatenate(probs)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/token_lid_dataset.csv")
    ap.add_argument("--out", default="results")
    ap.add_argument("--model", default="bert-base-multilingual-cased")
    ap.add_argument("--epochs", type=int, default=4)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=3e-5)
    ap.add_argument("--smoke-test", action="store_true")
    args = ap.parse_args()

    from transformers import (AutoModelForSequenceClassification, AutoTokenizer,
                              BertConfig, BertForSequenceClassification,
                              BertTokenizerFast, get_linear_schedule_with_warmup)

    torch.manual_seed(SEED); np.random.seed(SEED)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    df = load_data(args.data)
    tr, te = df[df.split == "train"], df[df.split == "test"]

    if args.smoke_test:
        import tempfile
        vocab = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"] + list("abcdefghijklmnopqrstuvwxyz")
        vf = os.path.join(tempfile.mkdtemp(), "vocab.txt")
        open(vf, "w").write("\n".join(vocab))
        tok = BertTokenizerFast(vf)
        cfg = BertConfig(vocab_size=len(vocab), hidden_size=64, num_hidden_layers=2,
                         num_attention_heads=2, intermediate_size=128, num_labels=3)
        model = BertForSequenceClassification(cfg)
        name = "BERT"
    else:
        tok = AutoTokenizer.from_pretrained(args.model)
        model = AutoModelForSequenceClassification.from_pretrained(args.model, num_labels=3)
        name = "BERT"
    model.to(device)

    # romanised tokens split into many word-pieces -> allow a few more positions
    tr_ds = TokDataset(tr.token, tr.label, tok)
    te_ds = TokDataset(te.token, te.label, tok)
    tr_dl = DataLoader(tr_ds, batch_size=args.batch_size, shuffle=True)
    te_dl = DataLoader(te_ds, batch_size=128)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    steps = len(tr_dl) * args.epochs
    sched = get_linear_schedule_with_warmup(opt, int(0.1 * steps), steps)

    for ep in range(1, args.epochs + 1):
        model.train()
        total = 0.0
        for batch, y in tr_dl:
            batch = {k: v.to(device) for k, v in batch.items()}
            loss = model(**batch, labels=y.to(device)).loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); sched.step(); opt.zero_grad()
            total += loss.item()
        print(f"epoch {ep}/{args.epochs}  train loss {total / len(tr_dl):.4f}")

    proba = predict(model, te_dl, device)
    pred = np.array(CLASSES)[proba.argmax(axis=1)]
    os.makedirs(args.out, exist_ok=True)
    m, report, cm = evaluate(name, te.label.values, pred, proba, args.out)
    print(report); print(cm)
    np.savez(os.path.join(args.out, "bert_test_probs.npz"),
             y_true=te.label.values, proba=proba)
    print("Saved BERT results. Re-run `python train_models.py` to merge them "
          "into the comparison table and overlay ROC plot.")


if __name__ == "__main__":
    main()
