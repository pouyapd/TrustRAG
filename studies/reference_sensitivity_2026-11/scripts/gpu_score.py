#!/usr/bin/env python
"""GPU (Colab) scorer for the larger localisers; mirrors scripts/add_model.py exactly in protocol.

Input: the private text exports written by `colbert_bridge.py export` (one JSON line per question:
qid, doc, question text, and the gold document's chunk texts in stored order). These contain dataset
text: they are uploaded only to the author's own Colab runtime and never published.

Output: <out>/<corpus>.<model>.orders.json  ({qid: full within-document ordering}) and
        <out>/<corpus>.<model>.meta.json    (model id, resolved HF commit, dtype, GPU, package
                                             versions, runtime, peak GPU memory)
Checkpoints every 100 questions; rerunning resumes.

Protocol, identical to add_model.py:
  * bi-encoders: the model's own documented prompts (SentenceTransformer.prompts "query" and
    "document"/"passage"), L2-normalised embeddings, cosine = dot product;
  * cross-encoders: (question, chunk) pairs, max_length 512 (the scored files' ce_max_length);
  * ordering: numpy argsort of the negated float64 scores, kind="stable" (ties keep document order).
Qwen3-Reranker is an LLM reranker scored with the model card's formula: P("yes") from the
next-token logits of "yes" vs "no" after the official prompt, with the card's default instruction.

    python gpu_score.py --texts qasper_full.texts.jsonl.gz --corpus qasper_full --model bge_m3 \
        [--dtype fp16|fp32] [--only-ids ids.json] [--tag pilot]
"""
from __future__ import annotations

import argparse
import gzip
import json
import os
import platform
import time
from pathlib import Path

import numpy as np
import torch

MODELS = {
    "bge_m3": ("dense", "BAAI/bge-m3", "BGE-M3 dense, Jan 2024, 568M"),
    "qwen3_emb_06b": ("dense", "Qwen/Qwen3-Embedding-0.6B", "Qwen3 Embedding 0.6B, Jun 2025"),
    "bge_reranker_v2_m3": ("cross", "BAAI/bge-reranker-v2-m3", "BGE reranker v2 m3, Mar 2024, 568M"),
    "qwen3_rerank_06b": ("llm_rerank", "Qwen/Qwen3-Reranker-0.6B", "Qwen3 Reranker 0.6B, Jun 2025"),
}
QWEN_INSTRUCT = "Given a web search query, retrieve relevant passages that answer the query"
QWEN_PREFIX = ("<|im_start|>system\nJudge whether the Document meets the requirements based on the Query "
               "and the Instruct provided. Note that the answer can only be \"yes\" or \"no\".<|im_end|>\n"
               "<|im_start|>user\n")
QWEN_SUFFIX = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"


def atomic_write(p: Path, obj) -> None:
    tmp = Path(str(p) + ".tmp")
    tmp.write_text(json.dumps(obj))
    os.replace(tmp, p)


def resolved_revision(repo_id: str) -> str | None:
    try:
        from huggingface_hub import model_info
        return model_info(repo_id).sha
    except Exception as e:  # noqa: BLE001
        return f"unresolved: {e.__class__.__name__}"


class QwenReranker:
    def __init__(self, repo_id: str, dtype, max_length: int = 1024):
        from transformers import AutoModelForCausalLM, AutoTokenizer
        self.tok = AutoTokenizer.from_pretrained(repo_id, padding_side="left")
        self.model = AutoModelForCausalLM.from_pretrained(repo_id, torch_dtype=dtype).cuda().eval()
        self.yes = self.tok.convert_tokens_to_ids("yes")
        self.no = self.tok.convert_tokens_to_ids("no")
        self.pre = self.tok.encode(QWEN_PREFIX, add_special_tokens=False)
        self.suf = self.tok.encode(QWEN_SUFFIX, add_special_tokens=False)
        self.max_length = max_length

    @torch.no_grad()
    def predict(self, pairs, batch_size=16):
        out = []
        for i in range(0, len(pairs), batch_size):
            texts = [f"<Instruct>: {QWEN_INSTRUCT}\n<Query>: {q}\n<Document>: {d}" for q, d in pairs[i:i + batch_size]]
            enc = self.tok(texts, padding=False, truncation="longest_first", return_attention_mask=False,
                           max_length=self.max_length - len(self.pre) - len(self.suf), add_special_tokens=False)
            enc["input_ids"] = [self.pre + ids + self.suf for ids in enc["input_ids"]]
            enc = self.tok.pad(enc, padding=True, return_tensors="pt").to("cuda")
            logits = self.model(**enc).logits[:, -1, :]
            two = torch.stack([logits[:, self.no], logits[:, self.yes]], dim=1).float()
            out.extend(torch.log_softmax(two, dim=1)[:, 1].exp().cpu().tolist())
        return np.asarray(out, dtype=np.float64)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--texts", required=True)
    ap.add_argument("--corpus", required=True)
    ap.add_argument("--model", required=True, choices=sorted(MODELS))
    ap.add_argument("--dtype", default="fp16", choices=["fp16", "fp32"])
    ap.add_argument("--only-ids", default=None)
    ap.add_argument("--tag", default="")
    ap.add_argument("--out", default="out")
    ap.add_argument("--batch", type=int, default=32)
    args = ap.parse_args()
    kind, repo_id, note = MODELS[args.model]
    dtype = torch.float16 if args.dtype == "fp16" else torch.float32
    out = Path(args.out); out.mkdir(exist_ok=True)
    stem = f"{args.corpus}.{args.model}{'.' + args.tag if args.tag else ''}"
    op, mp = out / f"{stem}.orders.json", out / f"{stem}.meta.json"
    orders = json.loads(op.read_text()) if op.exists() else {}
    rows = [json.loads(l) for l in gzip.open(args.texts, "rt", encoding="utf-8")]
    if args.only_ids:
        keep = set(json.loads(Path(args.only_ids).read_text()))
        rows = [r for r in rows if r["qid"] in keep]
    todo = [r for r in rows if r["qid"] not in orders]
    print(f"{len(todo)}/{len(rows)} questions to score with {repo_id} ({args.dtype})", flush=True)

    qprompt = dprompt = ""
    if kind == "dense":
        from sentence_transformers import SentenceTransformer
        enc = SentenceTransformer(repo_id, device="cuda", model_kwargs={"torch_dtype": dtype})
        qprompt = (enc.prompts or {}).get("query", "")
        dprompt = (enc.prompts or {}).get("document", (enc.prompts or {}).get("passage", ""))
    elif kind == "cross":
        from sentence_transformers import CrossEncoder
        enc = CrossEncoder(repo_id, max_length=512, device="cuda", model_kwargs={"torch_dtype": dtype})
    else:
        enc = QwenReranker(repo_id, dtype)
    print(f"query prompt {qprompt!r}; document prompt {dprompt!r}", flush=True)

    torch.cuda.reset_peak_memory_stats()
    t0, done, cache = time.time(), 0, {}
    for r in todo:
        if kind == "dense":
            if r["doc"] not in cache:
                cache = {r["doc"]: enc.encode([dprompt + c for c in r["chunks"]], batch_size=args.batch,
                                              normalize_embeddings=True, show_progress_bar=False,
                                              convert_to_numpy=True).astype(np.float64)}
            qv = enc.encode([qprompt + r["q"]], normalize_embeddings=True, show_progress_bar=False,
                            convert_to_numpy=True)[0].astype(np.float64)
            scores = cache[r["doc"]] @ qv
        else:
            pairs = [(r["q"], c) for c in r["chunks"]]
            scores = np.asarray(enc.predict(pairs, batch_size=args.batch), dtype=np.float64).reshape(-1)
        assert scores.shape[0] == len(r["chunks"]) and np.isfinite(scores).all(), r["qid"]
        orders[r["qid"]] = [int(i) for i in np.argsort(-scores, kind="stable")]
        done += 1
        if done % 100 == 0:
            atomic_write(op, orders)
            print(f"  {done}/{len(todo)} {time.time() - t0:.0f}s", flush=True)
    atomic_write(op, orders)
    elapsed = time.time() - t0
    import sentence_transformers
    import transformers
    meta = {"model": args.model, "identity": repo_id, "kind": kind, "note": note, "dtype": args.dtype,
            "resolved_revision": resolved_revision(repo_id), "query_prompt": qprompt,
            "document_prompt": dprompt, "ce_max_length": 512 if kind == "cross" else None,
            "llm_rerank_max_length": 1024 if kind == "llm_rerank" else None,
            "llm_rerank_instruction": QWEN_INSTRUCT if kind == "llm_rerank" else None,
            "questions": len(orders), "scored_this_run": done, "seconds_this_run": round(elapsed, 1),
            "peak_gpu_mem_gb": round(torch.cuda.max_memory_allocated() / 2**30, 2),
            "gpu": torch.cuda.get_device_name(0), "torch": torch.__version__,
            "transformers": transformers.__version__,
            "sentence_transformers": sentence_transformers.__version__,
            "python": platform.python_version(), "texts": Path(args.texts).name,
            "only_ids": Path(args.only_ids).name if args.only_ids else None}
    prev = json.loads(mp.read_text()) if mp.exists() else {}
    meta["seconds_total"] = round(prev.get("seconds_total", 0) + elapsed, 1)
    atomic_write(mp, meta)
    print("DONE", json.dumps(meta), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
