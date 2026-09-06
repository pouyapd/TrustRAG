"""A local annotation interface for labelling the taxonomy-validation units.

This is the tool a human annotator actually uses. It exists because the
alternative — hand-editing a 200-line JSONL file with 3 KB of retrieved context
per unit in a text editor — is slow, and worse, it is error-prone in ways that
silently corrupt the study: a mistyped label, a duplicated id, a unit skipped
because it was hard.

    python scripts/annotate.py --annotator a            # open the interface
    python scripts/annotate.py --annotator a --validate # check the finished file

**The blinding is enforced here, not merely intended.** The whole point of the
validation is that the human label is formed independently of the system's
proposed label. This module therefore never opens `proposed_labels_key.jsonl`,
refuses to load any file whose name resembles it, and builds every payload sent
to the browser from an explicit allowlist of fields — so even if the sheet were
regenerated with extra keys, nothing beyond the allowlist could reach the page.
`tests/test_annotation_tool.py` pins all three properties.

Nothing here proposes, suggests, defaults or infers a label. No model is
called. An empty label stays empty until a person chooses one.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import threading
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from src.evaluation.taxonomy import FailureModeV2  # noqa: E402

#: The label set, taken from the taxonomy rather than restated. A second copy
#: would eventually drift from the one `score_annotations.py` validates against,
#: and the annotator would only find out after labelling 200 units.
ALLOWED_LABELS = [str(mode) for mode in FailureModeV2]
ALLOWED_CONFIDENCE = ["high", "medium", "low"]

#: Never read. Named only so the guard below can refuse it.
WITHHELD_FILENAME = "proposed_labels_key.jsonl"

#: Exactly what the annotator may see. Anything else in the sheet is dropped
#: before it reaches the browser.
VISIBLE_FIELDS = (
    "annotation_id",
    "question",
    "reference_answers",
    "corpus_can_answer",
    "gold_evidence",
    "retrieved_context",
    "system_answer",
)

#: What the annotator supplies.
ANNOTATION_FIELDS = ("human_label", "human_confidence", "human_notes")

#: How the labels are laid out on screen. The taxonomy's own order groups by
#: internal attribution stage, which is not the order a person decides in.
#: These groups follow the decision procedure in docs/ANNOTATION_GUIDELINES.md —
#: retrieval first, then answer quality, with the unanswerable-only pair kept
#: apart so it is not reached by accident on an answerable question. Display
#: only: `ALLOWED_LABELS` remains the thing every label is validated against.
LABEL_GROUPS = [
    ("Step 2 — did the evidence reach the system?",
     ["no_retrieval", "wrong_retrieval"]),
    ("Step 3 — the evidence was there; was the answer right?",
     ["ok", "partial_answer", "incorrect_answer", "refusal_when_answerable",
      "hallucination"]),
    ("Step 1 — only when corpus_can_answer is false",
     ["ok_abstained", "answered_when_unanswerable"]),
]


class BlindingError(RuntimeError):
    """Raised when something would expose the withheld labels."""


def guard_path(path: Path) -> Path:
    """Refuse to read the withheld key, whatever route led here."""
    if WITHHELD_FILENAME in path.name:
        raise BlindingError(
            f"refusing to read {path.name}: the proposed labels must stay withheld "
            "until both annotators have finished"
        )
    return path


def read_jsonl(path: Path) -> list[dict]:
    guard_path(path)
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def text_completeness(item: dict) -> str:
    """Is this chunk/span the whole text its char_range claims?

    Returns "complete", "truncated", or "unverifiable". The sheet may state it
    outright via `text_complete`; older sheets do not, so the offsets are used
    as the fallback. Chunks without offsets are "unverifiable" rather than
    "complete" — an unmeasurable chunk is not a clean one.
    """
    stated = item.get("text_complete")
    if isinstance(stated, bool):
        return "complete" if stated else "truncated"
    span = item.get("char_range") or [None, None]
    if span[0] is None or span[1] is None:
        return "unverifiable"
    return "complete" if len(str(item.get("text", ""))) == span[1] - span[0] else "truncated"


def context_completeness(sheet: list[dict]) -> dict[str, int]:
    """Tally of complete / truncated / unverifiable retrieved chunks in a sheet."""
    tally = {"complete": 0, "truncated": 0, "unverifiable": 0}
    for unit in sheet:
        for chunk in unit.get("retrieved_context") or []:
            tally[text_completeness(chunk)] += 1
    return tally


def visible_unit(unit: dict) -> dict:
    """One unit, reduced to the fields an annotator is allowed to see."""
    return {field: unit.get(field) for field in VISIBLE_FIELDS}


class LockedAnnotation(Exception):
    """Raised when a write would overwrite a protected existing annotation."""


class Session:
    """Holds the sheet and the annotations, and owns every write to disk."""

    def __init__(self, package: Path, annotator: str) -> None:
        self.package = package
        self.annotator = annotator
        self.dir = package / f"annotator_{annotator}"
        self.sheet_path = self.dir / "annotation_sheet.jsonl"
        self.output_path = self.dir / "completed.jsonl"
        self.integrity_path = self.dir / ".sheet_integrity.json"
        self.locked_path = self.dir / ".locked_ids.json"
        self._lock = threading.Lock()

        if not self.sheet_path.exists():
            raise FileNotFoundError(f"no annotation sheet at {self.sheet_path}")

        self.units = read_jsonl(self.sheet_path)
        self.order = [u["annotation_id"] for u in self.units]
        self.by_id = {u["annotation_id"]: u for u in self.units}
        if len(self.by_id) != len(self.units):
            raise ValueError("the annotation sheet contains duplicate annotation_ids")

        self._record_integrity()
        self.annotations = self._load_existing()
        self.locked = self._load_locked()

    def _record_integrity(self) -> None:
        """Remember the sheet's checksum the first time we see it.

        Written once and then only compared, so a later edit to the sheet is
        detectable rather than invisible.
        """
        if self.integrity_path.exists():
            return
        self.integrity_path.write_text(
            json.dumps(
                {
                    "sheet": self.sheet_path.name,
                    "sha256": sha256(self.sheet_path),
                    "n_units": len(self.units),
                },
                indent=2,
            ),
            encoding="utf-8",
        )

    def _load_existing(self) -> dict[str, dict]:
        """Resume from a partly finished file, ignoring anything unrecognised."""
        annotations: dict[str, dict] = {}
        if not self.output_path.exists():
            return annotations
        for row in read_jsonl(self.output_path):
            unit_id = row.get("annotation_id")
            if unit_id not in self.by_id:
                continue
            label = str(row.get("human_label", "")).strip()
            if not label:
                continue
            annotations[unit_id] = {
                "human_label": label,
                "human_confidence": str(row.get("human_confidence", "")).strip(),
                "human_notes": str(row.get("human_notes", "")),
            }
        return annotations

    def _load_locked(self) -> set[str]:
        """Ids carried in from an earlier pass, protected from a stray keystroke.

        A locked unit can be read and navigated to like any other; it just
        cannot be overwritten unless the caller says so explicitly. Missing
        file means nothing is locked, which is the state of a fresh package.
        """
        if not self.locked_path.exists():
            return set()
        try:
            stored = json.loads(self.locked_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            return set()
        ids = stored.get("annotation_ids", []) if isinstance(stored, dict) else stored
        return {i for i in ids if i in self.by_id and i in self.annotations}

    def _write_locked(self) -> None:
        self.locked_path.write_text(
            json.dumps(
                {
                    "note": "Annotations protected from accidental overwrite. "
                            "Editing one through the interface requires an explicit "
                            "unlock and removes it from this list.",
                    "annotation_ids": sorted(self.locked),
                },
                indent=2,
            ),
            encoding="utf-8",
        )

    def save(self, unit_id: str, label: str, confidence: str, notes: str,
             unlock: bool = False) -> dict:
        """Record one annotation and flush the whole file to disk."""
        if unit_id not in self.by_id:
            raise KeyError(f"unknown annotation_id {unit_id!r}")
        label = (label or "").strip()
        confidence = (confidence or "").strip()
        if label and label not in ALLOWED_LABELS:
            raise ValueError(f"{label!r} is not one of {ALLOWED_LABELS}")
        if confidence and confidence not in ALLOWED_CONFIDENCE:
            raise ValueError(f"{confidence!r} is not one of {ALLOWED_CONFIDENCE}")

        with self._lock:
            if unit_id in self.locked:
                if not unlock:
                    raise LockedAnnotation(
                        f"{unit_id} carries an existing annotation and is locked. "
                        "Unlock it first if you really mean to change it."
                    )
                self.locked.discard(unit_id)
                self._write_locked()
            if label:
                self.annotations[unit_id] = {
                    "human_label": label,
                    "human_confidence": confidence,
                    "human_notes": notes or "",
                }
            else:
                # An explicit clear, so a misclick can be undone.
                self.annotations.pop(unit_id, None)
            self._flush()
        return {"annotation_id": unit_id, "saved": bool(label),
                "n_done": len(self.annotations), "n_total": len(self.units),
                "locked": sorted(self.locked)}

    def _flush(self) -> None:
        """Write every unit, atomically.

        All 200 rows are written on each save, labelled or not, so the file
        always carries the complete id set and a crash cannot leave a partial
        line. `os.replace` is atomic on Windows and POSIX alike.
        """
        rows = []
        for unit_id in self.order:
            row = dict(self.by_id[unit_id])
            annotation = self.annotations.get(unit_id)
            row["human_label"] = annotation["human_label"] if annotation else ""
            row["human_confidence"] = annotation["human_confidence"] if annotation else ""
            row["human_notes"] = annotation["human_notes"] if annotation else ""
            rows.append(row)

        temp = self.output_path.with_suffix(".jsonl.tmp")
        temp.write_text(
            "\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n",
            encoding="utf-8",
        )
        os.replace(temp, self.output_path)

    def state(self) -> dict:
        """Everything the page needs — and nothing the annotator must not see."""
        return {
            "annotator": self.annotator,
            "allowed_labels": ALLOWED_LABELS,
            "label_groups": [
                {"heading": heading, "labels": labels} for heading, labels in LABEL_GROUPS
            ],
            "allowed_confidence": ALLOWED_CONFIDENCE,
            "units": [visible_unit(self.by_id[i]) for i in self.order],
            "annotations": self.annotations,
            "locked": sorted(self.locked),
            "output_path": str(self.output_path),
        }


class Handler(BaseHTTPRequestHandler):
    session: Session = None  # set on the class before serving

    def log_message(self, *args) -> None:  # noqa: A003 - quiet by default
        pass

    def _send(self, status: int, body: bytes, content_type: str) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _json(self, status: int, payload: dict) -> None:
        self._send(status, json.dumps(payload).encode("utf-8"), "application/json")

    def do_GET(self) -> None:  # noqa: N802
        if self.path in ("/", "/index.html"):
            self._send(200, PAGE.encode("utf-8"), "text/html; charset=utf-8")
        elif self.path == "/api/state":
            self._json(200, self.session.state())
        else:
            self._json(404, {"error": "not found"})

    def do_POST(self) -> None:  # noqa: N802
        if self.path != "/api/save":
            self._json(404, {"error": "not found"})
            return
        length = int(self.headers.get("Content-Length", 0))
        try:
            payload = json.loads(self.rfile.read(length) or b"{}")
            result = self.session.save(
                payload.get("annotation_id", ""),
                payload.get("human_label", ""),
                payload.get("human_confidence", ""),
                payload.get("human_notes", ""),
                unlock=bool(payload.get("unlock")),
            )
        except LockedAnnotation as exc:
            self._json(409, {"error": str(exc), "locked": True})
            return
        except (KeyError, ValueError, json.JSONDecodeError) as exc:
            self._json(400, {"error": str(exc)})
            return
        self._json(200, result)


PAGE = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>TrustRAG annotation</title>
<style>
  :root {
    --bg:#f6f7f9; --panel:#fff; --ink:#1a1c1f; --muted:#5b6270; --line:#dfe3e8;
    --accent:#2f6fd0; --good:#1d7a4c; --warn:#a8620a;
  }
  * { box-sizing:border-box; }
  body { margin:0; font:14px/1.55 -apple-system,Segoe UI,Roboto,sans-serif;
         background:var(--bg); color:var(--ink); }
  header { position:sticky; top:0; z-index:5; background:var(--panel);
           border-bottom:1px solid var(--line); padding:10px 16px;
           display:flex; align-items:center; gap:14px; flex-wrap:wrap; }
  header h1 { font-size:15px; margin:0; font-weight:650; }
  .grow { flex:1; }
  .pill { font-size:12px; padding:2px 8px; border-radius:999px;
          border:1px solid var(--line); color:var(--muted); }
  .pill.done { color:var(--good); border-color:#bfe0cd; background:#f1f9f4; }
  .pill.todo { color:var(--warn); border-color:#f0dcc0; background:#fdf7ef; }
  button { font:inherit; padding:6px 12px; border:1px solid var(--line);
           background:#fff; border-radius:7px; cursor:pointer; }
  button:hover { border-color:#b9c0ca; }
  button:disabled { opacity:.45; cursor:default; }
  main { max-width:1080px; margin:0 auto; padding:16px; }
  .card { background:var(--panel); border:1px solid var(--line); border-radius:10px;
          padding:14px 16px; margin-bottom:14px; }
  .card h2 { font-size:12px; text-transform:uppercase; letter-spacing:.06em;
             color:var(--muted); margin:0 0 8px; font-weight:650; }
  .q { font-size:17px; font-weight:600; }
  .ref li { margin-bottom:4px; }
  .chunk { border:1px solid var(--line); border-radius:8px; padding:9px 11px;
           margin-bottom:8px; background:#fcfcfd; }
  .chunk .meta { font-size:11.5px; color:var(--muted); margin-bottom:5px;
                 font-family:ui-monospace,Consolas,monospace; cursor:pointer;
                 list-style:none; }
  .chunk .meta::-webkit-details-marker { display:none; }
  .chunk .meta::before { content:'▾ '; }
  details.chunk:not([open]) .meta::before { content:'▸ '; }
  .chunk .ptext { white-space:pre-wrap; margin-top:6px; }
  .tag { font-size:10.5px; letter-spacing:.05em; padding:1px 6px; border-radius:4px;
         border:1px solid var(--line); font-family:ui-monospace,Consolas,monospace; }
  .tag.full { color:var(--good); border-color:#bfe0cd; background:#f1f9f4; }
  .tag.trunc { color:#b23b3b; border-color:#e6c2c2; background:#fdf2f2; }
  .tag.unk { color:var(--warn); border-color:#f0dcc0; background:#fdf7ef; }
  .gold { border-left:3px solid var(--good); background:#f6fbf8; }
  .answer { white-space:pre-wrap; background:#fbfaf7; border:1px solid #eee3cf;
            border-radius:8px; padding:10px 12px; }
  .grouphead { font-size:11.5px; color:var(--muted); margin:10px 0 5px; font-weight:600; }
  .grouphead:first-child { margin-top:0; }
  .labels { display:grid; grid-template-columns:repeat(auto-fill,minmax(240px,1fr));
            gap:7px; }
  .labels button { text-align:left; display:flex; gap:8px; align-items:center; }
  .labels button.sel { border-color:var(--accent); background:#eaf1fb;
                       box-shadow:inset 0 0 0 1px var(--accent); }
  kbd { font:11px ui-monospace,Consolas,monospace; border:1px solid var(--line);
        border-bottom-width:2px; border-radius:4px; padding:0 5px; color:var(--muted);
        background:#fafbfc; }
  .conf button.sel { border-color:var(--accent); background:#eaf1fb;
                     box-shadow:inset 0 0 0 1px var(--accent); }
  textarea { width:100%; min-height:64px; padding:9px 11px; border:1px solid var(--line);
             border-radius:8px; font:inherit; resize:vertical; }
  .nav { display:flex; gap:8px; align-items:center; }
  .grid { display:flex; flex-wrap:wrap; gap:3px; }
  .grid a { width:22px; height:22px; display:grid; place-items:center; font-size:10px;
            border:1px solid var(--line); border-radius:4px; text-decoration:none;
            color:var(--muted); background:#fff; cursor:pointer; }
  .grid a.done { background:#e7f5ec; border-color:#bfe0cd; color:var(--good); }
  .grid a.cur { outline:2px solid var(--accent); outline-offset:1px; }
  .grid a.locked { background:#eef1f6; border-color:#c3ccda; color:#5b6270; }
  .pill.prog { font-variant-numeric:tabular-nums; font-weight:650; }
  .locked-banner { display:flex; gap:10px; align-items:center; flex-wrap:wrap;
                   background:#eef1f6; border:1px solid #c3ccda; border-radius:8px;
                   padding:9px 11px; margin-bottom:10px; font-size:12.5px; }
  .locked-banner b { color:var(--ink); }
  .warn { color:var(--warn); font-size:12.5px; }
  .hint { color:var(--muted); font-size:12.5px; }
  .unans { background:#fdf7ef; border-color:#f0dcc0; }
</style>
</head>
<body>
<header>
  <h1>TrustRAG annotation — annotator <span id="who"></span></h1>
  <span class="pill prog" id="prog"></span>
  <span class="pill" id="pos"></span>
  <span class="pill done" id="done"></span>
  <span class="pill todo" id="todo"></span>
  <span class="grow"></span>
  <div class="nav">
    <button id="prev">← Prev</button>
    <button id="next">Next →</button>
    <button id="nextTodo">Next unlabelled</button>
  </div>
</header>
<main>
  <div class="card"><h2>Progress <span class="hint">— click a square to jump</span></h2>
    <div class="grid" id="grid"></div></div>
  <div class="card"><h2>Annotation id</h2><code id="aid"></code></div>
  <div class="card"><h2>Question</h2><div class="q" id="question"></div></div>
  <div class="card" id="canAnswerCard"><h2>Corpus can answer</h2>
    <div id="canAnswer"></div></div>
  <div class="card"><h2>Reference answers</h2><ul class="ref" id="refs"></ul></div>
  <div class="card"><h2>Gold evidence <span class="hint">— what the dataset says supports the answer</span></h2>
    <div id="gold"></div></div>
  <div class="card"><h2>Retrieved context <span class="hint">— what the system actually retrieved, in rank order</span></h2>
    <div id="ctx"></div></div>
  <div class="card"><h2>System answer</h2><div class="answer" id="ans"></div></div>

  <div class="card">
    <h2>Your label <span class="hint">— exactly one, per docs/ANNOTATION_GUIDELINES.md</span></h2>
    <div id="labels"></div>
    <div class="locked-banner" id="lockedBanner" style="display:none">
      <span>🔒 <b>Existing annotation.</b> Carried in from an earlier pass and
        protected, so a stray keystroke cannot change it. Read it freely.</span>
      <button id="unlock">Unlock to edit this one</button></div>
    <p class="warn" id="labelWarn" style="display:none">
      No label chosen yet. Nothing is saved until you pick one.</p>
  </div>
  <div class="card"><h2>Your confidence</h2>
    <div class="conf nav" id="conf"></div></div>
  <div class="card"><h2>Your notes <span class="hint">— optional; please use it whenever you hesitate</span></h2>
    <textarea id="notes" placeholder="Anything that made this hard..."></textarea></div>
  <p class="hint">Saved automatically to <code id="out"></code> on every change.
     Keys: <kbd>1</kbd>–<kbd>9</kbd> label · <kbd>h</kbd>/<kbd>m</kbd>/<kbd>l</kbd>
     confidence · <kbd>←</kbd>/<kbd>→</kbd> move · <kbd>Enter</kbd> next unlabelled.</p>
</main>
<script>
let S=null, i=0, KEYMAP=[], LOCKED=new Set();
const isLocked = id => LOCKED.has(id);
const $=id=>document.getElementById(id);
const esc=s=>String(s==null?'':s).replace(/[&<>]/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;'}[c]));

async function boot(){
  S = await (await fetch('/api/state')).json();
  LOCKED = new Set(S.locked || []);
  $('who').textContent = S.annotator.toUpperCase();
  $('out').textContent = S.output_path;
  buildLabels(); buildConf(); buildGrid();
  const firstTodo = S.units.findIndex(u=>!S.annotations[u.annotation_id]);
  i = firstTodo === -1 ? 0 : firstTodo;
  render();
}
function buildLabels(){
  let n = 0, html = '';
  for(const g of S.label_groups){
    html += `<p class="grouphead">${g.heading}</p><div class="labels">`;
    for(const l of g.labels){
      n += 1;
      html += `<button data-label="${l}" data-key="${n}">`
            + `<kbd>${n}</kbd><span>${l}</span></button>`;
    }
    html += '</div>';
  }
  $('labels').innerHTML = html;
  KEYMAP = [];
  $('labels').querySelectorAll('button').forEach(b=>{
    KEYMAP[+b.dataset.key] = b.dataset.label;
    b.onclick = ()=>setLabel(b.dataset.label);
  });
}
function buildConf(){
  $('conf').innerHTML = S.allowed_confidence.map(c=>
    `<button data-c="${c}"><kbd>${c[0]}</kbd> ${c}</button>`).join('');
  $('conf').querySelectorAll('button').forEach(b=>
    b.onclick=()=>setConf(b.dataset.c));
}
function buildGrid(){
  $('grid').innerHTML = S.units.map((u,n)=>`<a data-n="${n}">${n+1}</a>`).join('');
  $('grid').querySelectorAll('a').forEach(a=>
    a.onclick=()=>{ i=+a.dataset.n; render(); window.scrollTo(0,0); });
}
function cur(){ return S.units[i]; }
function ann(){ return S.annotations[cur().annotation_id] || null; }

// Is the stored text the whole span its char_range claims? The sheet says so
// outright when it was built by the current builder; older sheets are measured
// against their offsets instead. Never guess "complete" when it cannot be
// checked — an unmeasurable chunk is exactly the case that went unnoticed once.
function completeness(o){
  if (typeof o.text_complete === 'boolean') return o.text_complete ? 'complete' : 'truncated';
  const r = o.char_range;
  if (!r || r[0]===null || r[1]===null || r[0]===undefined || r[1]===undefined) return 'unverifiable';
  return (o.text||'').length === r[1]-r[0] ? 'complete' : 'truncated';
}
// One passage, whole. Long chunks are collapsible, never clipped: the text in
// the page is always the entire chunk, so nothing an annotator needs for step 2
// can be hidden behind an ellipsis.
function passage(o, title, extra){
  const r = o.char_range || [null,null];
  const len = (o.text||'').length;
  const state = completeness(o);
  const tag = state==='complete'
    ? '<span class="tag full">FULL</span>'
    : state==='truncated'
      ? `<span class="tag trunc">TRUNCATED — ${len} of ${r[1]-r[0]} chars shown</span>`
      : '<span class="tag unk">LENGTH UNVERIFIED</span>';
  const range = `chars [${r[0]===null||r[0]===undefined?'?':r[0]}, `
              + `${r[1]===null||r[1]===undefined?'?':r[1]})`;
  const long = len > 1400;
  return `<details class="chunk ${extra}"${long?'':' open'}>`
       + `<summary class="meta">${title} — ${tag} · ${range} · ${len} chars`
       + `${long?' · click to show the full text':''}</summary>`
       + `<div class="ptext">${esc(o.text)}</div></details>`;
}
function render(){
  const u = cur(), a = ann();
  $('pos').textContent = `unit ${i+1} of ${S.units.length}`;
  const done = Object.keys(S.annotations).length;
  $('prog').textContent = `${done} / ${S.units.length}`;
  $('done').textContent = `${done} labelled`;
  $('todo').textContent = `${S.units.length-done} remaining`;
  $('aid').textContent = u.annotation_id;
  $('question').textContent = u.question;
  $('canAnswer').innerHTML = u.corpus_can_answer
    ? 'true — the corpus is supposed to contain the answer'
    : '<b>false</b> — the corpus is NOT supposed to contain the answer. '
      + 'Per the guidelines, only <code>ok_abstained</code> or '
      + '<code>answered_when_unanswerable</code> apply.';
  $('canAnswerCard').className = 'card' + (u.corpus_can_answer ? '' : ' unans');
  $('refs').innerHTML = (u.reference_answers||[]).map(r=>`<li>${esc(r)}</li>`).join('')
    || '<li class="hint">none</li>';
  $('gold').innerHTML = (u.gold_evidence||[]).map(g=>
    passage(g, `Gold evidence — ${esc(g.doc_id)}`, 'gold')).join('')
    || '<p class="hint">none recorded</p>';
  $('ctx').innerHTML = (u.retrieved_context||[]).map(c=>
    passage(c, `Retrieved context — Rank ${c.rank} — ${esc(c.doc_id)}`, '')).join('')
    || '<p class="hint">nothing retrieved</p>';
  $('ans').textContent = u.system_answer || '(empty)';

  $('labels').querySelectorAll('button').forEach(b=>
    b.classList.toggle('sel', !!a && a.human_label===b.dataset.label));
  $('conf').querySelectorAll('button').forEach(b=>
    b.classList.toggle('sel', !!a && a.human_confidence===b.dataset.c));
  $('notes').value = a ? a.human_notes : '';
  const locked = isLocked(u.annotation_id);
  $('lockedBanner').style.display = locked ? 'flex' : 'none';
  $('notes').readOnly = locked;
  $('labelWarn').style.display = a ? 'none' : 'block';
  $('prev').disabled = i===0; $('next').disabled = i===S.units.length-1;
  $('grid').querySelectorAll('a').forEach((el,n)=>{
    el.classList.toggle('done', !!S.annotations[S.units[n].annotation_id]);
    el.classList.toggle('locked', isLocked(S.units[n].annotation_id));
    el.classList.toggle('cur', n===i);
  });
}
async function save(label, conf, notes, unlock){
  const id = cur().annotation_id;
  if(isLocked(id) && !unlock){
    alert('This unit is locked because it already carries an annotation. '
        + 'Use the "Unlock to edit this one" button if you really mean to change it.');
    return;
  }
  const body = {annotation_id: id, human_label: label,
                human_confidence: conf, human_notes: notes, unlock: !!unlock};
  const r = await fetch('/api/save', {method:'POST',
    headers:{'Content-Type':'application/json'}, body: JSON.stringify(body)});
  if(!r.ok){ alert('Not saved: ' + (await r.json()).error); return; }
  LOCKED.delete(id);
  if(label) S.annotations[cur().annotation_id] = {human_label:label,
      human_confidence:conf, human_notes:notes};
  else delete S.annotations[cur().annotation_id];
  render();
}
function setLabel(l){
  const a = ann();
  save(a && a.human_label===l ? '' : l, a?a.human_confidence:'', $('notes').value);
}
function setConf(c){
  const a = ann();
  if(!a){ alert('Choose a label first.'); return; }
  save(a.human_label, a.human_confidence===c ? '' : c, $('notes').value);
}
$('notes').addEventListener('blur', ()=>{
  const a = ann();
  if(a && a.human_notes !== $('notes').value)
    save(a.human_label, a.human_confidence, $('notes').value);
});
$('unlock').onclick=()=>{
  const a = ann(); if(!a) return;
  if(!confirm('Unlock ' + cur().annotation_id + ' so it can be edited? '
            + 'Its current label stays until you change it.')) return;
  save(a.human_label, a.human_confidence, a.human_notes, true);
};
$('prev').onclick=()=>{ if(i>0){i--; render(); window.scrollTo(0,0);} };
$('next').onclick=()=>{ if(i<S.units.length-1){i++; render(); window.scrollTo(0,0);} };
$('nextTodo').onclick=()=>{
  const n = S.units.findIndex((u,k)=>k>i && !S.annotations[u.annotation_id]);
  const m = n===-1 ? S.units.findIndex(u=>!S.annotations[u.annotation_id]) : n;
  if(m===-1){ alert('Every unit has a label.'); return; }
  i=m; render(); window.scrollTo(0,0);
};
document.addEventListener('keydown', e=>{
  if(e.target.tagName==='TEXTAREA') return;
  if(e.key>='1' && e.key<='9'){
    const l = KEYMAP[+e.key]; if(l) setLabel(l);
  } else if(e.key==='h') setConf('high');
  else if(e.key==='m') setConf('medium');
  else if(e.key==='l') setConf('low');
  else if(e.key==='ArrowLeft') $('prev').click();
  else if(e.key==='ArrowRight') $('next').click();
  else if(e.key==='Enter') $('nextTodo').click();
});
boot();
</script>
</body></html>
"""


def validate(package: Path, annotator: str, filename: str = "completed.jsonl") -> int:
    """Check the finished file. Returns 0 only if every check passes.

    `filename` names the file inside the annotator directory, so a pass kept
    under a different name can be checked without disturbing completed.jsonl.
    """
    directory = package / f"annotator_{annotator}"
    sheet_path = directory / "annotation_sheet.jsonl"
    output_path = directory / filename
    integrity_path = directory / ".sheet_integrity.json"
    master_path = package / "annotation_sheet.jsonl"

    problems: list[str] = []
    notes: list[str] = []
    warnings: list[str] = []

    if not output_path.exists():
        print(f"FAIL  no completed file at {output_path}")
        return 1

    # --- JSONL validity -----------------------------------------------------
    rows, bad_lines = [], []
    for number, line in enumerate(
        output_path.read_text(encoding="utf-8").splitlines(), start=1
    ):
        if not line.strip():
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError as exc:
            bad_lines.append(f"line {number}: {exc}")
    if bad_lines:
        problems += [f"invalid JSON — {b}" for b in bad_lines]
        print("FAIL  the file is not valid JSONL")
        for b in bad_lines[:5]:
            print(f"      {b}")
        return 1
    notes.append(f"valid JSONL, {len(rows)} rows")

    sheet = read_jsonl(sheet_path)
    expected = {u["annotation_id"] for u in sheet}
    got = [r.get("annotation_id") for r in rows]

    # --- ids ----------------------------------------------------------------
    # The expected count comes from the sheet rather than a hardcoded 200, so
    # the check is against ground truth. For this package that *is* 200.
    if len(rows) != len(sheet):
        problems.append(
            f"expected exactly {len(sheet)} units (one per sheet row), found {len(rows)}"
        )
    duplicates = sorted({i for i in got if got.count(i) > 1})
    if duplicates:
        problems.append(f"duplicate annotation_id: {duplicates[:10]}")
    missing = sorted(expected - set(got))
    if missing:
        problems.append(f"missing {len(missing)} annotation_id(s): {missing[:10]}")
    unexpected = sorted(set(got) - expected)
    if unexpected:
        problems.append(f"unknown annotation_id(s): {unexpected[:10]}")
    if not (duplicates or missing or unexpected):
        notes.append(f"{len(set(got))} unique ids, matching the sheet exactly")

    # --- labels and confidence ---------------------------------------------
    unlabelled, bad_label, bad_conf = [], [], []
    for row in rows:
        unit_id = row.get("annotation_id")
        label = str(row.get("human_label", "")).strip()
        confidence = str(row.get("human_confidence", "")).strip()
        if not label:
            unlabelled.append(unit_id)
            continue
        if label not in ALLOWED_LABELS:
            bad_label.append(f"{unit_id}: {label!r}")
        if confidence not in ALLOWED_CONFIDENCE:
            bad_conf.append(f"{unit_id}: {confidence!r}")
    if unlabelled:
        problems.append(
            f"{len(unlabelled)} unit(s) have no human_label "
            f"(first: {unlabelled[:5]}) — none may be skipped"
        )
    if bad_label:
        problems.append(f"label outside the taxonomy: {bad_label[:5]}")
    if bad_conf:
        problems.append(f"confidence not high/medium/low: {bad_conf[:5]}")
    if not (unlabelled or bad_label or bad_conf):
        notes.append("every unit has a valid label and confidence")

    # --- the sheet must be untouched ---------------------------------------
    if integrity_path.exists():
        recorded = json.loads(integrity_path.read_text(encoding="utf-8"))
        current = sha256(sheet_path)
        if recorded.get("sha256") != current:
            problems.append(
                "annotation_sheet.jsonl has changed since annotation began "
                f"(recorded {recorded.get('sha256', '')[:12]}, now {current[:12]})"
            )
        else:
            notes.append("annotation_sheet.jsonl unchanged (sha256 matches)")
    else:
        notes.append("no recorded checksum — sheet integrity checked against the master only")

    if master_path.exists():
        master = {u["annotation_id"]: u for u in read_jsonl(master_path)}
        drifted = [
            u["annotation_id"] for u in sheet
            if u["annotation_id"] in master
            and visible_unit(u) != visible_unit(master[u["annotation_id"]])
        ]
        if drifted:
            problems.append(
                f"{len(drifted)} unit(s) differ from the master sheet: {drifted[:5]}"
            )
        else:
            notes.append("every unit matches the master annotation_sheet.jsonl")

    # --- the content the annotator saw must be preserved --------------------
    by_id = {u["annotation_id"]: u for u in sheet}
    altered = [
        r["annotation_id"] for r in rows
        if r.get("annotation_id") in by_id
        and visible_unit(r) != visible_unit(by_id[r["annotation_id"]])
    ]
    if altered:
        problems.append(f"{len(altered)} completed row(s) altered the unit content: {altered[:5]}")
    else:
        notes.append("completed rows preserve the original unit content")

    # --- the annotator must have seen the whole retrieved chunk -------------
    # A warning, not a failure: a package built before this check existed is
    # still scorable, and its labels are still the labels that were given. What
    # is not acceptable is leaving the reader to guess, so the count is stated
    # every time. Step 2 of the guidelines asks whether the evidence reached the
    # system; a prefix of a chunk cannot answer that.
    context = context_completeness(sheet)
    if context["truncated"] or context["unverifiable"]:
        warnings.append(
            f"{context['truncated']} retrieved chunk(s) hold less text than their "
            f"char_range covers and {context['unverifiable']} cannot be checked "
            f"(complete: {context['complete']}). Step-2 judgements on those units "
            "rest on an excerpt; rebuild the package with the current builder."
        )
    else:
        notes.append(
            f"retrieved context is complete for all {context['complete']} chunk(s)"
        )

    # --- report -------------------------------------------------------------
    print(f"validating {output_path}")
    for note in notes:
        print(f"  ok    {note}")
    for warning in warnings:
        print(f"  WARN  {warning}")
    for problem in problems:
        print(f"  FAIL  {problem}")
    if problems:
        print(f"\n{len(problems)} problem(s). The file is not ready for scoring.")
        return 1

    labels = {}
    for row in rows:
        labels[row["human_label"]] = labels.get(row["human_label"], 0) + 1
    print("\nlabel distribution:")
    for label, count in sorted(labels.items(), key=lambda kv: -kv[1]):
        print(f"  {label:28s} {count:>4}")
    print("\nAll checks passed. This file is ready for scoring once annotator B "
          "has independently finished.")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Local annotation interface for the taxonomy validation study"
    )
    parser.add_argument("--annotator", default="a", help="annotator id, e.g. a or b")
    parser.add_argument(
        "--package", default="reports/annotation/qasper_dev_300",
        help="annotation package directory",
    )
    parser.add_argument("--port", type=int, default=8900)
    parser.add_argument("--validate", action="store_true",
                        help="check the completed file instead of serving the interface")
    parser.add_argument("--file", default="completed.jsonl",
                        help="file inside the annotator directory to validate")
    parser.add_argument("--no-browser", action="store_true",
                        help="do not open a browser window")
    args = parser.parse_args()

    package = Path(args.package)
    if not package.is_absolute():
        package = REPO / package
    if not package.exists():
        print(f"no annotation package at {package}", file=sys.stderr)
        return 1

    if args.validate:
        return validate(package, args.annotator, args.file)

    try:
        session = Session(package, args.annotator)
    except (FileNotFoundError, ValueError, BlindingError) as exc:
        print(str(exc), file=sys.stderr)
        return 1

    Handler.session = session
    server = ThreadingHTTPServer(("127.0.0.1", args.port), Handler)
    url = f"http://127.0.0.1:{args.port}/"
    done = len(session.annotations)

    print(f"TrustRAG annotation — annotator {args.annotator.upper()}")
    print(f"  units      : {len(session.units)}  ({done} already labelled, "
          f"{len(session.units) - done} remaining)")
    print(f"  sheet      : {session.sheet_path}")
    print(f"  saving to  : {session.output_path}")
    print("  guidelines : docs/ANNOTATION_GUIDELINES.md")
    print(f"\n  open {url}  (Ctrl+C here when you are done)\n")
    print("  The proposed labels are not loaded and cannot be shown.")

    if not args.no_browser:
        threading.Timer(0.5, lambda: webbrowser.open(url)).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nstopped. Progress is saved; rerun the same command to continue.")
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
