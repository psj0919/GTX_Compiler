#!/usr/bin/env python3
"""Prepare laya-multilingual inputs and decode raw GTX logits.

Request JSON format::

    {"state": {...}, "questions": {
      "department": {"type": "choice", "instructions": "...", "criteria": {...}}
    }}

``prepare`` writes int32 token and marker files plus ``batch.json``.  After the
C++ runner writes ``logits-XXX.bin``, ``decode`` produces Laya-compatible typed
answers.  Tokenization deliberately mirrors Laya's public Python implementation.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np


QTYPES = {"choice": 0, "score": 1, "noul": 2}
QTYPE_NAMES = {value: key for key, value in QTYPES.items()}


def _tokenizer(path: Path):
    try:
        from transformers import AutoTokenizer
    except ImportError as exc:
        raise SystemExit("Laya tokenizer dependency is missing. Run `uv sync --extra laya`.") from exc
    return AutoTokenizer.from_pretrained(path, local_files_only=True, use_fast=True)


def _serialize_state(state) -> str:
    return state if isinstance(state, str) else json.dumps(state, ensure_ascii=False)


def _criterion(value) -> str:
    if isinstance(value, str):
        return value
    return json.dumps(value, ensure_ascii=False, separators=(", ", ": "), default=str)


def _normalise_question(question: dict) -> dict:
    qtype = question.get("type", question.get("t"))
    instructions = question.get("instructions", question.get("ins"))
    criteria = question.get("criteria", question.get("crit"))
    if qtype not in QTYPES or not isinstance(instructions, str):
        raise ValueError("each question needs type=choice|score|noul and string instructions")
    if qtype == "choice" and not isinstance(criteria, dict):
        raise ValueError("choice criteria must be an ordered JSON object")
    if qtype == "score" and not isinstance(criteria, list):
        raise ValueError("score criteria must be a JSON array")
    return {"t": qtype, "ins": instructions, "crit": criteria, "labels": question.get("labels")}


def _render_options(question: dict) -> tuple[list[str], list[str]]:
    qtype, criteria = question["t"], question.get("crit")
    if qtype == "choice":
        labels = [str(key) for key in criteria]
        text = [
            str(key) if value is None or value == "" else f"{key}: {_criterion(value)}"
            for key, value in criteria.items()
        ]
        return text, labels
    if qtype == "score":
        return [f"level {i}: {_criterion(value)}" for i, value in enumerate(criteria)], [str(i) for i in range(len(criteria))]
    labels = question.get("labels") or {"false": "false", "true": "true"}
    if not isinstance(labels, dict) or set(labels) != {"false", "true"}:
        raise ValueError("noul labels must contain exactly false and true")
    criteria = criteria or {}
    false_desc = criteria.get("false")
    true_desc = criteria.get("true")
    rendered = [
        f"{labels['false']}: " + (_criterion(false_desc) if false_desc not in (None, "") else "no, the statement does not hold"),
        f"{labels['true']}: " + (_criterion(true_desc) if true_desc not in (None, "") else "yes, the statement holds"),
    ]
    return rendered, [str(labels["false"]), str(labels["true"])]


def _encode(tokenizer, text: str, **kwargs) -> list[int]:
    return list(tokenizer(text, **kwargs)["input_ids"])


def build_sequence(tokenizer, state, question: dict, max_len: int, head_max_len: int):
    mask_token = tokenizer.mask_token
    options, labels = _render_options(question)
    instructions = question["ins"].replace(mask_token, " ")
    head_ids = _encode(tokenizer, f"{question['t']} question: {instructions}", add_special_tokens=False)
    option_ids = []
    for option in options:
        encoded = _encode(
            tokenizer,
            " " + option.replace(mask_token, " "),
            add_special_tokens=False,
            truncation=True,
            max_length=48,
        )
        option_ids.append([tokenizer.mask_token_id] + encoded)
    option_budget = head_max_len - sum(len(item) for item in option_ids)
    if option_budget < 16:
        per_option = max(4, (head_max_len - 16) // max(1, len(option_ids)))
        option_ids = [item[:per_option] for item in option_ids]
        option_budget = head_max_len - sum(len(item) for item in option_ids)
    head_ids = head_ids[: max(8, option_budget)]

    ids = [tokenizer.cls_token_id] + head_ids + [tokenizer.sep_token_id]
    markers = []
    for option in option_ids:
        markers.append(len(ids))
        ids.extend(option)
    ids.append(tokenizer.sep_token_id)
    room = max(0, max_len - len(ids) - 1)
    state_ids = _encode(
        tokenizer,
        _serialize_state(state).replace(mask_token, " "),
        add_special_tokens=False,
    )
    ids.extend(state_ids[:room])
    ids.append(tokenizer.sep_token_id)
    ids = ids[:max_len]
    markers = [marker for marker in markers if marker < len(ids)]
    if len(markers) != len(options):
        raise ValueError("token budget removed one or more option markers")
    return ids, markers, labels


def prepare(model_dir: Path, request_path: Path, output_dir: Path) -> Path:
    with request_path.open(encoding="utf-8") as f:
        request = json.load(f)
    with (model_dir / "rl_agent_config.json").open(encoding="utf-8") as f:
        config = json.load(f)
    questions = request.get("questions")
    if not isinstance(questions, dict) or not questions:
        raise ValueError("request.questions must be a non-empty JSON object")

    tokenizer = _tokenizer(model_dir / "tokenizer")
    max_len = int(request.get("max_len", config.get("max_len", 1024)))
    head_max_len = int(request.get("head_max_len", config.get("head_max_len", 256)))
    if max_len > 1024:
        raise ValueError("the initial GTX Laya path supports max_len <= 1024")

    output_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for index, (qid, raw_question) in enumerate(questions.items()):
        question = _normalise_question(raw_question)
        ids, markers, labels = build_sequence(tokenizer, request.get("state"), question, max_len, head_max_len)
        if len(markers) > 20:
            raise ValueError("the initial GTX Laya path supports at most 20 options per question")
        token_file = f"tokens-{index:03d}.bin"
        marker_file = f"markers-{index:03d}.bin"
        np.asarray(ids, dtype=np.int32).tofile(output_dir / token_file)
        np.asarray(markers, dtype=np.int32).tofile(output_dir / marker_file)
        rows.append(
            {
                "index": index,
                "id": qid,
                "type": question["t"],
                "qtype": QTYPES[question["t"]],
                "labels": labels,
                "criteria": question.get("crit"),
                "tokens": token_file,
                "token_count": len(ids),
                "markers": marker_file,
                "marker_count": len(markers),
                "logits": f"logits-{index:03d}.bin",
            }
        )
    batch = {
        "model_dir": str(model_dir.resolve()),
        "max_len": max_len,
        "head_max_len": head_max_len,
        "temperature": config.get("temperature", [1.0, 1.0, 1.0]),
        "temperature_by_options": config.get("temperature_by_options", {}),
        "rows": rows,
    }
    batch_path = output_dir / "batch.json"
    with batch_path.open("w", encoding="utf-8") as f:
        json.dump(batch, f, ensure_ascii=False, indent=2)
        f.write("\n")
    print(batch_path)
    return batch_path


def _temperature_bucket(qtype: str, count: int) -> str:
    size = "2" if count <= 2 else "3-5" if count <= 5 else "6-10" if count <= 10 else "11+"
    return f"{qtype}:{size}"


def _confidence(probabilities: np.ndarray) -> float:
    if probabilities.size < 2:
        return 1.0
    entropy = -(probabilities * np.log(np.clip(probabilities, 1e-12, 1.0))).sum()
    return float(np.clip(1.0 - entropy / math.log(probabilities.size), 0.0, 1.0))


def decode(batch_path: Path, output_path: Path | None = None) -> dict:
    with batch_path.open(encoding="utf-8") as f:
        batch = json.load(f)
    root = batch_path.parent
    answers = {}
    for row in batch["rows"]:
        logits = np.fromfile(root / row["logits"], dtype=np.float32)
        count = int(row["marker_count"])
        if logits.size != count:
            raise ValueError(f"{row['logits']}: expected {count} logits, found {logits.size}")
        qtype = row["type"]
        base_temperature = batch["temperature"][QTYPES[qtype]]
        temperature = batch.get("temperature_by_options", {}).get(
            _temperature_bucket(qtype, count), base_temperature
        )
        temperature = min(5.0, max(0.5, float(temperature)))
        scaled = logits / temperature
        probabilities = np.exp(scaled - scaled.max())
        probabilities /= probabilities.sum()
        answer_confidence = round(float(probabilities.max()), 4)

        if qtype == "choice":
            labels = row["labels"]
            answer = {
                "type": "choice",
                "choice": labels[int(probabilities.argmax())],
                "probabilities": {label: round(float(value), 4) for label, value in zip(labels, probabilities)},
                "confidence": round(_confidence(probabilities), 4),
                "answer_confidence": answer_confidence,
            }
        elif qtype == "score":
            answer = {
                "type": "score",
                "score": round(float((np.arange(count) * probabilities).sum()), 4),
                "legend": {str(i): value for i, value in enumerate(row["criteria"])},
                "probabilities": {str(i): round(float(value), 4) for i, value in enumerate(probabilities)},
                "confidence": round(_confidence(probabilities), 4),
                "answer_confidence": answer_confidence,
            }
        else:
            answer = {
                "type": "noul",
                "noul": round(float(probabilities[1]), 4),
                "confidence": answer_confidence,
                "answer_confidence": answer_confidence,
            }
        answers[row["id"]] = answer
    result = {"answers": answers, "routing": {"model": "multilingual", "reason": "pinned GTX checkpoint"}}
    text = json.dumps(result, ensure_ascii=False, indent=2) + "\n"
    if output_path:
        output_path.write_text(text, encoding="utf-8")
    else:
        print(text, end="")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    prep = sub.add_parser("prepare")
    prep.add_argument("--model-dir", required=True, type=Path)
    prep.add_argument("--request", required=True, type=Path)
    prep.add_argument("--output", required=True, type=Path)
    dec = sub.add_parser("decode")
    dec.add_argument("--batch", required=True, type=Path)
    dec.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.command == "prepare":
        prepare(args.model_dir, args.request, args.output)
    else:
        decode(args.batch, args.output)


if __name__ == "__main__":
    main()
