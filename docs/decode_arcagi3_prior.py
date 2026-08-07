#!/usr/bin/env python3
import json
import sys
import re
from pathlib import Path
from collections import defaultdict

ROOTS = []
if len(sys.argv) > 1:
    ROOTS = [Path(p).expanduser() for p in sys.argv[1:]]
else:
    ROOTS = [
        Path.home() / "glyphmatics-vil-raw/artifacts/arcagi3_top5_known_prior",
        Path.home() / "glyphmatics-vil-raw/docs",
        Path.home() / "glyphmatics-vil-raw",
        Path.cwd(),
    ]

def iter_files(root):
    if root.is_file():
        yield root
    elif root.is_dir():
        for p in root.rglob("*"):
            if p.is_file() and p.suffix.lower() in {
                ".md", ".json", ".jsonl", ".txt", ".py", ".log"
            }:
                yield p

def extract_json_objects(text):
    """
    Robustly extract {...} JSON objects from dirty markdown/log text.
    Handles:
      - JSONL
      - multiple JSON objects on one line
      - prompt characters
      - markdown around JSON
    """
    dec = json.JSONDecoder()
    out = []
    i = 0
    n = len(text)

    while i < n:
        j = text.find("{", i)
        if j < 0:
            break
        try:
            obj, end = dec.raw_decode(text[j:])
            if isinstance(obj, dict):
                out.append(obj)
            i = j + max(end, 1)
        except Exception:
            i = j + 1
    return out

rows = []
seen = set()

for root in ROOTS:
    for f in iter_files(root):
        try:
            txt = f.read_text(errors="ignore")
        except Exception:
            continue

        if "value_sha256" not in txt and '"glyph"' not in txt and "repeated_code_line" not in txt:
            continue

        for obj in extract_json_objects(txt):
            if not isinstance(obj, dict):
                continue
            if "value" not in obj:
                continue
            if "glyph" not in obj and "value_sha256" not in obj:
                continue

            key = (
                obj.get("glyph"),
                obj.get("value"),
                obj.get("value_sha256"),
                str(f),
            )
            if key in seen:
                continue
            seen.add(key)

            obj["_source_file"] = str(f)
            rows.append(obj)

buckets = {
    "world_model": [
        "WorldModel", "ActionEffect", "observe", "action_scores",
        "action_counts", "causes_change", "mover_colors",
        "infer_player_and_goal", "extract_entities"
    ],
    "goal_reader": [
        "GoalReader", "win_comparisons", "goal_field",
        "goal_value", "state_fields", "win_method_src",
        "level-complete", "complete method", "win condition"
    ],
    "bfs": [
        "BFS", "BFSSolver", "heapq", "solve_level",
        "deepcopy", "_bfs", "pre-solve", "frontier", "pq"
    ],
    "frame_diff": [
        "FrameDiff", "target_frame", "best_frame",
        "minority", "hist", "pixels_changed",
        "frame_diff", "changed_pixels"
    ],
    "click_action6": [
        "ACTION6", "action_id == 6", "click", "click_targets",
        "{'x'", "{'y'", "\"x\"", "\"y\"", "entity centres"
    ],
    "cnn_fallback": [
        "ForgeNet", "torch", "relu", "logits",
        "binary_cross_entropy", "CNN", "c1", "c2", "c3",
        "compute_loss", "Unsloth"
    ],
    "arc_grid": [
        "queries", "replies", "formatter", "augment",
        "synthesize", "convert_grid", "selection_algorithm",
        "ArcDecoder", "train_pairs", "submission"
    ],
}

classified = defaultdict(list)

for r in rows:
    value = str(r.get("value", ""))
    glyph = str(r.get("glyph", ""))
    count = r.get("count", 1)
    src = r.get("_source_file", "")

    matched = False
    for bucket, keys in buckets.items():
        if any(k.lower() in value.lower() for k in keys):
            classified[bucket].append((glyph, count, value, src))
            matched = True

    if not matched:
        classified["other"].append((glyph, count, value, src))

print("=== ARCAGI3 PRIOR DECODE ===")
print(f"Roots scanned: {len(ROOTS)}")
for r in ROOTS:
    print(f" - {r}")
print(f"Rows decoded: {len(rows)}")
print()

print("=== SOURCE FILES HIT ===")
src_counts = defaultdict(int)
for r in rows:
    src_counts[r["_source_file"]] += 1
for src, c in sorted(src_counts.items(), key=lambda x: -x[1])[:20]:
    print(f"{c:5} :: {src}")

for bucket in [
    "world_model",
    "goal_reader",
    "bfs",
    "frame_diff",
    "click_action6",
    "cnn_fallback",
    "arc_grid",
    "other",
]:
    items = classified.get(bucket, [])
    print(f"\n## {bucket.upper()} ({len(items)} lines)")
    for glyph, count, value, src in items[:80]:
        print(f"{glyph:>4} x{count:<2} :: {value}")

# Save clean extracted JSONL for later patching.
out_path = Path.home() / "glyphmatics-vil-raw/docs/arcagi3_prior_decoded.jsonl"
with out_path.open("w") as f:
    for r in rows:
        f.write(json.dumps(r, ensure_ascii=False) + "\n")

print()
print(f"[OK] wrote clean JSONL: {out_path}")
