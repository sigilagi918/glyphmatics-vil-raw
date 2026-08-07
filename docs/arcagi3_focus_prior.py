#!/usr/bin/env python3
import json
import re
from pathlib import Path
from collections import defaultdict

INP = Path.home() / "glyphmatics-vil-raw/docs/arcagi3_prior_decoded.jsonl"
OUT_JSONL = Path.home() / "glyphmatics-vil-raw/docs/arcagi3_prior_focused.jsonl"
OUT_MD = Path.home() / "glyphmatics-vil-raw/docs/arcagi3_prior_focused.md"

BUCKETS = {
    "world_model": [
        "WorldModel", "ActionEffect", "observe", "action_scores", "action_counts",
        "causes_change", "mover_colors", "infer_player_and_goal", "extract_entities",
        "changed_pixels", "mean_dx", "mean_dy"
    ],
    "goal_reader": [
        "GoalReader", "win_comparisons", "goal_field", "goal_value", "state_fields",
        "win_method_src", "win condition", "level-complete", "best_action_for_level"
    ],
    "bfs": [
        "BFSSolver", "BFS", "solve_level", "heapq", "frontier", "deepcopy(game)",
        "_bfs_solution", "_bfs_pre_solutions", "pre-solve"
    ],
    "frame_diff": [
        "FrameDiffModel", "frame_diff", "target_frame", "best_frame", "best_score",
        "minority", "hist_diff", "pixels_changed", "changed_pixels"
    ],
    "click_action6": [
        "ACTION6", "action_id == 6", "click_targets", "entity centres",
        "{'x'", "{'y'", "\"x\"", "\"y\""
    ],
    "cnn_fallback": [
        "ForgeNet", "CNN fallback", "torch", "relu", "logits",
        "binary_cross_entropy", "s.buf.append", "s.device", "cc1", "cc2"
    ],
    "arc_grid": [
        "synthesize", "ArcDecoder", "QwenFormatter", "augment", "selection_algorithm",
        "convert_grid_to_string", "split_multi_replies", "submission"
    ],
}

REJECT = [
    "ggml_", "ggml tensor", "tensor_extra_cl", "src1->", "dst->", "OpenCL",
    "const ggml", "uint32_t", "base64", "TX5D8Q", "L/tr", "/bW/9tf",
]

def is_junk(v: str) -> bool:
    if not v:
        return True
    if len(v) > 700:
        return True
    low = v.lower()
    if any(x.lower() in low for x in REJECT):
        return True
    # reject high-entropy encoded-looking lines
    compact = re.sub(r"\s+", "", v)
    if len(compact) > 200:
        alnum = sum(ch.isalnum() or ch in "+/=" for ch in compact)
        if alnum / max(1, len(compact)) > 0.92:
            return True
    return False

rows = []
with INP.open(errors="ignore") as f:
    for line in f:
        try:
            r = json.loads(line)
        except Exception:
            continue
        v = str(r.get("value", ""))
        if is_junk(v):
            continue
        rows.append(r)

focused = []
bucketed = defaultdict(list)

for r in rows:
    v = str(r.get("value", ""))
    low = v.lower()
    count = int(r.get("count", 1) or 1)

    matched = []
    for bucket, keys in BUCKETS.items():
        if any(k.lower() in low for k in keys):
            matched.append(bucket)

    if not matched:
        continue

    score = count * 10
    if r.get("kind") == "repeated_code_line":
        score += 20
    if any(x in low for x in ["worldmodel", "goalreader", "framediff", "bfssolver", "action6", "forgenet"]):
        score += 30

    rr = dict(r)
    rr["buckets"] = matched
    rr["focus_score"] = score
    focused.append(rr)
    for b in matched:
        bucketed[b].append(rr)

focused.sort(key=lambda x: (-x["focus_score"], str(x.get("glyph", ""))))

with OUT_JSONL.open("w") as f:
    for r in focused:
        f.write(json.dumps(r, ensure_ascii=False) + "\n")

with OUT_MD.open("w") as f:
    f.write("# ARCAGI3 Focused Prior\n\n")
    f.write(f"Input rows: {sum(1 for _ in INP.open(errors='ignore'))}\n\n")
    f.write(f"Focused rows: {len(focused)}\n\n")

    for bucket in BUCKETS:
        items = sorted(bucketed[bucket], key=lambda x: -x["focus_score"])
        f.write(f"## {bucket} ({len(items)})\n\n")
        for r in items[:80]:
            f.write(
                f"- `{r.get('glyph')}` x{r.get('count', 1)} "
                f"score={r.get('focus_score')} :: `{r.get('value')}`\n"
            )
        f.write("\n")

print("[OK] focused rows:", len(focused))
print("[OK] wrote:", OUT_JSONL)
print("[OK] wrote:", OUT_MD)

for bucket in BUCKETS:
    print(f"{bucket:14s}", len(bucketed[bucket]))
