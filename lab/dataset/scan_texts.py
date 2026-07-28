import json
from collections import defaultdict
from pathlib import Path

root = Path(__file__).parents[2]
files = sorted((root / "datasets").rglob("*.txt"))
fields = ["Vehicle", "Threshold Option", "Threshold Value", "Ch1_DBC_file", "Ch2_DBC_file", "Time_setup"]
miss = "<MISSING>"


def rel(p): return p.relative_to(root).as_posix()
def dump(v): return json.dumps(v, ensure_ascii=False, sort_keys=True)
def diff(a, b):
    if isinstance(a, dict) and isinstance(b, dict):
        ks = sorted(set(a) | set(b))
        return "; ".join(f"{k}: {dump(a.get(k, miss))} -> {dump(b.get(k, miss))}" for k in ks if a.get(k, miss) != b.get(k, miss))
    return f"{dump(a)} -> {dump(b)}"


def tail(p):
    with p.open("rb") as f:
        f.seek(0, 2); pos = f.tell(); buf = b""
        while pos:
            n = min(4096, pos); pos -= n; f.seek(pos); buf = f.read(n) + buf
            lines = buf.splitlines()
            if len(lines) > 1:
                return lines[-1].decode("utf-8", "replace")
    return ""


def duration(info):
    csv = info.with_name(info.name.replace("_info_", "_state_").replace(".txt", ".csv"))
    if not csv.exists():
        return None
    with csv.open(encoding="utf-8", errors="replace") as f:
        f.readline()
        first = f.readline().split(",", 1)[0]
    return float(tail(csv).split(",", 1)[0]) - float(first)


rows, bad = [], []
for f in files:
    try:
        rows.append((f, json.loads(f.read_text(encoding="utf-8", errors="replace"))))
    except Exception as e:
        bad.append((f, e))

print(f"txt={len(files)} json={len(rows)} bad={len(bad)}")
for f, e in bad:
    print(f"BAD {rel(f)}: {e}")

by_keys = defaultdict(list)
for f, d in rows:
    by_keys[frozenset(d)].append(f)
base_keys = max(by_keys, key=lambda k: len(by_keys[k]))
print(f"\nKEYS: {len(by_keys)} set(s), normal={len(by_keys[base_keys])}/{len(rows)}")
print("normal:", ", ".join(sorted(base_keys)))
for ks, fs in sorted(by_keys.items(), key=lambda x: -len(x[1])):
    if ks == base_keys:
        continue
    print(f"\n{len(fs)} file(s): -{sorted(base_keys - ks)} +{sorted(ks - base_keys)}")
    for f in fs:
        print(" ", rel(f))

for k in fields:
    vals, raw = defaultdict(list), {}
    for f, d in rows:
        v = d[k] if k in d else miss
        vals[dump(v)].append(f)
        raw[dump(v)] = v
    base = max(vals, key=lambda v: len(vals[v]))
    print(f"\n{k}: {len(vals)} value(s), normal={len(vals[base])}/{len(rows)} {base}")
    for v, fs in sorted(vals.items(), key=lambda x: (-len(x[1]), x[0])):
        if v == base:
            continue
        print(f"  {len(fs)} file(s): {diff(raw[base], raw[v])}")
        for f in fs:
            print(f"    {rel(f)}")

print("\nDURATION by Time_setup")
by_time, no_csv = defaultdict(list), []
for f, d in rows:
    sec = duration(f)
    (no_csv if sec is None else by_time[dump(d.get("Time_setup"))]).append((sec, f, d.get("Time_setup")))
for k, xs in sorted(by_time.items(), key=lambda x: -len(x[1])):
    ds = sorted(x[0] for x in xs)
    t = xs[0][2] if isinstance(xs[0][2], dict) else {}
    ac, abc, med = t.get("A", 0) + t.get("C", 0), sum(t.values()) if t else None, ds[len(ds) // 2]
    print(f"{k}: n={len(ds)}, log={ds[0]:.2f}/{med:.2f}/{ds[-1]:.2f}s(min/med/max), A+C={ac}s, A+B+C={abc}s")
    off = [(sec, f) for sec, f, _ in xs if abs(sec - med) > 0.1]
    for sec, f in off:
        print(f"  outlier {sec:.2f}s {rel(f)}")
for _, f, _ in no_csv:
    print(f"NO CSV {rel(f)}")
