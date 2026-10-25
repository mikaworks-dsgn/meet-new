"""axis_recommendations.json と book_recommendations.json を作り直す。

条件:
- 一致する軸: 元の軸と、比較相手のどれかの軸との類似度が MATCH_TH 以上
- 独自の軸: 元のコンテンツの10軸のうち、比較相手のどの軸とも類似度が
  UNIQUE_TH 以下の軸(行の最大が UNIQUE_TH 以下の軸)が、1つ以上ある
両方を満たす組み合わせだけを、件数の上限なしで保存する。
ただし、元の軸と一致した相手の軸で、一方に否定の接頭語(非・不・無・反)が付き、
もう一方が接頭語を除いた残りの言葉と同じ場合(例:「非日常」と「日常」)は除外する。

使い方: python scripts/rebuild_recs.py
"""
import json
import re
from collections import defaultdict
from pathlib import Path

import numpy as np

MATCH_TH = 0.5    # 一致する軸: 相手のどれかの軸と、この値以上
UNIQUE_TH = 0.20  # 独自の軸: 相手のどの軸とも、この値以下
NEG_PREFIXES = "非不無反"  # 否定・対義の接頭語

DATA = Path(__file__).resolve().parent.parent / "data"

E = np.load(DATA / "embeddings.npy").astype(np.float32)
idx = json.loads((DATA / "axis_index.json").read_text(encoding="utf-8"))

E = E / np.linalg.norm(E, axis=1, keepdims=True)
S = E @ E.T  # 全軸どうしのコサイン類似度 (1930 x 1930)

rows = defaultdict(list)  # コンテンツ名 -> 行番号のリスト(axis_index の順)
group = {}
for i, (name, axis, g) in enumerate(idx):
    rows[name].append(i)
    group[name] = g

contents = [n for n in rows if group[n] in ("メイン", "候補")]
books = [n for n in rows if group[n] == "名著"]


def has_unique_axis(src, dst):
    """src の10軸のうち、dst のどの軸とも UNIQUE_TH 以下の軸が1つ以上あるか。"""
    sub = S[np.ix_(rows[src], rows[dst])]
    return bool((sub.max(axis=1) <= UNIQUE_TH).any())


KANJI = r"[一-鿿々]"
NEG_WORD = re.compile(f"([{NEG_PREFIXES}])({KANJI}{{2,}})")  # 例:「非日常」→ 非 + 日常


def contains_word(text, word, prefix):
    """text の中に word があり、直前が漢字(熟語の一部)でも prefix でもないか。"""
    for m in re.finditer(re.escape(word), text):
        before = text[m.start() - 1] if m.start() > 0 else ""
        if before != prefix and not re.fullmatch(KANJI, before or "a"):
            return True
    return False


def is_negation_pair(a, b):
    """一方に「接頭語+言葉」があり、もう一方にその言葉がそのままあるか。"""
    for x, y in ((a, b), (b, a)):
        for m in NEG_WORD.finditer(x):
            prefix, word = m.group(1), m.group(2)
            if contains_word(y, word, prefix):
                return True
    return False


excluded = defaultdict(int)  # ファイルごとの除外件数


def build(targets, key_name):
    result = {}
    for src in contents:
        # 独自の軸の条件は、軸ではなくコンテンツどうしで決まるので、先に計算する
        ok_targets = [t for t in targets if t != src and has_unique_axis(src, t)]
        for i in rows[src]:
            recs = []
            for t in ok_targets:
                cols = rows[t]
                sims = S[i, cols]
                j = int(sims.argmax())  # 相手の中で、いちばん近い軸を1つ
                if sims[j] >= MATCH_TH:
                    if is_negation_pair(idx[i][1], idx[cols[j]][1]):
                        excluded[key_name] += 1
                        continue
                    recs.append({
                        key_name: t,
                        "axis": idx[cols[j]][1],
                        "similarity": float(sims[j]),
                    })
            recs.sort(key=lambda r: r["similarity"], reverse=True)
            result[f"{src}|{idx[i][1]}"] = recs
    return result


for fname, targets, key_name in [
    ("axis_recommendations.json", contents, "content"),
    ("book_recommendations.json", books, "book"),
]:
    out = build(targets, key_name)
    (DATA / fname).write_text(
        json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    n = sum(len(v) for v in out.values())
    print(f"{fname}: keys={len(out)} recs={n} excluded={excluded[key_name]}")
