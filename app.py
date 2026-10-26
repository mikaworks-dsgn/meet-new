"""まだ知らない好きに、出会う — 軸を選んで、ちょうどいい意外さのある提案を見る画面。

実行: streamlit run app.py
"""
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import streamlit as st

DATA = Path(__file__).resolve().parent / "data"
MAX_SHOW = 5  # 1つの軸について表示するおすすめの上限
NOT_FOUND = "ちょうどいい、意外な、組み合わせは、見つかりませんでした"


@st.cache_data
def load_json(name):
    return json.loads((DATA / name).read_text(encoding="utf-8"))


@st.cache_resource
def load_vectors():
    """正規化したベクトルと、コンテンツ名ごとの行番号・軸名を返す。"""
    vecs = np.load(DATA / "embeddings.npy").astype(np.float32)
    vecs /= np.linalg.norm(vecs, axis=1, keepdims=True)
    rows = defaultdict(list)
    names = []
    for i, (name, axis, _group) in enumerate(load_json("axis_index.json")):
        rows[name].append(i)
        names.append(axis)
    return vecs, dict(rows), names


def unique_axis(src, dst):
    """src の軸のうち、dst の全軸と比べて最も離れている軸と、
    その軸から見て dst で最も遠い軸を返す。
    戻り値: (src の軸, dst の軸, 2つの類似度, src の軸と dst の全軸との最大類似度)"""
    vecs, rows, names = load_vectors()
    sim = vecs[rows[src]] @ vecs[rows[dst]].T  # 10 x 10
    i = int(sim.max(axis=1).argmin())  # 相手のどの軸からも、いちばん遠い軸
    j = int(sim[i].argmin())
    return (names[rows[src][i]], names[rows[dst][j]],
            float(sim[i, j]), float(sim[i].max()))


main_axes = load_json("main_axes.json")
candidate_axes = load_json("candidate_axes.json")
axis_recs = load_json("axis_recommendations.json")
book_recs = load_json("book_recommendations.json")

contents = {**main_axes, **candidate_axes}  # メイン8件 → 候補62件の順


def label(name):
    return f"{name}(メイン)" if name in main_axes else name


def show_recs(content, recs, name_key):
    """おすすめを、類似度の高い順に MAX_SHOW 件まで表示する。"""
    for r in sorted(recs, key=lambda r: r["similarity"], reverse=True)[:MAX_SHOW]:
        u_src, u_dst, u_sim, u_max = unique_axis(content, r[name_key])
        with st.container(border=True):
            st.markdown(f"**{r[name_key]}**")
            st.caption(f"共通する軸:{r['axis']}({r['similarity']:.2f})"
                       f" / 独自の軸:{u_src} ⇔ {u_dst}({u_sim:.2f})")
            st.caption(f"「{u_src}」は、相手のどの軸とも類似度 {u_max:.2f} 以下")
            st.progress(min(max(r["similarity"], 0.0), 1.0),
                        text=f"類似度 {r['similarity']:.2f}")
    if len(recs) > MAX_SHOW:
        st.caption(f"ほか {len(recs) - MAX_SHOW} 件")


st.set_page_config(page_title="まだ知らない好きに、出会う", page_icon="🌱")
st.title("まだ知らない好きに、出会う")
st.write("好きなものを選んで、気になる軸を押すと、少し離れているけれど関係のある、"
         "コンテンツや名著を提案します。")

# 1. コンテンツを選ぶ
content = st.selectbox("好きなものを選ぶ", list(contents), format_func=label)

# コンテンツを変えたら、選んでいた軸をリセットする
if st.session_state.get("content") != content:
    st.session_state.content = content
    st.session_state.axis = None

# 2. 軸をボタンで並べる
st.subheader(f"「{content}」の軸")
cols = st.columns(2)
for n, axis in enumerate(contents[content]):
    selected = st.session_state.axis == axis
    if cols[n % 2].button(axis, key=f"axis_{n}", use_container_width=True,
                          type="primary" if selected else "secondary"):
        st.session_state.axis = axis
        st.rerun()

# 3. 選んだ軸のおすすめを表示する
axis = st.session_state.axis
if axis is None:
    st.info("軸のボタンを、1つ選んでください。")
else:
    key = f"{content}|{axis}"
    c_recs = axis_recs.get(key, [])
    b_recs = book_recs.get(key, [])

    st.divider()
    st.subheader(f"「{axis}」から見つけた、おすすめ")

    if not c_recs and not b_recs:
        st.warning(NOT_FOUND)
    else:
        left, right = st.columns(2)
        with left:
            st.markdown("#### コンテンツ")
            if c_recs:
                show_recs(content, c_recs, "content")
            else:
                st.caption("この軸では、見つかりませんでした。")
        with right:
            st.markdown("#### 名著")
            if b_recs:
                show_recs(content, b_recs, "book")
            else:
                st.caption("この軸では、見つかりませんでした。")
