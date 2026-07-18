#!/usr/bin/env python3
"""
按与 main.py / utils 相同的图构造方式，输出各数据集的 |V| 与 d_max。

图加载约定（与原先脚本一致）：
  - MovieLens / AmazonFashion / Facebook / GrQc：优先 entry 边表，否则 noedge。
  - Collab / PPA / Vessel：OGB train 正边对称化后的邻接，与 utils 一致。
"""

import os
import sys
import numpy as np
import torch

# 与 load_data.py 一致：OGB 依赖的 torch.load 需 weights_only=False（PyTorch 2.6+ 默认 True 会失败）
_original_torch_load = torch.load


def _safe_torch_load(*args, **kwargs):
    if "weights_only" not in kwargs:
        kwargs["weights_only"] = False
    return _original_torch_load(*args, **kwargs)


torch.load = _safe_torch_load

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, CURRENT_DIR)

import utils

# 与 main.py 一致；可通过环境变量 PRB_ROOT 覆盖
PROJECT_ROOT = os.environ.get("PRB_ROOT", "/mnt/data/xinyu/bandits_pj/PRB")


def max_degree_dmax(graph_manager):
    """与 main_dmax_track._max_degree_pre_update 相同。"""
    deg = np.asarray(graph_manager.degree).flatten()
    if deg.size == 0:
        return 0
    return int(np.max(deg))


def _pick_path(primary: str, fallback: str) -> str:
    if os.path.exists(primary):
        return primary
    if os.path.exists(fallback):
        return fallback
    return primary


def _load_summary(display_name: str, graph_manager) -> dict:
    g = graph_manager.get()
    return {
        "name": display_name,
        "|V|": int(g["num_nodes"]),
        "d_max": max_degree_dmax(graph_manager),
    }


def _print_dmax_explanation():
    print(
        "------------------------------------------------------------------\n"
        "d_max 含义（本脚本输出的数值）\n"
        "------------------------------------------------------------------\n"
        "  对当前加载的图 G，令 deg(i) 为 utils.graph_manager.degree[i]，与实验里\n"
        "  用于 PPR 的度向量一致。则：\n"
        "\n"
        "      d_max = max_i deg(i)\n"
        "\n"
        "  即「所有节点里最大的那个度」。有向二部/社交图中 deg 为按 A 行求和的出度；\n"
        "  OGB 对称无向图中为无向度。\n"
        "\n"
        "  与 main_dmax_track.py 中每轮 begin 时记录的 d_max 定义相同；本表是\n"
        "  「静态图」上算出的一个数。在线学习会不断加边，整条轨迹上\n"
        "  max_t d_max(t) 可以大于表中的 d_max（例如从 noedge 冷启动开始时表上为 0）。\n"
        "------------------------------------------------------------------\n"
    )


def main():
    prb_data = os.path.join(PROJECT_ROOT, "online_link_prediction", "data")
    ogb_root = os.path.join(PROJECT_ROOT, "dataset")

    jobs = []

    ml_entry = os.path.join(prb_data, "MovieLens", "movie_2000users_10000items_entry.npy")
    ml_noedge = os.path.join(prb_data, "MovieLens", "movie_2000users_10000items_noedge.npy")
    jobs.append(
        (
            "MovieLens",
            utils.MovieLens,
            _pick_path(ml_entry, ml_noedge),
            lambda gm: gm.load(2000, 10000),
        )
    )

    amz_entry = os.path.join(
        prb_data, "Amazon_fashion", "new", "amazon_fashion_4000users_entry.npy"
    )
    amz_noedge = os.path.join(
        prb_data,
        "Amazon_fashion",
        "new",
        "Insert",
        "Amazon_fashion_4000users_noedge.npy",
    )
    jobs.append(
        (
            "AmazonFashion",
            utils.Amazon_fashion,
            _pick_path(amz_entry, amz_noedge),
            lambda gm: gm.load(4000, 4000),
        )
    )

    fb_entry = os.path.join(prb_data, "Facebook", "facebook_combined_ALLusers_entry.npy")
    fb_noedge = os.path.join(
        prb_data, "Facebook", "Insert", "facebook_combined_ALLusers_noedge.npy"
    )
    jobs.append(
        (
            "Facebook",
            utils.Facebook,
            _pick_path(fb_entry, fb_noedge),
            lambda gm: gm.load(4039),
        )
    )

    grqc_entry = os.path.join(prb_data, "GrQc", "Insert", "GrQc_ALLusers_entry.npy")
    grqc_noedge = os.path.join(prb_data, "GrQc", "Insert", "GrQc_ALLusers_noedge.npy")
    jobs.append(
        (
            "GrQc",
            utils.Grqc,
            _pick_path(grqc_entry, grqc_noedge),
            lambda gm: gm.load(5242),
        )
    )

    jobs.append(("ogbl-Collab", utils.Collab, ogb_root, lambda gm: gm.load()))
    jobs.append(("ogbl-PPA", utils.PPA, ogb_root, lambda gm: gm.load()))
    jobs.append(("ogbl-Vessel", utils.Vessel, ogb_root, lambda gm: gm.load()))

    rows = []
    for display_name, cls, path, loader in jobs:
        if not os.path.exists(path):
            print(f"[{display_name}] 跳过：路径不存在\n  {path}\n")
            rows.append({"name": display_name, "|V|": None, "d_max": None})
            continue
        try:
            gm = cls(path)
            loader(gm)
            rows.append(_load_summary(display_name, gm))
        except Exception as e:
            print(f"[{display_name}] 跳过：加载失败\n  {e}\n")
            rows.append({"name": display_name, "|V|": None, "d_max": None})

    print(f"PROJECT_ROOT = {PROJECT_ROOT}\n")
    _print_dmax_explanation()
    hdr = f"{'Dataset':<16} | {'|V|':>12} | {'d_max':>10}"
    print(hdr)
    print("-" * len(hdr))
    for r in rows:
        v = r.get("|V|")
        dm = r.get("d_max")
        vs = f"{v}" if v is not None else "N/A"
        dms = f"{dm}" if dm is not None else "N/A"
        print(f"{r['name']:<16} | {vs:>12} | {dms:>10}")


if __name__ == "__main__":
    main()
