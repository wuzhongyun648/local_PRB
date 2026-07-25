"""Declarative GrQc ablation matrix."""

from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class Variant:
    name: str
    group: str
    entry_style: str = "insert"
    lr1: float = 0.01
    lr2: float = 0.004
    solver: str = "loc"
    eps: float = 1.91e-4
    power_steps: int = 50
    corrected_update: bool = False
    disjoint_split: bool = False
    shared_serving: bool = False
    positive_without_replacement: bool = False
    independent_eval: bool = False
    source_mode: str = "raw"

    def to_dict(self):
        return asdict(self)


VARIANTS = {
    # Group 1: anchors.
    "g1_original_loc": Variant(
        "g1_original_loc", "group1", entry_style="root",
        lr1=0.1, lr2=0.01, solver="loc", eps=1e-6,
    ),
    "g1_current_loc": Variant(
        "g1_current_loc", "group1", solver="loc",
    ),
    "g1_current_power": Variant(
        "g1_current_power", "group1", solver="power",
    ),
    # Group 2: semantics.
    "s0_legacy": Variant("s0_legacy", "group2"),
    "s1_selected_edge_update": Variant(
        "s1_selected_edge_update", "group2", corrected_update=True,
    ),
    "s2_disjoint_undirected_split": Variant(
        "s2_disjoint_undirected_split", "group2", disjoint_split=True,
    ),
    "s3_shared_serving": Variant(
        "s3_shared_serving", "group2", shared_serving=True,
    ),
    "s4_no_positive_replacement": Variant(
        "s4_no_positive_replacement", "group2",
        positive_without_replacement=True,
    ),
    "s5_independent_eval": Variant(
        "s5_independent_eval", "group2", independent_eval=True,
    ),
    "sf_all_fixed": Variant(
        "sf_all_fixed", "group2", corrected_update=True,
        disjoint_split=True, shared_serving=True,
        positive_without_replacement=True, independent_eval=True,
    ),
    # Group 3: reranking.
    "r0_eenet_direct": Variant(
        "r0_eenet_direct", "group3", solver="direct",
        corrected_update=True, disjoint_split=True, shared_serving=True,
        positive_without_replacement=True, independent_eval=True,
    ),
    "r1_locprb": Variant(
        "r1_locprb", "group3", solver="loc",
        corrected_update=True, disjoint_split=True, shared_serving=True,
        positive_without_replacement=True, independent_eval=True,
    ),
    "r2_power": Variant(
        "r2_power", "group3", solver="power",
        corrected_update=True, disjoint_split=True, shared_serving=True,
        positive_without_replacement=True, independent_eval=True,
    ),
    # Group 4: source.
    "p0_raw": Variant(
        "p0_raw", "group4", corrected_update=True, disjoint_split=True,
        shared_serving=True, positive_without_replacement=True,
        independent_eval=True, source_mode="raw",
    ),
    "p1_l1": Variant(
        "p1_l1", "group4", corrected_update=True, disjoint_split=True,
        shared_serving=True, positive_without_replacement=True,
        independent_eval=True, source_mode="l1",
    ),
    "p2_clip_l1": Variant(
        "p2_clip_l1", "group4", corrected_update=True, disjoint_split=True,
        shared_serving=True, positive_without_replacement=True,
        independent_eval=True, source_mode="clip_l1",
    ),
    "p3_softmax": Variant(
        "p3_softmax", "group4", corrected_update=True, disjoint_split=True,
        shared_serving=True, positive_without_replacement=True,
        independent_eval=True, source_mode="softmax",
    ),
    "p4_exploit_only": Variant(
        "p4_exploit_only", "group4", corrected_update=True,
        disjoint_split=True, shared_serving=True,
        positive_without_replacement=True, independent_eval=True,
        source_mode="exploit_only",
    ),
    "p5_exploit_explore": Variant(
        "p5_exploit_explore", "group4", corrected_update=True,
        disjoint_split=True, shared_serving=True,
        positive_without_replacement=True, independent_eval=True,
        source_mode="exploit_explore",
    ),
}


GROUPS = {
    group: [name for name, value in VARIANTS.items() if value.group == group]
    for group in ("group1", "group2", "group3", "group4")
}

