#!/usr/bin/env python3
"""
Core-group mapping for pruned SNN networks.

This script maps logical neurons to hardware core_groups.  The objective is to
place neurons that frequently exchange events into the same group, while keeping
per-group neuron count, event load, and local-edge usage within bounds.

Supported input model formats:
  - BP pruning model from tests/onchip_stdp_prune.py:
      fc1_weight: [hidden, input]
      fc2_weight: [classes, hidden]
  - Legacy deployment package:
      q_weights: [classifier, input]

Example:
  python3 tests/coregroup_mapping.py \\
    --model data/cache/bp_prune_model_1000h_10c.npz \\
    --output data/cache/coregroup_mapping_bp.npz \\
    --num-groups 16 --neurons-per-group 128 \\
    --input-hidden-topk 16 --hidden-output-topk 10
"""

from __future__ import annotations

import argparse
import json
import os

import numpy as np


def load_model_weights(path: str) -> dict:
    data = np.load(path, allow_pickle=True)
    keys = set(data.files)
    if {"fc1_weight", "fc2_weight"}.issubset(keys):
        fc1 = np.asarray(data["fc1_weight"], dtype=np.float32)
        fc2 = np.asarray(data["fc2_weight"], dtype=np.float32)
        if fc1.ndim != 2 or fc2.ndim != 2:
            raise ValueError("fc1_weight/fc2_weight must be 2-D")
        return {
            "kind": "bp_2layer",
            "fc1": fc1,
            "fc2": fc2,
            "n_input": int(fc1.shape[1]),
            "n_hidden": int(fc1.shape[0]),
            "n_output": int(fc2.shape[0]),
        }
    if "q_weights" in keys:
        q = np.asarray(data["q_weights"], dtype=np.float32)
        if q.ndim != 2:
            raise ValueError("q_weights must be 2-D")
        return {
            "kind": "legacy_input_output",
            "fc1": q,
            "fc2": None,
            "n_input": int(q.shape[1]),
            "n_hidden": int(q.shape[0]),
            "n_output": 0,
        }
    raise ValueError(
        f"Unsupported model format in {path}. Keys={sorted(keys)}; "
        "expected fc1_weight/fc2_weight or q_weights."
    )


def model_from_arrays(fc1: np.ndarray, fc2: np.ndarray | None = None) -> dict:
    """Build an internal model dict directly from in-memory weight arrays."""
    fc1 = np.asarray(fc1, dtype=np.float32)
    if fc1.ndim != 2:
        raise ValueError("fc1 must be 2-D [hidden, input]")
    if fc2 is not None:
        fc2 = np.asarray(fc2, dtype=np.float32)
        if fc2.ndim != 2:
            raise ValueError("fc2 must be 2-D [output, hidden]")
        if fc2.shape[1] != fc1.shape[0]:
            raise ValueError(f"fc2 hidden dim {fc2.shape[1]} != fc1 hidden {fc1.shape[0]}")
    return {
        "kind": "bp_2layer" if fc2 is not None else "legacy_input_output",
        "fc1": fc1,
        "fc2": fc2,
        "n_input": int(fc1.shape[1]),
        "n_hidden": int(fc1.shape[0]),
        "n_output": int(0 if fc2 is None else fc2.shape[0]),
    }


def load_input_rates(path: str | None, n_input: int, default_rate: float) -> np.ndarray:
    if path is None:
        return np.full(n_input, float(default_rate), dtype=np.float32)
    data = np.load(path, allow_pickle=True)
    if "input_rates" in data.files:
        rates = np.asarray(data["input_rates"], dtype=np.float32).reshape(-1)
    elif "test_imgs" in data.files:
        imgs = np.asarray(data["test_imgs"], dtype=np.float32)
        rates = imgs.reshape(len(imgs), -1).mean(axis=0)
    else:
        raise ValueError(f"{path} has no input_rates or test_imgs")
    if rates.shape[0] != n_input:
        raise ValueError(f"input rate length {rates.shape[0]} != n_input {n_input}")
    return np.maximum(rates.astype(np.float32), 1e-6)


def input_rates_from_images(images: np.ndarray, n_input: int) -> np.ndarray:
    imgs = np.asarray(images, dtype=np.float32)
    rates = imgs.reshape(len(imgs), -1).mean(axis=0)
    if rates.shape[0] != n_input:
        raise ValueError(f"image-derived input rate length {rates.shape[0]} != n_input {n_input}")
    return np.maximum(rates.astype(np.float32), 1e-6)


def topk_indices_per_source(weight_dst_by_src: np.ndarray, k: int,
                            min_abs_weight: float) -> list[np.ndarray]:
    """Return top-k destination indices for each source column."""
    n_dst, n_src = weight_dst_by_src.shape
    k = min(max(int(k), 0), n_dst)
    result: list[np.ndarray] = []
    abs_w = np.abs(weight_dst_by_src)
    for src in range(n_src):
        col = abs_w[:, src]
        valid = np.where(col > float(min_abs_weight))[0]
        if k == 0 or len(valid) == 0:
            result.append(np.asarray([], dtype=np.int32))
            continue
        if len(valid) > k:
            part = valid[np.argpartition(col[valid], -k)[-k:]]
        else:
            part = valid
        ordered = part[np.argsort(-col[part])]
        result.append(ordered.astype(np.int32))
    return result


def estimate_layer_rates(input_rates: np.ndarray,
                         fc1: np.ndarray,
                         fc2: np.ndarray | None,
                         ih_fanouts: list[np.ndarray],
                         ho_fanouts: list[np.ndarray] | None) -> tuple[np.ndarray, np.ndarray]:
    """Cheap event-rate proxy for mapping; not a neuron-accurate simulation."""
    hidden_drive = np.zeros(fc1.shape[0], dtype=np.float32)
    abs_fc1 = np.abs(fc1)
    for src, dsts in enumerate(ih_fanouts):
        if len(dsts) == 0:
            continue
        hidden_drive[dsts] += input_rates[src] * abs_fc1[dsts, src]
    hidden_rates = hidden_drive / max(float(hidden_drive.max()), 1e-6)
    hidden_rates = np.maximum(hidden_rates, 1e-6)

    if fc2 is None or ho_fanouts is None:
        return hidden_rates, np.zeros(0, dtype=np.float32)
    output_drive = np.zeros(fc2.shape[0], dtype=np.float32)
    abs_fc2 = np.abs(fc2)
    for hidden, dsts in enumerate(ho_fanouts):
        if len(dsts) == 0:
            continue
        output_drive[dsts] += hidden_rates[hidden] * abs_fc2[dsts, hidden]
    output_rates = output_drive / max(float(output_drive.max()), 1e-6)
    output_rates = np.maximum(output_rates, 1e-6)
    return hidden_rates, output_rates


def build_graph(model: dict,
                input_rates: np.ndarray,
                input_hidden_topk: int,
                hidden_output_topk: int,
                min_abs_weight: float,
                fc1_mask: np.ndarray | None = None,
                fc2_mask: np.ndarray | None = None) -> dict:
    n_input = model["n_input"]
    n_hidden = model["n_hidden"]
    n_output = model["n_output"]
    fc1 = model["fc1"]
    fc2 = model["fc2"]

    if fc1_mask is not None:
        fc1_mask = np.asarray(fc1_mask, dtype=bool)
        if fc1_mask.shape != fc1.shape:
            raise ValueError(f"fc1_mask shape {fc1_mask.shape} != fc1 shape {fc1.shape}")
        ih_fanouts = [
            np.where(fc1_mask[:, src])[0].astype(np.int32)
            for src in range(n_input)
        ]
    else:
        ih_fanouts = topk_indices_per_source(fc1, input_hidden_topk, min_abs_weight)
    ho_fanouts = None
    if fc2 is not None:
        if fc2_mask is not None:
            fc2_mask = np.asarray(fc2_mask, dtype=bool)
            if fc2_mask.shape != fc2.shape:
                raise ValueError(f"fc2_mask shape {fc2_mask.shape} != fc2 shape {fc2.shape}")
            ho_fanouts = [
                np.where(fc2_mask[:, hidden])[0].astype(np.int32)
                for hidden in range(n_hidden)
            ]
        else:
            ho_fanouts = topk_indices_per_source(fc2, hidden_output_topk, min_abs_weight)

    hidden_rates, output_rates = estimate_layer_rates(input_rates, fc1, fc2, ih_fanouts, ho_fanouts)

    total_nodes = n_input + n_hidden + n_output
    node_type = np.empty(total_nodes, dtype=object)
    node_type[:n_input] = "input"
    node_type[n_input:n_input + n_hidden] = "hidden"
    if n_output:
        node_type[n_input + n_hidden:] = "output"

    node_rate = np.zeros(total_nodes, dtype=np.float32)
    node_rate[:n_input] = input_rates
    node_rate[n_input:n_input + n_hidden] = hidden_rates
    if n_output:
        node_rate[n_input + n_hidden:] = output_rates

    edges: list[tuple[int, int, float, float]] = []
    abs_fc1 = np.abs(fc1)
    for src in range(n_input):
        for h in ih_fanouts[src]:
            dst = n_input + int(h)
            importance = float(abs_fc1[int(h), src])
            event_weight = float(input_rates[src] * importance)
            edges.append((src, dst, event_weight, importance))

    if fc2 is not None and ho_fanouts is not None:
        abs_fc2 = np.abs(fc2)
        for h in range(n_hidden):
            src = n_input + h
            for out in ho_fanouts[h]:
                dst = n_input + n_hidden + int(out)
                importance = float(abs_fc2[int(out), h])
                event_weight = float(hidden_rates[h] * importance)
                edges.append((src, dst, event_weight, importance))

    in_event = np.zeros(total_nodes, dtype=np.float32)
    out_event = np.zeros(total_nodes, dtype=np.float32)
    for src, dst, ew, _imp in edges:
        out_event[src] += ew
        in_event[dst] += ew
    node_weight = node_rate + in_event
    priority = node_weight + in_event + out_event

    return {
        "n_input": n_input,
        "n_hidden": n_hidden,
        "n_output": n_output,
        "total_nodes": total_nodes,
        "node_type": node_type,
        "node_rate": node_rate,
        "node_weight": node_weight,
        "priority": priority,
        "edges": edges,
    }


class Mapper:
    def __init__(self, graph: dict, num_groups: int, neurons_per_group: int,
                 load_cap: float, local_edge_cap: int,
                 lambda_load: float, lambda_cap: float,
                 lambda_balance: float):
        self.graph = graph
        self.K = int(num_groups)
        self.N_cap = int(neurons_per_group)
        self.L_cap = float(load_cap)
        self.E_cap = int(local_edge_cap)
        self.lambda_load = float(lambda_load)
        self.lambda_cap = float(lambda_cap)
        self.lambda_balance = float(lambda_balance)

        self.N = int(graph["total_nodes"])
        self.P = np.full(self.N, -1, dtype=np.int32)
        self.group_nodes: list[set[int]] = [set() for _ in range(self.K)]
        self.group_load = np.zeros(self.K, dtype=np.float64)
        self.group_local_edges = np.zeros(self.K, dtype=np.int64)
        self.node_weight = np.asarray(graph["node_weight"], dtype=np.float64)
        self.priority = np.asarray(graph["priority"], dtype=np.float64)
        self.out_adj: list[list[tuple[int, float]]] = [[] for _ in range(self.N)]
        self.in_adj: list[list[tuple[int, float]]] = [[] for _ in range(self.N)]
        for src, dst, ew, _imp in graph["edges"]:
            self.out_adj[src].append((dst, ew))
            self.in_adj[dst].append((src, ew))

    def local_gain(self, node: int, group: int) -> float:
        nodes = self.group_nodes[group]
        gain = 0.0
        for dst, ew in self.out_adj[node]:
            if dst in nodes:
                gain += ew
        for src, ew in self.in_adj[node]:
            if src in nodes:
                gain += ew
        return float(gain)

    def local_edge_count_gain(self, node: int, group: int) -> int:
        nodes = self.group_nodes[group]
        count = 0
        for dst, _ew in self.out_adj[node]:
            if dst in nodes:
                count += 1
        for src, _ew in self.in_adj[node]:
            if src in nodes:
                count += 1
        return count

    def placement_score(self, node: int, group: int) -> float:
        if len(self.group_nodes[group]) + 1 > self.N_cap:
            return float("inf")
        tentative_load = self.group_load[group] + self.node_weight[node]
        tentative_edges = self.group_local_edges[group] + self.local_edge_count_gain(node, group)
        load_penalty = max(0.0, tentative_load - self.L_cap)
        local_cap_penalty = max(0, int(tentative_edges) - self.E_cap)
        balance_penalty = tentative_load
        return (
            -self.local_gain(node, group)
            + self.lambda_load * load_penalty
            + self.lambda_cap * local_cap_penalty
            + self.lambda_balance * balance_penalty
        )

    def assign(self, node: int, group: int) -> None:
        self.P[node] = group
        self.group_nodes[group].add(node)
        self.group_load[group] += self.node_weight[node]
        self.group_local_edges[group] += self.local_edge_count_gain(node, group)

    def remove(self, node: int, group: int) -> None:
        self.group_nodes[group].remove(node)
        self.group_load[group] -= self.node_weight[node]
        self.group_local_edges[group] -= self.local_edge_count_gain(node, group)
        self.P[node] = -1

    def greedy(self) -> None:
        order = np.argsort(-self.priority)
        for node in order:
            scores = [self.placement_score(int(node), g) for g in range(self.K)]
            best = int(np.argmin(scores))
            if not np.isfinite(scores[best]):
                raise RuntimeError(
                    f"Unable to place node {node}; increase --num-groups or --neurons-per-group"
                )
            self.assign(int(node), best)

    def total_cost(self) -> float:
        cross = 0.0
        for src, dst, ew, _imp in self.graph["edges"]:
            if self.P[src] != self.P[dst]:
                cross += ew
        load_penalty = np.maximum(0.0, self.group_load - self.L_cap).sum()
        edge_penalty = np.maximum(0, self.group_local_edges - self.E_cap).sum()
        return float(cross + self.lambda_load * load_penalty + self.lambda_cap * edge_penalty)

    def refine(self, max_iter: int) -> None:
        if max_iter <= 0:
            return
        current_cost = self.total_cost()
        order = np.argsort(-self.priority)
        for _it in range(int(max_iter)):
            improved = False
            for node_raw in order:
                node = int(node_raw)
                old = int(self.P[node])
                best_group = old
                best_cost = current_cost
                self.remove(node, old)
                for group in range(self.K):
                    if group == old:
                        continue
                    if len(self.group_nodes[group]) + 1 > self.N_cap:
                        continue
                    self.assign(node, group)
                    cost = self.total_cost()
                    self.remove(node, group)
                    if cost < best_cost:
                        best_cost = cost
                        best_group = group
                self.assign(node, best_group)
                if best_group != old:
                    current_cost = best_cost
                    improved = True
            if not improved:
                break


def build_local_ids(P: np.ndarray, num_groups: int, neurons_per_group: int) -> tuple[np.ndarray, np.ndarray]:
    local = np.full(len(P), -1, dtype=np.int32)
    counts = np.zeros(num_groups, dtype=np.int32)
    for node, group in enumerate(P):
        idx = counts[group]
        if idx >= neurons_per_group:
            raise RuntimeError(f"group {group} exceeds local capacity")
        local[node] = idx
        counts[group] += 1
    global_id = (P.astype(np.int32) << int(np.ceil(np.log2(neurons_per_group)))) | local
    return local, global_id.astype(np.int32)


def summarize_mapping(graph: dict, P: np.ndarray, num_groups: int) -> dict:
    group_counts = np.bincount(P, minlength=num_groups).astype(int)
    group_load = np.zeros(num_groups, dtype=np.float64)
    for node, group in enumerate(P):
        group_load[group] += float(graph["node_weight"][node])

    local_edges = np.zeros(num_groups, dtype=np.int64)
    cross_edges = 0
    local_event = np.zeros(num_groups, dtype=np.float64)
    cross_event = 0.0
    total_event = 0.0
    for src, dst, ew, _imp in graph["edges"]:
        total_event += ew
        if P[src] == P[dst]:
            local_edges[P[src]] += 1
            local_event[P[src]] += ew
        else:
            cross_edges += 1
            cross_event += ew
    return {
        "group_counts": group_counts.tolist(),
        "group_load": group_load.tolist(),
        "group_local_edges": local_edges.astype(int).tolist(),
        "group_local_event_weight": local_event.tolist(),
        "cross_edges": int(cross_edges),
        "local_edges": int(local_edges.sum()),
        "total_edges": int(len(graph["edges"])),
        "cross_event_weight": float(cross_event),
        "local_event_weight": float(local_event.sum()),
        "total_event_weight": float(total_event),
        "local_edge_ratio": float(local_edges.sum() / max(len(graph["edges"]), 1)),
        "local_event_ratio": float(local_event.sum() / max(total_event, 1e-12)),
        "max_group_count": int(group_counts.max()) if len(group_counts) else 0,
        "max_group_load": float(group_load.max()) if len(group_load) else 0.0,
    }


def run_mapping(model: dict,
                input_rates: np.ndarray,
                num_groups: int = 16,
                neurons_per_group: int = 128,
                input_hidden_topk: int = 16,
                hidden_output_topk: int = 10,
                min_abs_weight: float = 0.0,
                fc1_mask: np.ndarray | None = None,
                fc2_mask: np.ndarray | None = None,
                load_cap: float = 0.0,
                local_edge_cap: int = 0,
                lambda_load: float = 10.0,
                lambda_cap: float = 1000.0,
                lambda_balance: float = 0.05,
                max_iter: int = 0) -> dict:
    """In-memory mapping API for training-time alternating prune/remap."""
    graph = build_graph(
        model=model,
        input_rates=input_rates,
        input_hidden_topk=input_hidden_topk,
        hidden_output_topk=hidden_output_topk,
        min_abs_weight=min_abs_weight,
        fc1_mask=fc1_mask,
        fc2_mask=fc2_mask,
    )
    total_capacity = int(num_groups) * int(neurons_per_group)
    if graph["total_nodes"] > total_capacity:
        raise ValueError(
            f"total nodes {graph['total_nodes']} exceed hardware capacity "
            f"{total_capacity} ({num_groups} groups x {neurons_per_group})"
        )
    if load_cap <= 0.0:
        load_cap = float(np.asarray(graph["node_weight"]).sum() / int(num_groups) * 1.20)
    if local_edge_cap <= 0:
        local_edge_cap = int(neurons_per_group * max(input_hidden_topk, hidden_output_topk, 1))

    mapper = Mapper(
        graph=graph,
        num_groups=num_groups,
        neurons_per_group=neurons_per_group,
        load_cap=load_cap,
        local_edge_cap=local_edge_cap,
        lambda_load=lambda_load,
        lambda_cap=lambda_cap,
        lambda_balance=lambda_balance,
    )
    mapper.greedy()
    mapper.refine(max_iter)
    local_id, global_id = build_local_ids(mapper.P, int(num_groups), int(neurons_per_group))
    summary = summarize_mapping(graph, mapper.P, int(num_groups))
    return {
        "graph": graph,
        "neuron_group": mapper.P.astype(np.int32),
        "neuron_local": local_id.astype(np.int32),
        "neuron_global_id": global_id.astype(np.int32),
        "summary": summary,
        "load_cap": float(load_cap),
        "local_edge_cap": int(local_edge_cap),
        "num_groups": int(num_groups),
        "neurons_per_group": int(neurons_per_group),
        "input_hidden_topk": int(input_hidden_topk),
        "hidden_output_topk": int(hidden_output_topk),
    }


def compute_connection_importance(model: dict,
                                  input_rates: np.ndarray,
                                  mapping: dict | None = None,
                                  mode: str = "rate-weight") -> dict:
    """Estimate dense connection importance.

    Current mainline implements rate-weight:
        input->hidden:  input_rate[p] * abs(W1[h, p])
        hidden->output: hidden_rate[h] * abs(W2[o, h])

    The function is intentionally isolated so gradient/EMA proxies can replace
    or augment it later.
    """
    if mode != "rate-weight":
        raise ValueError(f"unsupported importance mode: {mode}")
    fc1 = np.asarray(model["fc1"], dtype=np.float32)
    fc2 = model["fc2"]
    input_rates = np.asarray(input_rates, dtype=np.float32).reshape(-1)
    if input_rates.shape[0] != fc1.shape[1]:
        raise ValueError(f"input_rates length {input_rates.shape[0]} != fc1 input {fc1.shape[1]}")

    fc1_importance = np.abs(fc1) * input_rates.reshape(1, -1)

    fc2_importance = None
    if fc2 is not None:
        fc2 = np.asarray(fc2, dtype=np.float32)
        if mapping is not None:
            n_input = int(model["n_input"])
            n_hidden = int(model["n_hidden"])
            node_rate = np.asarray(mapping["graph"]["node_rate"], dtype=np.float32)
            hidden_rates = node_rate[n_input:n_input + n_hidden]
        else:
            hidden_drive = np.sum(fc1_importance, axis=1)
            hidden_rates = hidden_drive / max(float(hidden_drive.max()), 1e-6)
        fc2_importance = np.abs(fc2) * hidden_rates.reshape(1, -1)

    return {
        "fc1_importance": fc1_importance.astype(np.float32),
        "fc2_importance": None if fc2_importance is None else fc2_importance.astype(np.float32),
    }


def estimate_runtime_costs(model: dict,
                           mapping: dict,
                           cross_rate_coeff: float = 1.0,
                           load_balance_coeff: float = 1.0,
                           load_reward_cap: float = 0.5,
                           load_penalty_cap: float = 1.0) -> dict:
    """Estimate potential runtime cost for every dense edge under mapping P.

    Current cost model:

        cost(i,j) =
            cross_rate_coeff * norm(a_i) * is_cross(i,j)
          + load_balance_coeff * clamp(norm(L_dst_group - mean_g L_g))

    where a_i is the source firing-rate proxy and L_g is the active-graph event
    load delivered to group g.  The load term is allowed to become negative for
    below-average groups, but is capped to avoid over-rewarding low-load groups.
    This intentionally does not use the older fixed cross/load penalties.
    """
    fc1 = np.asarray(model["fc1"], dtype=np.float32)
    fc2 = model["fc2"]
    P = np.asarray(mapping["neuron_group"], dtype=np.int32)
    graph = mapping["graph"]
    node_rate = np.asarray(graph["node_rate"], dtype=np.float32)
    num_groups = int(mapping["num_groups"])
    group_event_load = np.zeros(num_groups, dtype=np.float32)
    for src, dst, ew, _imp in graph["edges"]:
        dst_group = int(P[int(dst)])
        if 0 <= dst_group < num_groups:
            group_event_load[dst_group] += float(ew)
    mean_group_event_load = float(group_event_load.mean()) if num_groups > 0 else 0.0
    group_load_delta_norm = (
        (group_event_load - mean_group_event_load) / max(mean_group_event_load, 1e-6)
    )
    group_load_delta_norm = np.clip(
        group_load_delta_norm,
        -float(load_reward_cap),
        float(load_penalty_cap),
    ).astype(np.float32)
    n_input = int(model["n_input"])
    n_hidden = int(model["n_hidden"])
    mean_source_rate = float(node_rate.mean()) if node_rate.size else 0.0
    node_rate_norm = node_rate / max(mean_source_rate, 1e-6)

    fc1_cost = np.zeros_like(fc1, dtype=np.float32)
    for src in range(n_input):
        src_group = P[src]
        dst_nodes = n_input + np.arange(n_hidden)
        dst_groups = P[dst_nodes]
        is_cross = (dst_groups != src_group).astype(np.float32)
        fc1_cost[:, src] = (
            float(cross_rate_coeff) * float(node_rate_norm[src]) * is_cross
            + float(load_balance_coeff) * group_load_delta_norm[dst_groups]
        )

    fc2_cost = None
    if fc2 is not None:
        fc2 = np.asarray(fc2, dtype=np.float32)
        n_output = int(model["n_output"])
        fc2_cost = np.zeros_like(fc2, dtype=np.float32)
        dst_nodes = n_input + n_hidden + np.arange(n_output)
        dst_groups = P[dst_nodes]
        for hidden in range(n_hidden):
            src_node = n_input + hidden
            src_group = P[src_node]
            is_cross = (dst_groups != src_group).astype(np.float32)
            fc2_cost[:, hidden] = (
                float(cross_rate_coeff) * float(node_rate_norm[src_node]) * is_cross
                + float(load_balance_coeff) * group_load_delta_norm[dst_groups]
            )

    return {
        "fc1_runtime_cost": fc1_cost,
        "fc2_runtime_cost": fc2_cost,
        "group_event_load": group_event_load,
        "mean_group_event_load": np.array(mean_group_event_load, dtype=np.float32),
        "group_load_delta_norm": group_load_delta_norm,
    }


def select_topk_masks(importance: np.ndarray,
                      runtime_cost: np.ndarray,
                      topk: int,
                      runtime_alpha: float = 1.0,
                      min_importance: float = 0.0) -> np.ndarray:
    """Per-source top-k mask from dense importance and runtime cost matrices."""
    importance = np.asarray(importance, dtype=np.float32)
    runtime_cost = np.asarray(runtime_cost, dtype=np.float32)
    if importance.shape != runtime_cost.shape:
        raise ValueError(f"importance shape {importance.shape} != cost shape {runtime_cost.shape}")
    n_dst, n_src = importance.shape
    mask = np.zeros_like(importance, dtype=np.float32)
    k = min(max(int(topk), 0), n_dst)
    if k == 0:
        return mask
    denom = np.maximum(1.0e-6, 1.0 + float(runtime_alpha) * runtime_cost)
    score = importance / denom
    for src in range(n_src):
        candidates = np.where(importance[:, src] > float(min_importance))[0]
        if len(candidates) == 0:
            continue
        kk = min(k, len(candidates))
        keep = candidates[np.argpartition(score[candidates, src], -kk)[-kk:]]
        mask[keep, src] = 1.0
    return mask


def build_initial_topk_masks(model: dict,
                             input_rates: np.ndarray,
                             input_hidden_topk: int,
                             hidden_output_topk: int,
                             min_importance: float = 0.0,
                             importance_mode: str = "rate-weight") -> dict:
    imp = compute_connection_importance(
        model=model,
        input_rates=input_rates,
        mapping=None,
        mode=importance_mode,
    )
    fc1_cost = np.zeros_like(imp["fc1_importance"], dtype=np.float32)
    fc1_mask = select_topk_masks(
        imp["fc1_importance"], fc1_cost, input_hidden_topk,
        runtime_alpha=0.0, min_importance=min_importance)
    fc2_mask = None
    if imp["fc2_importance"] is not None:
        fc2_cost = np.zeros_like(imp["fc2_importance"], dtype=np.float32)
        fc2_mask = select_topk_masks(
            imp["fc2_importance"], fc2_cost, hidden_output_topk,
            runtime_alpha=0.0, min_importance=min_importance)
    return {
        "fc1_mask": fc1_mask,
        "fc2_mask": fc2_mask,
        "fc1_density": float(fc1_mask.mean()),
        "fc2_density": None if fc2_mask is None else float(fc2_mask.mean()),
    }


def build_cost_aware_masks(model: dict,
                           mapping: dict,
                           input_rates: np.ndarray,
                           input_hidden_topk: int,
                           hidden_output_topk: int,
                           cross_rate_coeff: float = 1.0,
                           load_balance_coeff: float = 1.0,
                           load_reward_cap: float = 0.5,
                           load_penalty_cap: float = 1.0,
                           min_abs_weight: float = 0.0,
                           runtime_alpha: float = 1.0,
                           importance_mode: str = "rate-weight") -> dict:
    """Generate dense-range active masks from importance and mapping-derived cost."""
    imp = compute_connection_importance(model, input_rates, mapping, mode=importance_mode)
    costs = estimate_runtime_costs(
        model=model,
        mapping=mapping,
        cross_rate_coeff=cross_rate_coeff,
        load_balance_coeff=load_balance_coeff,
        load_reward_cap=load_reward_cap,
        load_penalty_cap=load_penalty_cap,
    )
    mask1 = select_topk_masks(
        imp["fc1_importance"], costs["fc1_runtime_cost"],
        input_hidden_topk, runtime_alpha=runtime_alpha,
        min_importance=min_abs_weight)
    mask2 = None
    if imp["fc2_importance"] is not None:
        mask2 = select_topk_masks(
            imp["fc2_importance"], costs["fc2_runtime_cost"],
            hidden_output_topk, runtime_alpha=runtime_alpha,
            min_importance=min_abs_weight)

    return {
        "fc1_mask": mask1,
        "fc2_mask": mask2,
        "fc1_density": float(mask1.mean()),
        "fc2_density": None if mask2 is None else float(mask2.mean()),
        "fc1_importance": imp["fc1_importance"],
        "fc2_importance": imp["fc2_importance"],
        "fc1_runtime_cost": costs["fc1_runtime_cost"],
        "fc2_runtime_cost": costs["fc2_runtime_cost"],
        "group_event_load": costs["group_event_load"],
        "mean_group_event_load": costs["mean_group_event_load"],
        "group_load_delta_norm": costs["group_load_delta_norm"],
    }


def write_outputs(path: str, graph: dict, P: np.ndarray, local_id: np.ndarray,
                  global_id: np.ndarray, summary: dict, args: argparse.Namespace) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    src = np.asarray([e[0] for e in graph["edges"]], dtype=np.int32)
    dst = np.asarray([e[1] for e in graph["edges"]], dtype=np.int32)
    event_weight = np.asarray([e[2] for e in graph["edges"]], dtype=np.float32)
    importance = np.asarray([e[3] for e in graph["edges"]], dtype=np.float32)
    same_group = (P[src] == P[dst]).astype(np.uint8)

    np.savez_compressed(
        path,
        neuron_group=P.astype(np.int32),
        neuron_local=local_id.astype(np.int32),
        neuron_global_id=global_id.astype(np.int32),
        node_type=np.asarray(graph["node_type"]).astype(str),
        node_rate=np.asarray(graph["node_rate"], dtype=np.float32),
        node_weight=np.asarray(graph["node_weight"], dtype=np.float32),
        edge_src=src,
        edge_dst=dst,
        edge_event_weight=event_weight,
        edge_importance=importance,
        edge_same_group=same_group,
        n_input=np.array(graph["n_input"], dtype=np.int32),
        n_hidden=np.array(graph["n_hidden"], dtype=np.int32),
        n_output=np.array(graph["n_output"], dtype=np.int32),
        input_offset=np.array(0, dtype=np.int32),
        hidden_offset=np.array(graph["n_input"], dtype=np.int32),
        output_offset=np.array(graph["n_input"] + graph["n_hidden"], dtype=np.int32),
        num_groups=np.array(args.num_groups, dtype=np.int32),
        neurons_per_group=np.array(args.neurons_per_group, dtype=np.int32),
        local_id_width=np.array(int(np.ceil(np.log2(args.neurons_per_group))), dtype=np.int32),
        input_hidden_topk=np.array(args.input_hidden_topk, dtype=np.int32),
        hidden_output_topk=np.array(args.hidden_output_topk, dtype=np.int32),
    )

    json_path = os.path.splitext(path)[0] + ".json"
    meta = {
        "model": args.model,
        "output_npz": path,
        "num_groups": args.num_groups,
        "neurons_per_group": args.neurons_per_group,
        "input_hidden_topk": args.input_hidden_topk,
        "hidden_output_topk": args.hidden_output_topk,
        "load_cap": args.load_cap,
        "local_edge_cap": args.local_edge_cap,
        "summary": summary,
    }
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Map pruned SNN neurons to core_groups")
    p.add_argument("--model", required=True, help="Input model .npz")
    p.add_argument("--output", required=True, help="Output mapping .npz")
    p.add_argument("--activity", default=None,
                   help="Optional .npz with input_rates or test_imgs for input firing rates")
    p.add_argument("--num-groups", type=int, default=16)
    p.add_argument("--neurons-per-group", type=int, default=128)
    p.add_argument("--input-hidden-topk", type=int, default=16)
    p.add_argument("--hidden-output-topk", type=int, default=10)
    p.add_argument("--min-abs-weight", type=float, default=0.0)
    p.add_argument("--default-input-rate", type=float, default=0.15)
    p.add_argument("--load-cap", type=float, default=0.0,
                   help="Max event load per group; 0 means auto average*1.20")
    p.add_argument("--local-edge-cap", type=int, default=0,
                   help="Max local edges per group; 0 means neurons_per_group*max(topk)")
    p.add_argument("--lambda-load", type=float, default=10.0)
    p.add_argument("--lambda-cap", type=float, default=1000.0)
    p.add_argument("--lambda-balance", type=float, default=0.05)
    p.add_argument("--max-iter", type=int, default=0,
                   help="Optional local refinement iterations; 0 keeps fast greedy mapping")
    p.add_argument("--seed", type=int, default=1)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    np.random.seed(args.seed)

    model = load_model_weights(args.model)
    input_rates = load_input_rates(args.activity, model["n_input"], args.default_input_rate)
    mapping = run_mapping(
        input_rates=input_rates,
        model=model,
        num_groups=args.num_groups,
        neurons_per_group=args.neurons_per_group,
        input_hidden_topk=args.input_hidden_topk,
        hidden_output_topk=args.hidden_output_topk,
        min_abs_weight=args.min_abs_weight,
        load_cap=args.load_cap,
        local_edge_cap=args.local_edge_cap,
        lambda_load=args.lambda_load,
        lambda_cap=args.lambda_cap,
        lambda_balance=args.lambda_balance,
        max_iter=args.max_iter,
    )
    args.load_cap = mapping["load_cap"]
    args.local_edge_cap = mapping["local_edge_cap"]
    graph = mapping["graph"]
    summary = mapping["summary"]
    write_outputs(
        args.output,
        graph,
        mapping["neuron_group"],
        mapping["neuron_local"],
        mapping["neuron_global_id"],
        summary,
        args,
    )

    print("=" * 72)
    print("Core-group Mapping Complete")
    print("=" * 72)
    print(f"  model:            {args.model}")
    print(f"  output:           {args.output}")
    print(f"  json:             {os.path.splitext(args.output)[0] + '.json'}")
    print(f"  model kind:        {model['kind']}")
    print(f"  nodes:            input={graph['n_input']} hidden={graph['n_hidden']} "
          f"output={graph['n_output']} total={graph['total_nodes']}")
    print(f"  groups:           {args.num_groups} x {args.neurons_per_group}")
    print(f"  topk:             input->hidden={args.input_hidden_topk}, "
          f"hidden->output={args.hidden_output_topk}")
    print(f"  group counts:     {summary['group_counts']}")
    print(f"  max group count:  {summary['max_group_count']} / {args.neurons_per_group}")
    print(f"  max group load:   {summary['max_group_load']:.6f} / cap {args.load_cap:.6f}")
    print(f"  edges:            total={summary['total_edges']} "
          f"local={summary['local_edges']} cross={summary['cross_edges']}")
    print(f"  local edge ratio: {summary['local_edge_ratio'] * 100.0:.2f}%")
    print(f"  local event ratio:{summary['local_event_ratio'] * 100.0:.2f}%")


if __name__ == "__main__":
    main()
