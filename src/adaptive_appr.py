"""Reversible adaptive APPR with shared Python/Numba kernels."""

import time
import warnings

import numpy as np

try:
    from numba import njit

    NUMBA_AVAILABLE = True
except ModuleNotFoundError:
    NUMBA_AVAILABLE = False
    warnings.warn(
        "Numba is unavailable; adaptive APPR is using Python kernels.",
        RuntimeWarning,
        stacklevel=2,
    )

    def njit(*args, **kwargs):
        if args and callable(args[0]):
            return args[0]

        def decorator(func):
            return func

        return decorator


SCRATCH = 1
DYNAMIC = 0
RESET_WRITE_WEIGHT = 0.1


@njit(cache=True)
def _simulate_candidate_work(
    indptr,
    indices,
    degree,
    candidate_residual,
    current_support,
    previous_support,
    changed_nodes,
    alpha,
    eps,
    include_history_seeds,
):
    """Return exact push/edge counts for a candidate residual."""
    num_nodes = len(degree)
    queue = np.empty(num_nodes + 1, dtype=np.int64)
    queued = np.zeros(num_nodes, dtype=np.bool_)
    front = 0
    rear = 0

    for idx in range(len(current_support)):
        u = current_support[idx]
        if (
            not queued[u]
            and abs(candidate_residual[u]) >= eps * degree[u]
        ):
            queue[rear] = u
            rear = (rear + 1) % (num_nodes + 1)
            queued[u] = True
    if include_history_seeds:
        for idx in range(len(previous_support)):
            u = previous_support[idx]
            if (
                not queued[u]
                and abs(candidate_residual[u]) >= eps * degree[u]
            ):
                queue[rear] = u
                rear = (rear + 1) % (num_nodes + 1)
                queued[u] = True
        for idx in range(len(changed_nodes)):
            u = changed_nodes[idx]
            if (
                not queued[u]
                and abs(candidate_residual[u]) >= eps * degree[u]
            ):
                queue[rear] = u
                rear = (rear + 1) % (num_nodes + 1)
                queued[u] = True

    pushes = 0
    edge_visits = 0
    while front != rear:
        u = queue[front]
        front = (front + 1) % (num_nodes + 1)
        queued[u] = False
        value = candidate_residual[u]
        if abs(value) < eps * degree[u]:
            continue
        pushes += 1
        candidate_residual[u] = 0.0
        pushed = alpha * value / degree[u]
        for edge_idx in range(indptr[u], indptr[u + 1]):
            edge_visits += 1
            v = indices[edge_idx]
            candidate_residual[v] += pushed
            if (
                not queued[v]
                and abs(candidate_residual[v]) >= eps * degree[v]
            ):
                queue[rear] = v
                rear = (rear + 1) % (num_nodes + 1)
                queued[v] = True
    return pushes, edge_visits


@njit(cache=True)
def predict_adaptive_branch(
    indptr,
    indices,
    degree,
    p,
    residual,
    previous_source,
    source,
    current_support,
    previous_support,
    changed_nodes,
    alpha,
    eps,
    can_continue,
    insertion_kind,
):
    """Measure both exact candidate residuals without mutating live state."""
    scratch_init_cost = float(
        2 * len(degree)
        + 3 * len(current_support)
        + len(previous_support)
    )
    scratch_candidate = np.zeros(len(degree), dtype=np.float64)
    scratch_edge_lb = 0
    for idx in range(len(current_support)):
        u = current_support[idx]
        value = source[u]
        scratch_candidate[u] = value
        threshold = eps * degree[u]
        if abs(value) >= threshold:
            scratch_edge_lb += degree[u]
    scratch_pushes, scratch_edges = _simulate_candidate_work(
        indptr,
        indices,
        degree,
        scratch_candidate,
        current_support,
        previous_support,
        changed_nodes,
        alpha,
        eps,
        False,
    )
    scratch_cost = (
        RESET_WRITE_WEIGHT * scratch_init_cost
        + scratch_pushes
        + scratch_edges
    )

    if not can_continue:
        return (
            SCRATCH,
            np.inf,
            scratch_cost,
            0,
            scratch_pushes,
            0,
            scratch_edge_lb,
            0,
            scratch_edges,
        )

    max_candidates = (
        len(current_support) + len(previous_support) + len(changed_nodes)
    )
    candidates = np.empty(max_candidates, dtype=np.int64)
    candidate_count = 0

    for idx in range(len(current_support)):
        u = current_support[idx]
        seen = False
        for j in range(candidate_count):
            if candidates[j] == u:
                seen = True
                break
        if not seen:
            candidates[candidate_count] = u
            candidate_count += 1
    for idx in range(len(previous_support)):
        u = previous_support[idx]
        seen = False
        for j in range(candidate_count):
            if candidates[j] == u:
                seen = True
                break
        if not seen:
            candidates[candidate_count] = u
            candidate_count += 1
    for idx in range(len(changed_nodes)):
        u = changed_nodes[idx]
        seen = False
        for j in range(candidate_count):
            if candidates[j] == u:
                seen = True
                break
        if not seen:
            candidates[candidate_count] = u
            candidate_count += 1

    insert_a = -1
    insert_b = -1
    correction_aa = 0.0
    correction_ab = 0.0
    correction_ba = 0.0
    correction_bb = 0.0
    if insertion_kind != 0:
        insert_a = int(changed_nodes[0])
        insert_b = int(changed_nodes[1])
        old_degree_a = degree[insert_a] - 1.0
        mass_a = p[insert_a] / old_degree_a
        scale = 1.0 / (1.0 - alpha)
        correction_aa = -mass_a * scale
        correction_ab = alpha * mass_a * scale
        if insertion_kind == 2:
            old_degree_b = degree[insert_b] - 1.0
            mass_b = p[insert_b] / old_degree_b
            correction_ba = alpha * mass_b * scale
            correction_bb = -mass_b * scale

    dynamic_init_cost = float(
        candidate_count
        + 2 * (len(current_support) + len(previous_support))
        + len(changed_nodes)
    )
    dynamic_candidate = residual.copy()
    dynamic_edge_lb = 0
    for idx in range(candidate_count):
        u = candidates[idx]
        value = residual[u] + source[u] - previous_source[u]
        if u == insert_a:
            value += correction_aa + correction_ba
        elif u == insert_b:
            value += correction_ab + correction_bb
        dynamic_candidate[u] = value
        threshold = eps * degree[u]
        if abs(value) >= threshold:
            dynamic_edge_lb += degree[u]
    dynamic_pushes, dynamic_edges = _simulate_candidate_work(
        indptr,
        indices,
        degree,
        dynamic_candidate,
        current_support,
        previous_support,
        changed_nodes,
        alpha,
        eps,
        True,
    )
    dynamic_cost = dynamic_init_cost + dynamic_pushes + dynamic_edges

    mode = DYNAMIC if dynamic_cost < scratch_cost else SCRATCH
    return (
        mode,
        dynamic_cost,
        scratch_cost,
        dynamic_pushes,
        scratch_pushes,
        dynamic_edge_lb,
        scratch_edge_lb,
        dynamic_edges,
        scratch_edges,
    )


@njit(cache=True)
def execute_selected_branch(
    mode,
    indptr,
    indices,
    degree,
    p,
    residual,
    previous_source,
    source,
    current_support,
    previous_support,
    changed_nodes,
    alpha,
    eps,
    queue,
    queued,
    insertion_kind,
):
    """Apply the selected transition and local pushes in one compiled kernel."""
    num_nodes = len(p)
    if mode == SCRATCH:
        p.fill(0.0)
        residual.fill(0.0)
        for idx in range(len(current_support)):
            u = current_support[idx]
            residual[u] = source[u]
    else:
        if insertion_kind != 0:
            a = int(changed_nodes[0])
            b = int(changed_nodes[1])
            old_degree_a = degree[a] - 1.0
            mass_a = p[a] / old_degree_a
            p[a] *= degree[a] / old_degree_a
            scale = 1.0 / (1.0 - alpha)
            residual[a] -= mass_a * scale
            residual[b] += alpha * mass_a * scale
            if insertion_kind == 2:
                old_degree_b = degree[b] - 1.0
                mass_b = p[b] / old_degree_b
                p[b] *= degree[b] / old_degree_b
                residual[b] -= mass_b * scale
                residual[a] += alpha * mass_b * scale

        i = 0
        j = 0
        while i < len(current_support) or j < len(previous_support):
            if j >= len(previous_support) or (
                i < len(current_support)
                and current_support[i] < previous_support[j]
            ):
                u = current_support[i]
                i += 1
            elif (
                i >= len(current_support)
                or previous_support[j] < current_support[i]
            ):
                u = previous_support[j]
                j += 1
            else:
                u = current_support[i]
                i += 1
                j += 1
            residual[u] += source[u] - previous_source[u]

    front = 0
    rear = 0
    push_count = 0
    edge_visits = 0
    initial_active = 0

    if mode == SCRATCH:
        for idx in range(len(current_support)):
            u = current_support[idx]
            if not queued[u] and abs(residual[u]) >= eps * degree[u]:
                queue[rear] = u
                rear = (rear + 1) % (num_nodes + 1)
                queued[u] = True
                initial_active += 1
    else:
        for idx in range(len(current_support)):
            u = current_support[idx]
            if not queued[u] and abs(residual[u]) >= eps * degree[u]:
                queue[rear] = u
                rear = (rear + 1) % (num_nodes + 1)
                queued[u] = True
                initial_active += 1
        for idx in range(len(previous_support)):
            u = previous_support[idx]
            if not queued[u] and abs(residual[u]) >= eps * degree[u]:
                queue[rear] = u
                rear = (rear + 1) % (num_nodes + 1)
                queued[u] = True
                initial_active += 1
        for idx in range(len(changed_nodes)):
            u = changed_nodes[idx]
            if not queued[u] and abs(residual[u]) >= eps * degree[u]:
                queue[rear] = u
                rear = (rear + 1) % (num_nodes + 1)
                queued[u] = True
                initial_active += 1

    while front != rear:
        u = queue[front]
        front = (front + 1) % (num_nodes + 1)
        queued[u] = False
        value = residual[u]
        if abs(value) < eps * degree[u]:
            continue
        push_count += 1
        p[u] += (1.0 - alpha) * value
        residual[u] = 0.0
        pushed = alpha * value / degree[u]
        for edge_idx in range(indptr[u], indptr[u + 1]):
            edge_visits += 1
            v = indices[edge_idx]
            residual[v] += pushed
            if not queued[v] and abs(residual[v]) >= eps * degree[v]:
                queue[rear] = v
                rear = (rear + 1) % (num_nodes + 1)
                queued[v] = True

    for idx in range(len(previous_support)):
        previous_source[previous_support[idx]] = 0.0
    for idx in range(len(current_support)):
        u = current_support[idx]
        previous_source[u] = source[u]

    return push_count, edge_visits, initial_active


def resolve_adaptive_kernels(backend):
    if backend == "auto":
        backend = "numba" if NUMBA_AVAILABLE else "python"
    if backend == "numba":
        if not NUMBA_AVAILABLE:
            raise RuntimeError(
                "The numba adaptive backend was requested, but numba is unavailable"
            )
        return predict_adaptive_branch, execute_selected_branch
    if backend == "python":
        return (
            getattr(predict_adaptive_branch, "py_func", predict_adaptive_branch),
            getattr(execute_selected_branch, "py_func", execute_selected_branch),
        )
    raise ValueError(f"Unknown adaptive backend: {backend!r}")


class AdaptiveAPPR:
    """State holder around the shared reversible adaptive kernels."""

    def __init__(self, backend="auto"):
        self.backend = backend
        self.predict_impl, self.execute_impl = resolve_adaptive_kernels(backend)
        self.reset()

    def reset(self):
        self.p = None
        self.r = None
        self.previous_source = None
        self.previous_support = np.empty(0, dtype=np.int64)
        self.degree = None
        self.nnz = None
        self.alpha = None
        self.queue = None
        self.queued = None
        self.last_prediction = None
        self.last_execution = None
        self.stats = {
            "solves": 0,
            "scratch_resets": 0,
            "dynamic_continuations": 0,
            "cold_starts": 0,
            "invalid_graph_resets": 0,
            "scratch_pushes": 0,
            "dynamic_pushes": 0,
            "scratch_edge_visits": 0,
            "dynamic_edge_visits": 0,
            "scratch_initial_active_nodes": 0,
            "dynamic_initial_active_nodes": 0,
            "predicted_dynamic_cost_sum": 0.0,
            "predicted_scratch_cost_sum": 0.0,
            "predicted_dynamic_pushes_sum": 0,
            "predicted_scratch_pushes_sum": 0,
            "predicted_dynamic_edge_lb_sum": 0,
            "predicted_scratch_edge_lb_sum": 0,
            "predicted_dynamic_edges_sum": 0,
            "predicted_scratch_edges_sum": 0,
        }

    def _ensure_workspace(self, num_nodes):
        if self.p is None or len(self.p) != num_nodes:
            self.p = np.zeros(num_nodes, dtype=np.float64)
            self.r = np.zeros(num_nodes, dtype=np.float64)
            self.previous_source = np.zeros(num_nodes, dtype=np.float64)
            self.queue = np.zeros(num_nodes + 1, dtype=np.int64)
            self.queued = np.zeros(num_nodes, dtype=np.bool_)

    def _resolve_change(self, degree, nnz, changed_nodes_hint):
        if self.nnz is None:
            return np.empty(0, dtype=np.int64), 0, False
        graph_changed = nnz != self.nnz
        if changed_nodes_hint is None:
            degree_delta = degree - self.degree
            changed = np.flatnonzero(degree_delta).astype(np.int64)
        else:
            changed = np.asarray(changed_nodes_hint, dtype=np.int64)
        insertion_kind = 0
        if graph_changed and len(changed) == 2:
            delta = degree[changed] - self.degree[changed]
            if (
                nnz - self.nnz == 2
                and np.all(delta == 1)
                and np.all(degree[changed] > 1)
            ):
                insertion_kind = 2
            elif (
                nnz - self.nnz == 1
                and np.count_nonzero(delta == 1) == 1
                and np.count_nonzero(delta == 0) == 1
            ):
                source_pos = int(np.flatnonzero(delta == 1)[0])
                target_pos = 1 - source_pos
                if degree[changed[source_pos]] > 1:
                    changed = np.asarray(
                        [changed[source_pos], changed[target_pos]],
                        dtype=np.int64,
                    )
                    insertion_kind = 1
        invalid = (
            (graph_changed and insertion_kind == 0)
            or (not graph_changed and len(changed) != 0)
        )
        return changed, insertion_kind, invalid

    def predict(
        self,
        num_nodes,
        indptr,
        indices,
        degree,
        source,
        alpha,
        eps,
        current_support,
        changed_nodes_hint,
        force_scratch=False,
    ):
        self._ensure_workspace(num_nodes)
        degree = np.asarray(degree)
        source = np.asarray(source, dtype=np.float64)
        current_support = np.asarray(current_support, dtype=np.int64)
        nnz = int(indptr[-1])
        cold = (
            self.nnz is None
            or self.alpha != alpha
            or self.degree is None
            or len(self.degree) != num_nodes
        )
        changed, insertion_kind, invalid = self._resolve_change(
            degree, nnz, changed_nodes_hint
        )
        can_continue = not cold and not invalid and not force_scratch
        prediction = self.predict_impl(
            indptr,
            indices,
            degree,
            self.p,
            self.r,
            self.previous_source,
            source,
            current_support,
            self.previous_support,
            changed if insertion_kind else np.empty(0, dtype=np.int64),
            alpha,
            eps,
            can_continue,
            insertion_kind,
        )
        mode = SCRATCH if force_scratch or cold or invalid else int(prediction[0])
        self.last_prediction = {
            "mode": mode,
            "dynamic_cost": float(prediction[1]),
            "scratch_cost": float(prediction[2]),
            "dynamic_predicted_pushes": int(prediction[3]),
            "scratch_predicted_pushes": int(prediction[4]),
            "dynamic_edge_lb": int(prediction[5]),
            "scratch_edge_lb": int(prediction[6]),
            "dynamic_predicted_edges": int(prediction[7]),
            "scratch_predicted_edges": int(prediction[8]),
            "cold_start": bool(cold),
            "invalid_graph": bool(invalid),
            "changed_nodes": changed if insertion_kind else np.empty(0, dtype=np.int64),
            "insertion_kind": insertion_kind,
            "nnz": nnz,
        }
        return self.last_prediction

    def scratch_prediction(self, num_nodes, indptr):
        """Build a forced-scratch transition without running the DYN predictor."""
        self._ensure_workspace(num_nodes)
        return {
            "mode": SCRATCH,
            "dynamic_cost": 0.0,
            "scratch_cost": 0.0,
            "dynamic_active": 0,
            "scratch_active": 0,
            "dynamic_predicted_pushes": 0,
            "scratch_predicted_pushes": 0,
            "dynamic_edge_lb": 0,
            "scratch_edge_lb": 0,
            "dynamic_predicted_edges": 0,
            "scratch_predicted_edges": 0,
            "cold_start": self.nnz is None,
            "invalid_graph": False,
            "changed_nodes": np.empty(0, dtype=np.int64),
            "insertion_kind": 0,
            "nnz": int(indptr[-1]),
        }

    def execute(
        self,
        indptr,
        indices,
        degree,
        source,
        alpha,
        eps,
        current_support,
        prediction,
    ):
        degree = np.asarray(degree)
        source = np.asarray(source, dtype=np.float64)
        current_support = np.asarray(current_support, dtype=np.int64)
        mode = int(prediction["mode"])
        changed = np.asarray(prediction["changed_nodes"], dtype=np.int64)
        pushes, edge_visits, initial_active = self.execute_impl(
            mode,
            indptr,
            indices,
            degree,
            self.p,
            self.r,
            self.previous_source,
            source,
            current_support,
            self.previous_support,
            changed,
            alpha,
            eps,
            self.queue,
            self.queued,
            int(prediction["insertion_kind"]),
        )
        self.previous_support = current_support.copy()
        if mode == SCRATCH or self.degree is None or prediction["invalid_graph"]:
            self.degree = degree.copy()
        elif len(changed):
            self.degree[changed] = degree[changed]
        elif prediction["cold_start"]:
            self.degree = degree.copy()
        self.nnz = int(prediction["nnz"])
        self.alpha = alpha
        self.last_execution = {
            "mode": mode,
            "pushes": int(pushes),
            "edge_visits": int(edge_visits),
            "initial_active_nodes": int(initial_active),
        }
        self.stats["solves"] += 1
        self.stats["cold_starts"] += int(prediction["cold_start"])
        self.stats["invalid_graph_resets"] += int(prediction["invalid_graph"])
        if np.isfinite(prediction["dynamic_cost"]):
            self.stats["predicted_dynamic_cost_sum"] += prediction["dynamic_cost"]
        if np.isfinite(prediction["scratch_cost"]):
            self.stats["predicted_scratch_cost_sum"] += prediction["scratch_cost"]
        self.stats["predicted_dynamic_pushes_sum"] += prediction[
            "dynamic_predicted_pushes"
        ]
        self.stats["predicted_scratch_pushes_sum"] += prediction[
            "scratch_predicted_pushes"
        ]
        self.stats["predicted_dynamic_edge_lb_sum"] += prediction["dynamic_edge_lb"]
        self.stats["predicted_scratch_edge_lb_sum"] += prediction["scratch_edge_lb"]
        self.stats["predicted_dynamic_edges_sum"] += prediction[
            "dynamic_predicted_edges"
        ]
        self.stats["predicted_scratch_edges_sum"] += prediction[
            "scratch_predicted_edges"
        ]
        if mode == SCRATCH:
            self.stats["scratch_resets"] += 1
            self.stats["scratch_pushes"] += int(pushes)
            self.stats["scratch_edge_visits"] += int(edge_visits)
            self.stats["scratch_initial_active_nodes"] += int(initial_active)
        else:
            self.stats["dynamic_continuations"] += 1
            self.stats["dynamic_pushes"] += int(pushes)
            self.stats["dynamic_edge_visits"] += int(edge_visits)
            self.stats["dynamic_initial_active_nodes"] += int(initial_active)
        return self.p

    def solve(
        self,
        num_nodes,
        indptr,
        indices,
        degree,
        source,
        alpha,
        eps,
        current_support,
        changed_nodes_hint,
        force_scratch=False,
    ):
        prediction = self.predict(
            num_nodes,
            indptr,
            indices,
            degree,
            source,
            alpha,
            eps,
            current_support,
            changed_nodes_hint,
            force_scratch=force_scratch,
        )
        return self.execute(
            indptr,
            indices,
            degree,
            source,
            alpha,
            eps,
            current_support,
            prediction,
        )

    def diagnose_branches(
        self,
        indptr,
        indices,
        degree,
        source,
        alpha,
        eps,
        current_support,
        prediction,
    ):
        """Execute both branches on cloned state; caller excludes this time."""
        outputs = {}
        for name, mode in (("dynamic", DYNAMIC), ("scratch", SCRATCH)):
            if mode == DYNAMIC and (
                prediction["cold_start"] or prediction["invalid_graph"]
            ):
                continue
            p = self.p.copy()
            r = self.r.copy()
            previous_source = self.previous_source.copy()
            queue = np.zeros_like(self.queue)
            queued = np.zeros_like(self.queued)
            t0 = time.perf_counter()
            stats = self.execute_impl(
                mode,
                indptr,
                indices,
                degree,
                p,
                r,
                previous_source,
                source,
                current_support,
                self.previous_support,
                prediction["changed_nodes"],
                alpha,
                eps,
                queue,
                queued,
                int(prediction["insertion_kind"]),
            )
            outputs[name] = {
                "p": p,
                "r": r,
                "pushes": int(stats[0]),
                "edge_visits": int(stats[1]),
                "initial_active_nodes": int(stats[2]),
                "time": time.perf_counter() - t0,
            }
        return outputs
