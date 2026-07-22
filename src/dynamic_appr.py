"""Stateful dynamic APPR used by the online LocPRB runner."""

import numpy as np
import warnings

try:
    from numba import njit
    NUMBA_AVAILABLE = True
except ModuleNotFoundError:
    NUMBA_AVAILABLE = False
    warnings.warn(
        "Numba is unavailable; DYN-APPR is using the much slower Python "
        "push kernel.",
        RuntimeWarning,
        stacklevel=2,
    )

    def njit(*args, **kwargs):
        if args and callable(args[0]):
            return args[0]

        def decorator(func):
            return func

        return decorator


@njit(cache=True)
def appr_push(
    indptr, indices, degree, p, residual, alpha, eps, seed_nodes=None
):
    """Continue local pushes from an existing ``(p, residual)`` state."""
    num_nodes = len(p)
    queue = np.zeros(num_nodes + 1, dtype=np.int64)
    queued = np.zeros(num_nodes, dtype=np.bool_)
    front = 0
    rear = 0
    push_count = 0
    edge_visits = 0
    initial_active = 0

    if seed_nodes is None:
        for u in range(num_nodes):
            if abs(residual[u]) >= eps * degree[u]:
                queue[rear] = u
                rear = (rear + 1) % (num_nodes + 1)
                queued[u] = True
                initial_active += 1
    else:
        for seed_idx in range(len(seed_nodes)):
            u = seed_nodes[seed_idx]
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

    return push_count, edge_visits, initial_active


@njit(cache=True)
def source_pressure_reset(
    current_support, previous_support, source, previous_source, degree, eps
):
    """Compare sparse dynamic-delta pressure with fresh-source pressure."""
    scratch_pressure = 0.0
    for idx in range(len(current_support)):
        u = current_support[idx]
        scratch_pressure += abs(source[u]) / (eps * degree[u])

    delta_pressure = 0.0
    i = 0
    j = 0
    while i < len(current_support) or j < len(previous_support):
        if j >= len(previous_support) or (
            i < len(current_support)
            and current_support[i] < previous_support[j]
        ):
            u = current_support[i]
            i += 1
        elif i >= len(current_support) or previous_support[j] < current_support[i]:
            u = previous_support[j]
            j += 1
        else:
            u = current_support[i]
            i += 1
            j += 1
        delta_pressure += abs(source[u] - previous_source[u]) / (
            eps * degree[u]
        )
    return (
        scratch_pressure > 0.0 and delta_pressure >= scratch_pressure,
        delta_pressure,
        scratch_pressure,
    )


def get_push_impl(backend="auto"):
    """Resolve the DYN push kernel without maintaining two implementations."""
    if backend == "auto":
        backend = "numba" if NUMBA_AVAILABLE else "python"
    if backend == "numba":
        if not NUMBA_AVAILABLE:
            raise RuntimeError(
                "The numba PPR backend was requested, but numba is unavailable"
            )
        return appr_push
    if backend == "python":
        return getattr(appr_push, "py_func", appr_push)
    raise ValueError(f"Unknown PPR backend: {backend!r}")


class DynamicAPPR:
    """Maintain APPR state across source changes and edge insertions.

    The maintained state satisfies

        (I - alpha P) p + (1 - alpha) r = (1 - alpha) s.

    Source changes are applied sparsely to ``r``. One undirected edge insertion
    between consecutive solves is handled by the endpoint INSERTUPDATE repair.
    """

    def __init__(
        self, push_impl=None, scratch_impl=None, diagnostics=False,
        auto_record=True,
    ):
        self.push_impl = appr_push if push_impl is None else push_impl
        if scratch_impl is None:
            from .ppr_solver import get_scratch_into_kernel

            python_push = getattr(appr_push, "py_func", None)
            backend = (
                "python"
                if not NUMBA_AVAILABLE or self.push_impl is python_push
                else "numba"
            )
            scratch_impl = get_scratch_into_kernel(backend)
        self.scratch_impl = scratch_impl
        self.diagnostics = diagnostics
        self.auto_record = auto_record
        self.stats = {
            "solves": 0,
            "initializations": 0,
            "fallback_resets": 0,
            "insert_updates": 0,
            "source_updates": 0,
            "pushes": 0,
            "edge_visits": 0,
            "initial_active_nodes": 0,
            "adaptive_resets": 0,
            "dynamic_continuations": 0,
            "adaptive_reset_pushes": 0,
            "fast_reset_rounds": 0,
        }
        self.last_stats = {}
        self._pending_stats = None
        self.reset()

    def reset(self):
        self.p = None
        self.r = None
        self.source = None
        self.source_support = None
        self.degree = None
        self.alpha = None
        self.num_nodes = None
        self.nnz = None
        self.queue = None
        self.queued = None
        self.consecutive_adaptive_resets = 0
        self.fast_reset_locked = False

    def _insert_one_direction(self, u, v, new_degree, alpha):
        old_degree = new_degree - 1.0
        if old_degree <= 0.0:
            raise ValueError(
                "DYN-APPR requires positive pre-insertion endpoint degrees"
            )

        mass_per_old_edge = self.p[u] / old_degree
        self.p[u] *= new_degree / old_degree
        self.r[u] -= mass_per_old_edge / (1.0 - alpha)
        self.r[v] += alpha * mass_per_old_edge / (1.0 - alpha)

    def _insert_update(self, changed_nodes, degree, alpha):
        if len(changed_nodes) != 2:
            raise ValueError(
                "DYN-APPR expects exactly two endpoints for one undirected "
                f"edge insertion; changed nodes: {changed_nodes.tolist()}"
            )
        a, b = int(changed_nodes[0]), int(changed_nodes[1])
        self._insert_one_direction(a, b, degree[a], alpha)
        self._insert_one_direction(b, a, degree[b], alpha)

    def solve(
        self,
        num_nodes,
        indptr,
        indices,
        degree,
        source,
        alpha,
        eps,
        source_indices=None,
        changed_nodes_hint=None,
    ):
        # Keep the caller's integer degree view. Converting the full vector to
        # float64 every round is unnecessary: divisions below already promote
        # endpoint values, and the push kernel accepts integer thresholds.
        degree = np.asarray(degree)
        source = np.asarray(source, dtype=np.float64)
        nnz = int(indptr[-1])

        cold_start = (
            self.p is None
            or self.num_nodes != num_nodes
            or self.alpha != alpha
        )
        initialize = cold_start
        fallback_reset = False
        changed_nodes = np.empty(0, dtype=np.int64)
        graph_changed = not initialize and nnz != self.nnz
        valid_insert = False
        invalid_change = False

        if source_indices is None:
            current_support = np.flatnonzero(source != 0.0)
        else:
            current_support = np.asarray(source_indices, dtype=np.int64)

        adaptive_reset = False
        fast_reset = False
        delta_pressure = 0.0
        scratch_pressure = 0.0
        if not initialize:
            fast_reset = self.fast_reset_locked
            if fast_reset:
                adaptive_reset = True
            else:
                (
                    adaptive_reset,
                    delta_pressure,
                    scratch_pressure,
                ) = source_pressure_reset(
                    current_support,
                    self.source_support,
                    source,
                    self.source,
                    degree,
                    eps,
                )

        reset_degree_full = bool(adaptive_reset and graph_changed)
        if not initialize and adaptive_reset:
            if changed_nodes_hint is not None:
                changed_nodes = np.asarray(changed_nodes_hint, dtype=np.int64)
                reset_degree_full = graph_changed and len(changed_nodes) != 2
        elif not initialize:
            if changed_nodes_hint is None:
                degree_delta = degree - self.degree
                changed_nodes = np.flatnonzero(degree_delta)
                valid_insert = (
                    graph_changed
                    and nnz - self.nnz == 2
                    and len(changed_nodes) == 2
                    and np.all(degree_delta[changed_nodes] == 1)
                    and np.all(degree[changed_nodes] > 1)
                )
                invalid_change = (
                    np.any(degree_delta < 0)
                    or np.any(degree_delta > 1)
                    or (graph_changed and not valid_insert)
                    or (not graph_changed and len(changed_nodes) != 0)
                )
            else:
                changed_nodes = np.asarray(changed_nodes_hint, dtype=np.int64)
                valid_insert = (
                    graph_changed
                    and nnz - self.nnz == 2
                    and len(changed_nodes) == 2
                    and np.all(
                        degree[changed_nodes] - self.degree[changed_nodes] == 1
                    )
                    and np.all(degree[changed_nodes] > 1)
                )
                invalid_change = (
                    (graph_changed and not valid_insert)
                    or (not graph_changed and len(changed_nodes) != 0)
                )
            if invalid_change:
                initialize = True
                fallback_reset = True

        if initialize or adaptive_reset:
            if self.p is None or len(self.p) != num_nodes:
                self.p = np.zeros(num_nodes, dtype=np.float64)
                self.r = np.zeros(num_nodes, dtype=np.float64)
                self.queue = np.zeros(num_nodes + 1, dtype=np.int64)
                self.queued = np.zeros(num_nodes + 1, dtype=np.bool_)
            scratch_result = self.scratch_impl(
                indptr,
                indices,
                degree,
                source,
                alpha,
                eps,
                current_support,
                self.p,
                self.r,
                self.queue,
                self.queued,
                True,
                True,
            )
            push_count, edge_visits, initial_active = scratch_result[1:4]
        else:
            if len(changed_nodes):
                self._insert_update(changed_nodes, degree, alpha)

            source_support = np.union1d(
                current_support, self.source_support
            ).astype(np.int64)
            self.r[source_support] += (
                source[source_support] - self.source[source_support]
            )
            active_seed = np.union1d(source_support, changed_nodes).astype(
                np.int64
            )
            push_count, edge_visits, initial_active = self.push_impl(
                indptr, indices, degree, self.p, self.r, alpha, eps, active_seed
            )

        # During a forced reset window the dynamic source state is unused.
        # Refresh it only on the final forced round so the next pressure probe
        # still compares against the immediately preceding source.
        refresh_source_state = not fast_reset
        if refresh_source_state:
            if self.source is None or len(self.source) != num_nodes:
                self.source = np.zeros(num_nodes, dtype=np.float64)
            elif self.source_support is not None:
                self.source[self.source_support] = 0.0
            self.source[current_support] = source[current_support]
            self.source_support = current_support.copy()

        source_changed = int(
            len(current_support)
            if initialize or adaptive_reset
            else len(source_support)
        )
        self.alpha = alpha
        self.num_nodes = num_nodes
        if initialize or self.degree is None or fallback_reset or reset_degree_full:
            self.degree = degree.copy()
        elif len(changed_nodes):
            self.degree[changed_nodes] = degree[changed_nodes]
        self.nnz = nnz
        inserted = int(
            not initialize and not adaptive_reset and len(changed_nodes) == 2
        )
        if adaptive_reset:
            self.consecutive_adaptive_resets += 1
            if (
                not fast_reset
                and self.consecutive_adaptive_resets >= 3
            ):
                self.fast_reset_locked = True
        else:
            self.consecutive_adaptive_resets = 0
        self._pending_stats = (
            cold_start,
            initialize,
            fallback_reset,
            adaptive_reset,
            fast_reset,
            delta_pressure,
            scratch_pressure,
            inserted,
            source_changed,
            int(initial_active),
            int(push_count),
            int(edge_visits),
        )
        if self.auto_record:
            self.record_last_stats()
        return self.p

    def record_last_stats(self):
        """Commit counters outside the solver timing when requested."""
        if self._pending_stats is None:
            return
        (
            cold_start,
            initialize,
            fallback_reset,
            adaptive_reset,
            fast_reset,
            delta_pressure,
            scratch_pressure,
            inserted,
            source_changed,
            initial_active,
            push_count,
            edge_visits,
        ) = self._pending_stats
        self.last_stats = {
            "cold_start": bool(cold_start),
            "fallback_reset": bool(fallback_reset),
            "adaptive_reset": bool(adaptive_reset),
            "fast_reset": bool(fast_reset),
            "delta_pressure": delta_pressure,
            "scratch_pressure": scratch_pressure,
            "insert_update": bool(inserted),
            "source_changed_nodes": source_changed,
            "initial_active_nodes": initial_active,
            "pushes": push_count,
            "edge_visits": edge_visits,
        }
        if self.diagnostics:
            self.last_stats.update(
                p_l1=float(np.sum(np.abs(self.p))),
                residual_l1=float(np.sum(np.abs(self.r))),
            )
        self.stats["solves"] += 1
        self.stats["initializations"] += int(initialize)
        self.stats["fallback_resets"] += int(fallback_reset)
        self.stats["insert_updates"] += inserted
        self.stats["source_updates"] += source_changed
        self.stats["pushes"] += int(push_count)
        self.stats["edge_visits"] += int(edge_visits)
        self.stats["initial_active_nodes"] += int(initial_active)
        self.stats["adaptive_resets"] += int(adaptive_reset)
        self.stats["dynamic_continuations"] += int(
            not initialize and not adaptive_reset
        )
        self.stats["adaptive_reset_pushes"] += int(
            push_count if adaptive_reset else 0
        )
        self.stats["fast_reset_rounds"] += int(fast_reset)
        self._pending_stats = None

    def record_external_fast_reset(
        self, source_changed, initial_active, push_count, edge_visits
    ):
        """Record a locked reusable-scratch solve run directly by the caller."""
        self._pending_stats = (
            False,
            False,
            False,
            True,
            True,
            0.0,
            0.0,
            0,
            int(source_changed),
            int(initial_active),
            int(push_count),
            int(edge_visits),
        )
        self.record_last_stats()
