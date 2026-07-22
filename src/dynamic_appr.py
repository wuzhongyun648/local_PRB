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

    def __init__(self, push_impl=None, diagnostics=False):
        self.push_impl = appr_push if push_impl is None else push_impl
        self.diagnostics = diagnostics
        self.stats = {
            "solves": 0,
            "initializations": 0,
            "fallback_resets": 0,
            "insert_updates": 0,
            "source_updates": 0,
            "pushes": 0,
            "edge_visits": 0,
            "initial_active_nodes": 0,
        }
        self.last_stats = {}
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
        if not initialize:
            graph_changed = nnz != self.nnz
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

        if source_indices is None:
            current_support = np.flatnonzero(source != 0.0)
        else:
            current_support = np.asarray(source_indices, dtype=np.int64)

        if initialize:
            self.p = np.zeros(num_nodes, dtype=np.float64)
            self.r = source.copy()
            self.source = source.copy()
            self.source_support = current_support.copy()
            active_seed = current_support
        else:
            if len(changed_nodes):
                self._insert_update(changed_nodes, degree, alpha)

            source_support = np.union1d(
                current_support, self.source_support
            ).astype(np.int64)
            self.r[source_support] += (
                source[source_support] - self.source[source_support]
            )
            self.source[self.source_support] = 0.0
            self.source[current_support] = source[current_support]
            self.source_support = current_support.copy()
            active_seed = np.union1d(source_support, changed_nodes).astype(
                np.int64
            )

        source_changed = int(
            np.count_nonzero(source)
            if initialize
            else len(source_support)
        )
        push_count, edge_visits, initial_active = self.push_impl(
            indptr, indices, degree, self.p, self.r, alpha, eps, active_seed
        )
        self.alpha = alpha
        self.num_nodes = num_nodes
        if initialize or self.degree is None:
            self.degree = degree.copy()
        elif len(changed_nodes):
            self.degree[changed_nodes] = degree[changed_nodes]
        self.nnz = nnz
        inserted = int(not initialize and len(changed_nodes) == 2)
        self.last_stats = {
            "cold_start": bool(cold_start),
            "fallback_reset": bool(fallback_reset),
            "insert_update": bool(inserted),
            "source_changed_nodes": source_changed,
            "initial_active_nodes": int(initial_active),
            "pushes": int(push_count),
            "edge_visits": int(edge_visits),
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
        return self.p
