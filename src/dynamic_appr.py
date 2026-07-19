"""Stateful dynamic APPR used by the online LocPRB runner."""

import numpy as np


def appr_push(indptr, indices, degree, p, residual, alpha, eps):
    """Continue local pushes from an existing ``(p, residual)`` state."""
    num_nodes = len(p)
    queue = np.zeros(num_nodes + 1, dtype=np.int64)
    queued = np.zeros(num_nodes, dtype=np.bool_)
    front = 0
    rear = 0
    push_count = 0

    for u in range(num_nodes):
        if abs(residual[u]) >= eps * degree[u]:
            queue[rear] = u
            rear = (rear + 1) % (num_nodes + 1)
            queued[u] = True

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
            v = indices[edge_idx]
            residual[v] += pushed
            if not queued[v] and abs(residual[v]) >= eps * degree[v]:
                queue[rear] = v
                rear = (rear + 1) % (num_nodes + 1)
                queued[v] = True
    return push_count


class DynamicAPPR:
    """Maintain APPR state across source changes and edge insertions.

    The maintained state satisfies

        (I - alpha P) p + (1 - alpha) r = (1 - alpha) s.

    Source changes are applied sparsely to ``r``. One undirected edge insertion
    between consecutive solves is handled by the endpoint INSERTUPDATE repair.
    """

    def __init__(self):
        self.stats = {
            "solves": 0,
            "initializations": 0,
            "fallback_resets": 0,
            "insert_updates": 0,
            "source_updates": 0,
            "pushes": 0,
        }
        self.last_stats = {}
        self.reset()

    def reset(self):
        self.p = None
        self.r = None
        self.source = None
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

    def solve(self, num_nodes, indptr, indices, degree, source, alpha, eps):
        degree = np.asarray(degree, dtype=np.float64)
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
            degree_delta = degree - self.degree
            changed_nodes = np.flatnonzero(degree_delta)
            graph_changed = nnz != self.nnz
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
            if invalid_change:
                initialize = True
                fallback_reset = True

        if initialize:
            self.p = np.zeros(num_nodes, dtype=np.float64)
            self.r = source.copy()
            self.source = source.copy()
        else:
            if len(changed_nodes):
                self._insert_update(changed_nodes, degree, alpha)

            source_support = np.flatnonzero(source != self.source)
            self.r[source_support] += (
                source[source_support] - self.source[source_support]
            )
            self.source.fill(0.0)
            self.source[source_support] = source[source_support]

        source_changed = int(
            np.count_nonzero(source)
            if initialize
            else len(source_support)
        )
        push_count = appr_push(
            indptr, indices, degree, self.p, self.r, alpha, eps
        )
        self.alpha = alpha
        self.num_nodes = num_nodes
        self.degree = degree.copy()
        self.nnz = nnz
        inserted = int(not initialize and len(changed_nodes) == 2)
        self.last_stats = {
            "cold_start": bool(cold_start),
            "fallback_reset": bool(fallback_reset),
            "insert_update": bool(inserted),
            "source_changed_nodes": source_changed,
            "pushes": int(push_count),
        }
        self.stats["solves"] += 1
        self.stats["initializations"] += int(initialize)
        self.stats["fallback_resets"] += int(fallback_reset)
        self.stats["insert_updates"] += inserted
        self.stats["source_updates"] += source_changed
        self.stats["pushes"] += int(push_count)
        return self.p.copy()
