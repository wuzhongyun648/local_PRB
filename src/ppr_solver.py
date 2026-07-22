import numpy as np
import scipy.sparse as sp
import time
import warnings

try:
    from numba import njit
    NUMBA_AVAILABLE = True
except ModuleNotFoundError:
    NUMBA_AVAILABLE = False
    warnings.warn(
        "Numba is unavailable; LocPRB is using the much slower Python "
        "APPR kernel.",
        RuntimeWarning,
        stacklevel=2,
    )

    def njit(*args, **kwargs):
        if args and callable(args[0]):
            return args[0]
        def decorator(func):
            return func
        return decorator


def power_iteration(P: sp.spmatrix, alpha: float, h: np.ndarray, t: int) -> np.ndarray:
    v = (1 - alpha) * h
    for _ in range(t):
        v = alpha * (P @ v) + (1 - alpha) * h
    return v

@njit(cache=True)
def _appr_diagnostics(
    num_nodes, indptr, indices, degree, h, alpha, eps, seed_nodes=None
):
    front = 0
    rear = 0
    queue = np.zeros(num_nodes + 1,dtype = np.int64)
    q_mark = np.zeros(num_nodes + 1, dtype = np.bool_)
    p = np.zeros(num_nodes)
    r = np.zeros(num_nodes)
    push_count = 0
    edge_visits = 0
    initial_active = 0
    
    if seed_nodes is None:
        for idx in range(num_nodes):
            val = h[idx]
            r[idx] = val
            if eps * degree[idx] <= np.abs(val):
                queue[rear] = idx
                rear = (rear + 1) % (num_nodes + 1)
                q_mark[idx] = True
                initial_active += 1
    else:
        for seed_idx in range(len(seed_nodes)):
            idx = seed_nodes[seed_idx]
            val = h[idx]
            r[idx] = val
            if not q_mark[idx] and eps * degree[idx] <= np.abs(val):
                queue[rear] = idx
                rear = (rear + 1) % (num_nodes + 1)
                q_mark[idx] = True
                initial_active += 1
    
    while (rear - front) != 0: 
        u = queue[front]
        front = (front + 1) % (num_nodes + 1)
        q_mark[u] = False
        r_val = r[u]
        if eps * degree[u] > np.abs(r[u]):
            continue
        push_count += 1
        p[u] += r_val * (1. - alpha) 
        r[u] = 0.0
        push_val = alpha * r_val / degree[u]
        for v in indices[indptr[u]:indptr[u + 1]]:
            edge_visits += 1
            r[v] += push_val
            if not q_mark[v] and eps * degree[v] <= np.abs(r[v]):
                queue[rear] = v
                rear = (rear + 1) % (num_nodes + 1)
                q_mark[v] = True
        
    return p, push_count, edge_visits, initial_active


def get_appr_kernel(backend="auto"):
    """Resolve the scratch APPR kernel without duplicating its algorithm."""
    if backend == "auto":
        backend = "numba" if NUMBA_AVAILABLE else "python"
    if backend == "numba":
        if not NUMBA_AVAILABLE:
            raise RuntimeError(
                "The numba PPR backend was requested, but numba is unavailable"
            )
        return _appr_diagnostics
    if backend == "python":
        return getattr(_appr_diagnostics, "py_func", _appr_diagnostics)
    raise ValueError(f"Unknown PPR backend: {backend!r}")


def appr_with_diagnostics(
    num_nodes, indptr, indices, degree, h, alpha, eps, backend="auto",
    seed_nodes=None,
):
    """Return scratch APPR and detailed online-work counters."""
    return get_appr_kernel(backend)(
        num_nodes, indptr, indices, degree, h, alpha, eps, seed_nodes
    )


def appr_with_stats(
    num_nodes, indptr, indices, degree, h, alpha, eps, backend="auto",
    seed_nodes=None,
):
    """Return the scratch APPR vector and its local-push count."""
    p, pushes, _, _ = appr_with_diagnostics(
        num_nodes, indptr, indices, degree, h, alpha, eps,
        backend=backend, seed_nodes=seed_nodes
    )
    return p, pushes


def appr(
    num_nodes, indptr, indices, degree, h, alpha, eps, backend="auto",
    seed_nodes=None,
):
    """Compatibility wrapper returning only the scratch APPR vector."""
    return get_appr_kernel(backend)(
        num_nodes, indptr, indices, degree, h, alpha, eps, seed_nodes
    )[0]


def generate_random_graph(n_nodes, density=0.1):
    adj = sp.random(n_nodes, n_nodes, density=density, format='csr')
    adj.data[:] = 1.0
    adj = adj + adj.T
    adj.data[:] = 1.0 
    degree = np.array(adj.sum(axis=1)).flatten()
    degree[degree == 0] = 1 
    D_inv = sp.diags(1.0 / degree)
    P = adj.dot(D_inv) # P = A @ D_inv
    return adj, P, degree

def compare_results(case_name, vec_pi, vec_appr, h):
    print(f"\n[{case_name}]")
    print(f"  Input h Sum: {np.sum(h):.4f} | Range: [{np.min(h):.4f}, {np.max(h):.4f}]")
    print(f"  PowerIter Sum: {np.sum(vec_pi):.4f}")
    print(f"  APPR Sum:      {np.sum(vec_appr):.4f}")
    
    l1_diff = np.sum(np.abs(vec_pi - vec_appr))
    print(f"  L1 Difference: {l1_diff:.6f}")
    

if __name__ == "__main__":
    N = 1000
    alpha = 0.85
    epsilon = 1e-8 
    iterations = 50 
    adj, P, degree = generate_random_graph(N, density=0.05)
    indptr = adj.indptr
    indices = adj.indices
    h1 = np.zeros(N)
    h1[0] = 1.0 
    pi_res1 = power_iteration(P, alpha, h1, iterations)
    time_start1 = time.time()
    appr_res1 = appr(N, indptr, indices, degree, h1, alpha, epsilon)
    time_end1 = time.time()
    time1 = time_end1 - time_start1
    h2 = np.random.rand(N)
    h2 = h2 / np.sum(h2)
    pi_res2 = power_iteration(P, alpha, h2, iterations)
    time_start2 = time.time()
    appr_res2 = appr(N, indptr, indices, degree, h2, alpha, epsilon)
    time_end2 = time.time()
    time2 = time_end2 - time_start2
    h3 = np.zeros(N)
    h3[0] = 1.0
    h3[1] = -0.5
    h3[2] = -0.5 
    pi_res3 = power_iteration(P, alpha, h3, iterations)
    time_start3 = time.time()
    appr_res3 = appr(N, indptr, indices, degree, h3, alpha, epsilon)
    time_end3 = time.time()
    time3 = time_end3 - time_start3
    h4 = np.zeros(N)
    h4[10] = 10000.0
    pi_res4 = power_iteration(P, alpha, h4, iterations)
    time_start4 = time.time()
    appr_res4 = appr(N, indptr, indices, degree, h4, alpha, epsilon)
    time_end4 = time.time()
    time4 = time_end4 - time_start4
    total_time = time1 + time2+ time3 + time4
    print(total_time)
