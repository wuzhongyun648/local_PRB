# Corrections Required for DYN-APPR

This note documents two corrections needed to make Algorithm 3 in
`Local_Bandits_on_VLGs.pdf` consistent with the PPR equation and the
linear-system invariant used by the paper.

## 1. Correct `INSERTONEDIR`

The paper defines

\[
\pi = \alpha P\pi + (1-\alpha)s
\]

and maintains the APPR invariant

\[
(I-\alpha P)p+(1-\alpha)r=(1-\alpha)s.
\]

Suppose a directed edge `(u, v)` is inserted and the degree of `u` changes
from `d_u - 1` to `d_u`. Define

\[
q = \frac{p_{\mathrm{old}}(u)}{d_u-1}.
\]

The invariant-preserving update is

\[
p(u) \leftarrow p(u)\frac{d_u}{d_u-1},
\]

\[
r(u) \leftarrow r(u)-\frac{q}{1-\alpha},
\qquad
r(v) \leftarrow r(v)+\frac{\alpha q}{1-\alpha}.
\]

Therefore lines 13-15 of Algorithm 3 should be replaced by:

```text
q <- p(u) / d_u  // after p(u) has been rescaled
r(u) <- r(u) - q / (1 - alpha)
r(v) <- r(v) + alpha * q / (1 - alpha)
```

The printed update

```text
Delta <- p(u) / (alpha * d_u)
r(u) <- r(u) - Delta
r(v) <- r(v) + (1 - alpha) * Delta
```

does not preserve the stated invariant for general `alpha`.

For an undirected edge `(a, b)`, apply the corrected operation to `(a, b)`
and `(b, a)`.

## 2. Account for a Changing Personalization Vector

Algorithm 1 constructs a new sparse personalization vector `s_t` at every
round. Algorithm 3 carries `(p, r)` from the previous round, but the printed
pseudocode does not update the residual when `s_{t-1}` changes to `s_t`.

After applying `INSERTUPDATE` and before calling APPR, add:

```text
r <- r + (s_t - s_{t-1})
```

This follows directly from the invariant: changing its right-hand side from
`(1-alpha)s_{t-1}` to `(1-alpha)s_t` requires adding `s_t-s_{t-1}` to `r`.
Because both vectors are supported only on the candidate sets, this update is
sparse.

Algorithm 3 must therefore retain the previous `s` in addition to `(p, r)`.

## Corrected Algorithm Outline

```text
Initialize p <- 0, r <- s_1
APPR(alpha, eps, G_0, p, r)

For round t > 1:
    if edge (a, b) was inserted:
        INSERTONEDIR_CORRECTED(a, b)
        INSERTONEDIR_CORRECTED(b, a)
    r <- r + (s_t - s_{t-1})
    APPR(alpha, eps, G_{t-1}, p, r)
```

With these corrections, every pre-push state satisfies the stated
linear-system invariant, and APPR restores
`max_u |r(u)| / d_u < eps`.
