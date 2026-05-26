
# Expressiveness of MPNNs with Cypher

## Theorem (Characterization of Cypher-MPNNs)
Let $M$ be a fixed-depth MPNN with $K$ layers over a property graph
$G=(V,E,\rho)$. For each node $v\in V$, suppose every layer has the
form

$$
h_v^{(k)} = \mathrm{UPDATE}_k\!\Bigl(h_v^{(k-1)},\;
            \mathrm{AGGREGATE}_k\!\bigl(\{\,
            \mathrm{MESSAGE}_k(h_v^{(k-1)},\,h_u^{(k-1)},\,e_{vu}) :
            (u,e_{vu})\in N^{(k)}(v)\,\}\bigr)\Bigr).
$$

Then $M$ is a **Cypher-MPNN** iff:

0. $\mathrm{INIT}$ is expressible as a finite Cypher expression over node properties (typically the trivial $h_v^{(0)} = v.x$).

and, for every layer $k\in\{1,\dots,K\}$,

1. $N^{(k)}(v)$ is definable by a finite Cypher pattern query.
2. $\mathrm{MESSAGE}_k$ is expressible as a finite Cypher expression over node features, edge features, and fixed parameters.
3. $\mathrm{AGGREGATE}_k$ is a permutation-invariant aggregation expressible via Cypher grouping and aggregate/list operations.
4. $\mathrm{UPDATE}_k$ is expressible as a finite Cypher expression.
5. The hidden dimension is finite and fixed.
6. The number of layers $K$ is finite and fixed.

### Definition (pure-Cypher functions $\mathcal{C}$)
A function $f$ from a property graph $G=(V,E,\lambda,\rho)$ to a tensor
output is **pure-Cypher** iff it is computed by a Cypher query that
uses only the constructs in the table below:

| category | Cypher constructs | role |
|----------|-------------------|------|
| structural        | `MATCH`, `WHERE`, `WITH`, `UNWIND`, `RETURN`, `CALL { ... }` | pattern matching, row composition, scoping |
| read access       | `n.x`, label tests, edge bindings | reads from $G$ |
| arithmetic / control | `+ - * / ^ %`, comparison, `CASE` | scalar computation |
| list operations   | list comprehension, `range`, `reduce`, indexing | vector / list manipulation |
| scalar built-ins  | `sqrt`, `log`, `exp`, `abs`, `sign`, … | non-linearities |
| aggregators       | `sum`, `avg`, `min`, `max`, `count`, `collect` | multiset reduction |
| transient writes  | `SET n.prop = …` | scratch storage within one query (deterministic w.r.t. $G$) |

The class of such functions is denoted $\mathcal{C}$. 


### Proof
We prove both directions.

**($\Rightarrow$) If $M$ is a Cypher-MPNN, the conditions hold.**

`Condition 1:` For message passing cypher must first bind source/target node pairs
and relationships via pattern matching; therefore $N^{(k)}(v)$ must be
expressible by a finite pattern query. 

`Conditions 2, 4:` For each matched row $(v,u,e_{vu})$ Cypher can only compute values through expressions
over bound variables, properties, parameters, scalar functions, and
list operations; therefore $\mathrm{MESSAGE}_k$ and $\mathrm{UPDATE}_k$
must be Cypher-expressible scalar/list functions.

`Condition 3:` The aggregation step collapses many neighbour messages into one value
per target $v$; Cypher does this by grouping rows and applying
aggregate functions, so $\mathrm{AGGREGATE}_k$ must be expressible as a
grouped aggregation. 

`Conditions 5, 6:` Finally, a Cypher query has finite
syntax — it can only unroll finitely many layers and finitely many
vector dimensions without external recursion. Hence $K$ and the hidden
width are fixed and finite.

**($\Leftarrow$) If the conditions hold, $M$ is a Cypher-MPNN.**

Induction over layers, with the invariant
$\;v.\texttt{h\_curr} = h_v^{(k)}\;$ for every active node $v$ after layer $k$, i.e. the property `v.h_curr` is overwritten at the end of each layer to hold the *current* layer's embedding.

**`Base (k=0)`** By **Condition 0**, $\mathrm{INIT}$ is Cypher-expressible
over node properties. Cypher binds every active node and writes
$h_v^{(0)} = \mathrm{INIT}(v)$ into `n.h_curr`, establishing the
invariant $v.\texttt{h\_curr} = h_v^{(0)}$.

```cypher
MATCH (n)
SET n.h_curr = n.feature_vector   // = h_n^{(0)}
```


**`Inductive step (k>0)`** Assume the invariant after layer $k-1$, so every
active node holds its previous embedding as `n.h_curr`:

```cypher
MATCH (n)
WHERE n.id = x
// IH: layers 1..k-1 already applied; each ran the layer chain below
// and finished with  SET n.h_curr = h_n^{(k-1)}
RETURN n.h_curr   // = h_n^{(k-1)}
```

Layer $k$ is then expressible in Cypher as four steps mirroring
conditions 1–4:

**Condition 1 (neighbourhood).** Cypher enumerates $N^{(k)}(v)$ via
pattern matching:
```cypher
MATCH (v)-[e]-(u)   // or the finite pattern defining N_k(v)
WHERE ...           // optional restrictions on e and u
```

**Condition 2 (message).** For each matched row, Cypher computes the
per-edge message:
```cypher
WITH v, u, e, v.h_curr AS h_v, u.h_curr AS h_u
// ── MESSAGE_k: any cypher expression in (h_v, h_u, e) ──
WITH v,
     <cypher expression for msg from (h_v, h_u, e)>
     AS msg
```

**Condition 3 (aggregation).** Cypher groups by $v$ and reduces the
multiset of messages:
```cypher
WITH v, collect(msg) AS msgs
// ── AGGREGATE_k: any cypher expression over multiset msgs ──
WITH v,
     <cypher expression for agg from msgs>
     AS agg
```

**Condition 4 (update).** Cypher applies the pointwise update and
overwrites `v.h_curr`, re-establishing the invariant:
```cypher
WITH v, v.h_curr AS h_v, agg
// ── UPDATE_k: any cypher expression in (h_v, agg) ──
WITH v,
     <cypher expression for h_new from (h_v, agg)>
     AS h_new
SET v.h_curr = h_new
```

The invariant holds for layer $k$. By induction, after $K$ layers
$v.\texttt{h\_curr} = h_v^{(K)}$ for every active node $v$; therefore the
full forward pass is Cypher-representable. 

### Code example
> **Note.** Pseudo-Cypher. `<...>` = inline slot for the layer
> primitive. `*0..(K-1)` needs a literal/parameter in real Cypher.
> Weights via `$theta.{msg,agg,upd}[k]`.

```cypher
// Sampled subgraph S = (V, E) and seed node n
WITH $V AS V, $E AS E, $seed AS n

// ---- Initialization — store h_0(u) on every u in V ----
UNWIND V AS u
WITH u, u.x AS h_0
SET u.h_curr = h_0
// --------------------------------------------------------

// ---- Layer 1 — frontier = (K-1)-hop of seed; 1-hop gather reaches K-hop ----
// note: '(K-1)' is illustrative; real Cypher needs a parameter, e.g. *0..$Kminus1
OPTIONAL MATCH (n)-[*0..(K-1)]-(u)
WHERE u IN V
WITH DISTINCT u
OPTIONAL MATCH (u)-[e]-(m)
WHERE m IN V AND e IN E
WITH u, u.h_curr AS h_prev,
     collect({m: m, e: e, h_m: m.h_curr}) AS N_in
UNWIND N_in AS nb

// ── MESSAGE_1: cypher expression in (h_prev, nb.h_m, nb.e, $theta.msg[1]) ──
WITH u, h_prev,
     <cypher expression for msg from (h_prev, nb.h_m, nb.e, $theta.msg[1])>
     AS msg

WITH u, h_prev, collect(msg) AS msgs
// ── AGGREGATE_1: cypher expression over (msgs, $theta.agg[1]) ──
WITH u, h_prev,
     <cypher expression for agg from (msgs, $theta.agg[1])>
     AS agg

// ── UPDATE_1: cypher expression in (h_prev, agg, $theta.upd[1]) ──
WITH u,
     <cypher expression for h_1 from (h_prev, agg, $theta.upd[1])>
     AS h_1
SET u.h_curr = h_1
// -----------------------------------------------------------------------------

// ... layer k: frontier = (K-k)-hop of n; 1-hop gather reaches (K-k+1)-hop ...

// ---- Layer K — frontier = {n}; 1-hop gather reaches 1-hop ----
WITH n AS u
OPTIONAL MATCH (u)-[e]-(m)
WHERE m IN V AND e IN E
WITH u, u.h_curr AS h_prev,
     collect({m: m, e: e, h_m: m.h_curr}) AS N_in
UNWIND N_in AS nb

// ── MESSAGE_K: cypher expression in (h_prev, nb.h_m, nb.e, $theta.msg[K]) ──
WITH u, h_prev,
     <cypher expression for msg from (h_prev, nb.h_m, nb.e, $theta.msg[K])>
     AS msg

WITH u, h_prev, collect(msg) AS msgs
// ── AGGREGATE_K: cypher expression over (msgs, $theta.agg[K]) ──
WITH u, h_prev,
     <cypher expression for agg from (msgs, $theta.agg[K])>
     AS agg

// ── UPDATE_K: cypher expression in (h_prev, agg, $theta.upd[K]) ──
WITH u,
     <cypher expression for h_K from (h_prev, agg, $theta.upd[K])>
     AS h_K
SET u.h_curr = h_K
// ---------------------------------------------------------------

RETURN elementId(n) AS nodeId, n.h_curr AS embedding
```


# What functions can be expressed in Cypher?

### Tensor Language ↔ Cypher correspondence
The Tensor Language $\mathrm{TL}(\Omega)$ of Geerts & Reutter has
primitive elements that each have a direct Cypher counterpart:

| TL($\Omega$) construct | Cypher realisation | Notes |
|------------------------|--------------------|-------|
| $P_s(x)$ — feature $s$ of node bound to $x$ | `n.feature_s` where `n` is a bound node variable | Direct property access |
| $E(x,y)$ — edge indicator | `MATCH (x)-->(y)` or `MATCH (x)--(y)` | Pattern matching = edge predicate |
| $\mathbb{1}_{x=y}$ — equality indicator | `WHERE id(x) = id(y)` | Node identity check |
| $\sum_x \varphi(x)$ — summation over all nodes | `MATCH (n) WITH ..., sum(...) AS ...` | Unguarded aggregation; uses global `MATCH` |
| Guarded $\sum_y E(x,y)\cdot\varphi(y)$ — sum over neighbours | `MATCH (n)--(m) WITH n, sum(m.feat) AS ...` | Neighbour pattern = guard; 1 hop = depth 1 |
| $\varphi \cdot \psi$ — pointwise product | `n.a * n.b` or `val1 * val2` in return expression | Arithmetic in `RETURN` / `WITH` |
| $w \cdot \varphi(x)$ — apply weight vector to feature vector | `reduce(s = 0.0, i IN range(0, size(w)-1) \| s + w[i] * n.features[i])` | Dot product in pure Cypher; `w` and `n.features` must have same length |
| $f(\varphi_1,\dots,\varphi_p)$ — scalar function in $\Omega$ | Cypher built-in: `sqrt()`, `log()`, `abs()`, etc. | Only *built-in* functions; no UDFs in pure Cypher |
| Summation depth $t$ | Number of chained `WITH ... sum(...)` stages | Each `WITH` aggregation = one summation level |
| Index variables $x_1, \dots, x_k$ | Distinct node variables in `MATCH` | $k{+}1$ TL indices ↔ $k$-tuple pattern in Cypher |


## Concrete functions expressed in cypher

Each entry shows the math (as in the GNN literature) and a pure-Cypher realisation. Vectors are represented as Cypher lists matrices as list-of-lists. `nb.h_m` is the neighbour's previous embedding, `h_v` the target's, `e` the edge features, and `$theta.[k]` the per-layer parameter map.

### Message functions

| Architecture | Math | Cypher expression |
|--------------|------|-------------------|
| Identity (GraphSAGE-mean, GIN) | $h_u$ | `nb.h_m` |
| GCN (symmetric norm) | $\dfrac{h_u}{\sqrt{(d_v+1)(d_u+1)}}$ | `[i IN range(0, size(nb.h_m)-1) \| nb.h_m[i] / sqrt((u.degree + 1) * (nb.m.degree + 1))]` |
| Linear projection of neighbour | $W\,h_u$ | `[i IN range(0, size(W)-1) \| reduce(s = 0.0, j IN range(0, size(W[0])-1) \| s + W[i][j] * nb.h_m[j])]` |
| Edge-aware linear | $W\,[h_u \,\Vert\, e]$ | concat list `nb.h_m + nb.e`, then same linear pattern as above |
| GAT pre-softmax score | $a^{\!\top}[W h_v \,\Vert\, W h_u]$ + LeakyReLU | dot-product via `reduce`, then `CASE WHEN s > 0 THEN s ELSE alpha*s END` |
| Weighted neighbour | $\alpha_{vu}\, h_u$ | `[i IN range(0, size(nb.h_m)-1) \| alpha * nb.h_m[i]]` |

### Aggregation functions

| Aggregator | Math | Cypher expression |
|-----------|------|-------------------|
| Sum | $\sum_u m_u$ | `[i IN range(0, size(msgs[0])-1) \| reduce(s = 0.0, m IN msgs \| s + m[i])]` |
| Mean | $\dfrac{1}{\lvert N(v)\rvert}\sum_u m_u$ | `[i IN range(0, size(msgs[0])-1) \| reduce(s = 0.0, m IN msgs \| s + m[i]) / size(msgs)]` |
| Component-wise max | $\max_u m_u$ | `[i IN range(0, size(msgs[0])-1) \| reduce(mx = msgs[0][i], m IN msgs \| CASE WHEN m[i] > mx THEN m[i] ELSE mx END)]` |
| Component-wise min | $\min_u m_u$ | identical to max with `<` swapped in |
| GIN sum + $\varepsilon\,h_v$ | $(1+\varepsilon)\,h_v + \sum_u m_u$ | `[i IN range(0, size(h_v)-1) \| (1.0 + eps)*h_v[i] + reduce(s=0.0, m IN msgs \| s + m[i])]` |
| GAT weighted sum (post-softmax) | $\sum_u \alpha_{vu}\, m_u$ | softmax over scores via `reduce` (see activation table), then per-component sum as above |

### Update

| Update | Math | Cypher expression |
|--------|------|-------------------|
| Linear + ReLU | $\mathrm{ReLU}(W\,\mathrm{agg} + b)$ | two stages: `WITH ... [i IN range(0, size(W)-1) \| reduce(s = 0.0, j IN range(0, size(agg)-1) \| s + W[i][j]*agg[j]) + b[i]] AS z`, then `[z_i IN z \| CASE WHEN z_i > 0 THEN z_i ELSE 0.0 END]` |
| GraphSAGE concat-update | $\sigma\!\bigl(W\,[h_v \,\Vert\, \mathrm{agg}]\bigr)$ | concat `h_v + agg` (list concat), then linear + activation as above |
| GIN MLP-update | $\mathrm{MLP}\!\bigl((1+\varepsilon)\,h_v + \sum_u m_u\bigr)$ | two linear-then-activation stages chained via `WITH ... AS h_layer1`, `WITH ... AS h_layer2` |
| Residual / skip | $h_v + W\,\mathrm{agg}$ | `[i IN range(0, size(h_v)-1) \| h_v[i] + reduce(s = 0.0, j IN range(0, size(agg)-1) \| s + W[i][j]*agg[j])]` |
| Gated (simplified) | $z\odot\tilde h + (1-z)\odot h_v$ | compute $z$ as linear + sigmoid, $\tilde h$ as linear + tanh, then `[i IN range(0, size(h_v)-1) \| z[i]*h_tilde[i] + (1.0 - z[i])*h_v[i]]` |

# `Example:` GAT + GCN - GNN implementation

Two-layer GNN over the same sampled subgraph: **Layer 1 = GAT**
(single-head attention with `ELU` activation), **Layer 2 = GCN**
(symmetric normalisation + linear + `ReLU`). Concrete Cypher inlined
into the intuitive template; all weights live in `$theta`:

- `$theta.msg[1].W`, `$theta.msg[1].a`  — GAT projection + attention vector
- `$theta.upd[2].W`, `$theta.upd[2].b`  — GCN linear

```cypher
// Sampled subgraph S = (V, E) and seed node n
WITH $V AS V, $E AS E, $seed AS n

// ---- Initialization ----
UNWIND V AS u
WITH u, u.x AS h_0
SET u.h_curr = h_0

// =============================================================================
// Layer 1 — GAT
// =============================================================================
// frontier = 1-hop of seed (so layer-2 sees seed's gather)
OPTIONAL MATCH (n)-[*0..1]-(u) WHERE u IN V
WITH DISTINCT u
OPTIONAL MATCH (u)-[e]-(m) WHERE m IN V AND e IN E
WITH u, u.h_curr AS h_prev,
     collect({m: m, e: e, h_m: m.h_curr}) AS N_in
UNWIND N_in AS nb

// project both endpoints through W
WITH u, h_prev, nb,
     [i IN range(0, size($theta.msg[1].W)-1) |
        reduce(s = 0.0, j IN range(0, size(h_prev)-1)
               | s + $theta.msg[1].W[i][j] * h_prev[j])]  AS Wh_v,
     [i IN range(0, size($theta.msg[1].W)-1) |
        reduce(s = 0.0, j IN range(0, size(nb.h_m)-1)
               | s + $theta.msg[1].W[i][j] * nb.h_m[j])]  AS Wh_u

// edge score: LeakyReLU(a^T [Wh_v || Wh_u])
WITH u, h_prev, Wh_u,
     reduce(s = 0.0, k IN range(0, size($theta.msg[1].a)-1)
            | s + $theta.msg[1].a[k] * (Wh_v + Wh_u)[k]) AS raw
WITH u, h_prev, Wh_u,
     CASE WHEN raw > 0 THEN raw ELSE 0.2 * raw END AS score

// gather per-u list of (Wh_u, score) for softmax
WITH u, h_prev, collect({Wh_u: Wh_u, score: score}) AS items

// softmax over neighbours (max-shift, numerically stable)
WITH u, h_prev, items,
     reduce(mx = items[0].score, it IN items
            | CASE WHEN it.score > mx THEN it.score ELSE mx END) AS mx
WITH u, h_prev,
     [it IN items | {Wh_u: it.Wh_u, w: exp(it.score - mx)}] AS exps
WITH u, h_prev, exps,
     reduce(s = 0.0, x IN exps | s + x.w) AS Z
WITH u, h_prev,
     [x IN exps | {Wh_u: x.Wh_u, alpha: x.w / Z}] AS attended

// AGGREGATE_1: weighted sum  alpha_uv * Wh_u
WITH u,
     [i IN range(0, size(attended[0].Wh_u)-1)
      | reduce(s = 0.0, x IN attended | s + x.alpha * x.Wh_u[i])] AS agg

// UPDATE_1: ELU activation
WITH u,
     [a IN agg | CASE WHEN a > 0 THEN a ELSE exp(a) - 1.0 END] AS h_1
SET u.h_curr = h_1

// =============================================================================
// Layer 2 — GCN
// =============================================================================
// frontier = {n}; gather 1-hop
WITH n AS u
OPTIONAL MATCH (u)-[e]-(m) WHERE m IN V AND e IN E
WITH u, u.h_curr AS h_prev,
     collect({m: m, e: e, h_m: m.h_curr}) AS N_in
UNWIND N_in AS nb

// MESSAGE_2: symmetric norm  h_m / sqrt((d_v+1)(d_u+1))
WITH u, h_prev,
     [i IN range(0, size(nb.h_m)-1)
      | nb.h_m[i] / sqrt((u.degree + 1.0) * (nb.m.degree + 1.0))] AS msg

// AGGREGATE_2: element-wise sum
WITH u, h_prev, collect(msg) AS msgs
WITH u, h_prev,
     [i IN range(0, size(msgs[0])-1)
      | reduce(s = 0.0, m IN msgs | s + m[i])] AS agg

// UPDATE_2: W·agg + b  →  ReLU
WITH u, agg,
     [i IN range(0, size($theta.upd[2].W)-1)
      | reduce(s = 0.0, j IN range(0, size(agg)-1)
               | s + $theta.upd[2].W[i][j] * agg[j])
        + $theta.upd[2].b[i]] AS z
WITH u, [zi IN z | CASE WHEN zi > 0 THEN zi ELSE 0.0 END] AS h_2
SET u.h_curr = h_2

RETURN elementId(n) AS nodeId, n.h_curr AS embedding
```