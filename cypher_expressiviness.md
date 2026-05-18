
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

Then $M$ is a **Cypher-MPNN** iff, for every layer $k\in\{1,\dots,K\}$,

1. $N^{(k)}(v)$ is definable by a finite Cypher pattern query.
2. $\mathrm{MESSAGE}_k$ is expressible as a finite Cypher scalar/list
   expression over node features, edge features, and fixed parameters.
3. $\mathrm{AGGREGATE}_k$ is a permutation-invariant aggregation
   expressible via Cypher grouping and aggregate/list operations.
4. $\mathrm{UPDATE}_k$ is expressible as a finite Cypher scalar/list
   expression.
5. The hidden dimension is finite and fixed.
6. The number of layers $K$ is finite and fixed.

**Equivalently.** $M$ is a Cypher-MPNN iff every layer decomposes into:
finite pattern match → per-edge message → grouped aggregation →
pointwise update. This matches Cypher's core execution model: pattern
matching is central to graph access, and Cypher / GQL-style pipelines
support projection, filtering, grouping, aggregation, and linear
composition of query parts.

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

The class of such functions is denoted $\mathcal{C}$. It **excludes**
`CREATE`, `MERGE`, `DELETE`, stored procedures, APOC calls, and any
user-defined functions.


### Proof
We prove both directions.

**($\Rightarrow$) If $M$ is a Cypher-MPNN, the conditions hold.**

Cypher can only compute a layer through the graph operations available
to it. For message passing it must first bind source/target node pairs
and relationships via pattern matching; therefore $N^{(k)}(v)$ must be
expressible by a finite pattern query *(condition 1)*. For each matched
row $(v,u,e_{vu})$ Cypher can only compute values through expressions
over bound variables, properties, parameters, scalar functions, and
list operations; therefore $\mathrm{MESSAGE}_k$ and $\mathrm{UPDATE}_k$
must be Cypher-expressible scalar/list functions *(conditions 2, 4)*.
The aggregation step collapses many neighbour messages into one value
per target $v$; Cypher does this by grouping rows and applying
aggregate functions, so $\mathrm{AGGREGATE}_k$ must be expressible as a
grouped aggregation *(condition 3)*. Finally, a Cypher query has finite
syntax — it can only unroll finitely many layers and finitely many
vector dimensions without external recursion. Hence $K$ and the hidden
width are fixed and finite *(conditions 5, 6)*.

**($\Leftarrow$) If the conditions hold, $M$ is a Cypher-MPNN.**

Induction over layers, with the invariant
$\;v.h_k = h_v^{(k)}\;$ for every active node $v$ after layer $k$.

*Base ($k=0$).* The initial feature map is finite-dimensional and
either stored as node properties or computable from them; Cypher binds
every active node and computes $h_v^{(0)}$. The invariant
$v.h_0 = h_v^{(0)}$ holds.

*Inductive step.* Assume the invariant after layer $k-1$. By
condition 1, Cypher enumerates the neighbourhood:

```cypher
MATCH (v)-[e]-(u)   // or the finite pattern defining N_k(v)
WHERE e = ... and u = ... // possible resitrictions on e and u
```

By condition 2, for each matched row Cypher computes the per-edge
message:
```cypher
WITH v, u, e, v.h_{k-1} AS h_v, u.h_{k-1} AS h_u
WITH v, MESSAGE_k(h_v, h_u, e) AS msg
```

By condition 3, Cypher groups by $v$ and reduces:
```cypher
WITH v, collect(msg) AS msgs
WITH v, AGGREGATE_k(msgs) AS agg
```

By condition 4, Cypher computes the update and stores it as $v.h_k$:
```cypher
WITH v, UPDATE_k(v.h_{k-1}, agg) AS h_new
SET v.h_k = h_new
```

The invariant holds for layer $k$. By induction, after $K$ layers
Cypher computes $h_v^{(K)}$ for every active node $v$; therefore the
full forward pass is Cypher-representable. 

### Intuitive code example
> **Note.** Pseudo-Cypher: `MESSAGE_k`, `AGGREGATE_k`, `UPDATE_k` are
> placeholders for corresponding code blocks
> Likewise `*0..(K-1)` is illustrative; real Cypher
> requires a literal or parameter in a variable-length range
> (e.g. `*0..$Kminus1`).
>
> Learnable weights are passed in via the parameter map `$theta`, with
> `$theta.msg[k]`, `$theta.agg[k]`, `$theta.upd[k]` holding the
> per-layer parameters for the message, aggregation, and update
> functions.

```cypher
// Sampled subgraph S = (V, E) and seed node n
WITH $V AS V, $E AS E, $seed AS n

// ---- Initialization — store h_0(u) on every u in V ----
UNWIND V AS u
WITH u, u.x AS h_0
SET u.h = h_0
// --------------------------------------------------------

// ---- Layer 1 — frontier = (K-1)-hop of seed; 1-hop gather reaches K-hop ----
// note: '(K-1)' is illustrative; real Cypher needs a parameter, e.g. *0..$Kminus1
OPTIONAL MATCH (n)-[*0..(K-1)]-(u)
WHERE u IN V
WITH DISTINCT u
OPTIONAL MATCH (u)-[e]-(m)
WHERE m IN V AND e IN E
WITH u, u.h AS h_prev,
     collect({m: m, e: e, h_m: m.h}) AS N_in
UNWIND N_in AS nb
WITH u, h_prev, MESSAGE_1(h_prev, nb.h_m, nb.e, $theta.msg[1]) AS msg
WITH u, h_prev, collect(msg) AS msgs
WITH u, h_prev, AGGREGATE_1(msgs, $theta.agg[1]) AS agg
WITH u, UPDATE_1(h_prev, agg, $theta.upd[1]) AS h_1
SET u.h = h_1
// -----------------------------------------------------------------------------

// ... layer k: frontier = (K-k)-hop of n; 1-hop gather reaches (K-k+1)-hop ...

// ---- Layer K — frontier = {n}; 1-hop gather reaches 1-hop ----
WITH n AS u
OPTIONAL MATCH (u)-[e]-(m)
WHERE m IN V AND e IN E
WITH u, u.h AS h_prev,
     collect({m: m, e: e, h_m: m.h}) AS N_in
UNWIND N_in AS nb
WITH u, h_prev, MESSAGE_K(h_prev, nb.h_m, nb.e, $theta.msg[K]) AS msg
WITH u, h_prev, collect(msg) AS msgs
WITH u, h_prev, AGGREGATE_K(msgs, $theta.agg[K]) AS agg
WITH u, UPDATE_K(h_prev, agg, $theta.upd[K]) AS h_K
SET u.h = h_K
// ---------------------------------------------------------------

RETURN elementId(n) AS nodeId, n.h AS embedding
```


# What functions can be expressed in Cypher?
So as long as the neighbor indexing, message function, aggregation operation, and update functions for every layer are expressible independently in cypher, the k-MPNN can be expressed

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

Each entry shows the math (as in the GNN literature) and a pure-Cypher
realisation. Vectors are represented as Cypher lists; matrices as
list-of-lists. `nb.h_m` is the neighbour's previous embedding, `h_v`
the target's, `e` the edge features, and `$theta.*[k]` the per-layer
parameter map.

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