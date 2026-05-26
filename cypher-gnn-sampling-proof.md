# Cypher-Expressibility of GNN Sampling Algorithms

### Setup

Fix finite property graph $G = (V, E, \rho)$.

**Cypher-representable domain.** A domain $D$ is *Cypher-representable* if it is built from the following grammar:

- **Atoms:** nodes $V$, relationships $E$, integers $\mathbb{Z}$, floats $\mathbb{R}$, booleans $\mathbb{B}$, strings $\text{Str}$.
- **Composites:** finite lists $[D]$, maps $\{\text{Str} \to D\}$, or tuples $D_1 \times \cdots \times D_k$ whose components are Cypher-representable.

Equivalently: $D$ is Cypher-representable iff every $d \in D$ can be written as a Cypher literal. This restricts sampler state to values Cypher can natively store and manipulate.

**Multiset–list equivalence.** Write $L \equiv_{\text{ms}} R$ iff for every $\bar{x}$,
$$|\{j \in \{1,\ldots,|L|\} : L[j] = \bar{x}\}| = R(\bar{x}).$$
That is, $L$ and $R$ agree on multiplicities; order of $L$ is ignored.

**Cypher-expressibility.** A function $f$ on Cypher-representable values is *Cypher-expressible* if it can be computed by a read-only Cypher fragment. The allowed clauses are: `MATCH`, `OPTIONAL MATCH`, `WHERE`, `WITH`, `UNWIND`, `collect`, `ORDER BY`, `LIMIT`, `CALL { }`, `RETURN`, list comprehensions, `reduce`, and scalar/aggregate expressions. The fragment may take Cypher parameters (`$name`) for externally supplied data.

**Sampler state.** A *sampler state* is a tuple $\Sigma = (R_1, \ldots, R_m)$ where each $R_i$ is a finite multiset over a fixed schema $D_{i1} \times \cdots \times D_{ik_i}$ of Cypher-representable domains. The arity $m$, the schemas, and $k_i$ are fixed constants independent of $G$.

**Encoding.** Each tuple $\bar{x} \in R_i$ is encoded as a Cypher map with keys equal to schema field names. $R_i$ is encoded as a Cypher list $L_i$ of such maps. The full state $\Sigma^{\text{Cy}}$ is encoded as a single Cypher map `state` whose keys are the component names $n_1, \ldots, n_m$ and whose values are the corresponding lists:

$$\Sigma^{\text{Cy}} \;=\; \texttt{\{$n_1$: $L_1$, \ldots, $n_m$: $L_m$\}}$$

This is Cypher-representable by the **Composites** clause ($\{\text{Str} \to D\}$ map of lists). Individual components are accessed as `state.`$n_i$, and the state is updated by constructing a new map with `WITH \{…\} AS state`.

*Example.* Take a GraphSAGE-style state with two components: a frontier $R_1$ over schema $(p, \ell)$ and a sampled-edge set $R_2$ over schema $(p, c, e, \ell)$. Concretely, $R_1 = \{(v_3, 0), (v_7, 0)\}$ and $R_2 = \emptyset$. The encoding is

```cypher
MATCH (v3) WHERE id(v3) = 3
MATCH (v7) WHERE id(v7) = 7
WITH {frontier: [{p: v3, l: 0}, {p: v7, l: 0}],
      sampled:  []} AS state
```

so `state` is a single Cypher value with `state.frontier` $\equiv_{\text{ms}} R_1$ and `state.sampled` $\equiv_{\text{ms}} R_2$. Nodes are stored as node references rather than ids, so subsequent `MATCH` clauses can pattern against them directly. Multiplicity is preserved positionally: $(v_3, 0)$ appearing twice in $R_1$ would appear as two copies of `{p: v3, l: 0}` in `state.frontier`.

**Random tape.** A *random tape* is an injective function $\xi: K \to \mathbb{R}$ where $K$ is a Cypher-representable key set fixed by the algorithm (typically candidate identifiers, e.g. node ids or $(\text{step}, \text{candidate-id})$ pairs). $\xi$ is supplied to Cypher as a parameter `$xi` — concretely a map keyed by elements of $K$, accessed via `$xi[k]`. Injectivity means `ORDER BY $xi[k]` is total (no ties) on any finite subset of $K$.

The tape abstraction makes the proof deterministic-given-$\xi$, which is convenient for the equivalence statement $Q_A(G, \texttt{\$xi}) \equiv_{\text{ms}} O$. In practice Cypher provides `rand()` (uniform on $[0,1)$, evaluated per row), so `ORDER BY rand()` is a valid in-engine substitute when reproducibility is not required; the proof goes through with $\xi$ realised by `rand()` and ties broken arbitrarily (with probability 1 over a continuous distribution there are none).

**Key function.** Each step $t$ comes with a *key function* $\kappa_t$: a Cypher-expressible function mapping each candidate $c \in C_t$ to a key in $K$. The composition $\xi \circ \kappa_t$ assigns a unique real to each candidate. $\kappa_t$ is part of the algorithm spec, not the tape.

### Bounded transition sampler

A *bounded transition sampler* $A$ is a tuple $(T, \texttt{INIT}, \{(\texttt{GEN}_t, \texttt{SCORE}_t, \kappa_t, \texttt{SELECT}_t, \texttt{UPDATE}_t)\}_{t<T}, \texttt{OUT})$ where $T \in \mathbb{N}$ is fixed, $\Sigma_t$ ranges over *sampler states* (same schema as defined above, indexed by step $t$), and the algorithm runs as:

$$
\begin{aligned}
\Sigma_0 &= \texttt{INIT}(G) && \text{build the initial state from } G \text{ (runs once)} \\[4pt]
C_t &= \texttt{GEN}_t(G, \Sigma_t) && \text{enumerate candidate tuples from } G \text{ relative to current state} \\
W_t &= \texttt{SCORE}_t(G, \Sigma_t, C_t) && \text{assign a weight to each candidate} \\
P_t &= \texttt{SELECT}_t(W_t, \xi \circ \kappa_t) && \text{draw a sub-multiset using the random tape} \\
\Sigma_{t+1} &= \texttt{UPDATE}_t(G, \Sigma_t, P_t) && \text{fold selected candidates back into the state} \\[4pt]
O &= \texttt{OUT}(\Sigma_T) && \text{extract the final output from the terminal state (runs once)}
\end{aligned}
$$

$C_t, W_t, P_t$ are finite multisets over fixed Cypher-representable schemas whose arity and field types are constants of the algorithm, independent of $G$. $\texttt{INIT}$ and $\texttt{OUT}$ are Cypher-expressible functions of $G$ and $\Sigma_T$ respectively; $\texttt{INIT}$'s output domain is the state schema. The output of $A$ on $(G, \xi)$ is $O$.

### Lemmas

**Lemma 1 (multiset–list).** *Cypher lists faithfully represent finite multisets, and standard multiset operations match standard list operations.*

For every finite multiset $R$ over a Cypher-representable schema, there is a Cypher list $L$ with $L \equiv_{\text{ms}} R$. Conversely every Cypher list of records induces a multiset. Multiset union $\uplus$ is realised by list concatenation; multiset filter, projection, and grouping by `[x IN L WHERE …]`, list comprehensions, and `UNWIND … WITH … collect`.

*Proof.* Construction. Fix a schema with field names $f_1, \ldots, f_k$ and a multiset $R$.

*Encoding $R \mapsto L$.* Enumerate $R$ in any order respecting multiplicity, mapping each tuple $\bar{x} = (x_1, \ldots, x_k)$ to the Cypher map literal `{f1: x_1, ..., fk: x_k}`. The list $L$ is the Cypher literal of these maps. By construction $|\{j : L[j] = \bar{x}\}| = R(\bar{x})$ for every $\bar{x}$, so $L \equiv_{\text{ms}} R$.

*Decoding $L \mapsto R$.* Define $R(\bar{x}) := |\{j : L[j] = \bar{x}\}|$. Finiteness of Cypher lists gives finiteness of $R$.

*Operations.* Let $L_1, L_2$ encode $R_1, R_2$. Each multiset operation is realised by the indicated Cypher fragment, and equality of multiplicities follows from list semantics:

- **Union $R_1 \uplus R_2$**: list concatenation.

```cypher
WITH L1 + L2 AS L
```

- **Filter $\{\bar{x} \in R : \pi(\bar{x})\}$**: list comprehension with predicate.

```cypher
WITH [x IN L WHERE pi(x)] AS L_filtered
```

- **Projection $\{g(\bar{x}) : \bar{x} \in R\}$** (multiset; collisions kept): list comprehension with map expression.

```cypher
WITH [x IN L | {h1: g1(x), ..., hm: gm(x)}] AS L_proj
```

- **Group-aggregate $\{(k, \alpha(\{\bar{x} \in R : \kappa(\bar{x}) = k\})) : k \in \kappa(R)\}$**: unwind, group, recollect.

```cypher
UNWIND L AS x
WITH kappa(x) AS k, collect(x) AS grp
WITH collect({key: k, agg: alpha(grp)}) AS L_grouped
```

Empty case: `L = []` gives `UNWIND` zero rows; `collect` over zero rows returns `[]`, so $L_{\text{grouped}} = []$ matches the empty grouping. Multiplicities match by Cypher's `UNWIND`/`collect` semantics. □

**Lemma 2 (graph-relative enumeration).** *For each row in the current state, pattern-matching its values against the graph and collecting all results is a single `UNWIND … MATCH … WITH state, collect` clause.*

*Proof.* Take a state component `state.`$n$ encoding multiset $R$, and a Cypher graph pattern $\varphi$ that binds graph variables $\bar{y}$ given a state row $x$. The following fragment produces a new list of all $(x, \bar{y})$ matches while preserving `state`:

```cypher
UNWIND state.n AS x
MATCH (varphi binding y1, ..., yq given x)
WITH state, collect({x: x, y1: y1, ..., yq: yq}) AS candidates
```

Three things happen in sequence:

1. **`UNWIND state.n AS x`** — turns the list into one row per entry, with duplicates respected. If a tuple appears $k$ times in `state.n`, it produces $k$ rows. If `state.n = []`, zero rows.
2. **`MATCH …`** — for each row, finds every assignment of $\bar{y}$ satisfying $\varphi$ in $G$. Since $G$ is finite this always terminates. Rows with no match are dropped; use `OPTIONAL MATCH` if zero-match rows must be preserved.
3. **`WITH state, collect(…)`** — gathers every `(x, y1, …, yq)` combination back into a single list, which encodes the result multiset. `state` is carried forward; without it, subsequent steps lose access to the sampler state.

Empty-state case: `state.n = []` ⇒ zero rows ⇒ `collect` returns `[]`, `state` retained, invariant preserved.

*Cross-component generation.* If `GEN` requires joining multiple state components (e.g. PinSAGE walks × visits), chain `UNWIND` clauses:
```cypher
UNWIND state.walks AS w
UNWIND state.visits AS v
WHERE v.seed = w.seed
...
WITH state, collect({...}) AS candidates
```
The pattern composes inside the allowed fragment.

*Example.* For GraphSAGE, `state.frontier` holds nodes to expand. The candidates are all their neighbours:

```cypher
UNWIND state.frontier AS x
MATCH (x.p)-[e]-(c)
WITH state, collect({p: x.p, c: c, e: e, l: x.l}) AS candidates
```

Each frontier node produces one candidate entry per neighbour. If the same node appears twice in the frontier, its neighbours appear twice in `candidates` — matching the multiset semantics of $R$. □

**Lemma 3 (closure under sequential composition).** *If $F_1, F_2$ are Cypher-expressible fragments, each producing one or more bound variables consumed by the next, then their composition via `WITH` chaining is Cypher-expressible.*

*Proof.* Each fragment uses only allowed clauses by hypothesis. `WITH` is allowed. Cypher's row-pipeline semantics passes named bindings from one clause to the next; nothing in the allowed-clause list breaks under sequential composition. Hence the concatenation $F_1 \mathbin{;} \texttt{WITH …} \mathbin{;} F_2$ is itself in the allowed fragment. □

### Theorem

Let $A = (T, \texttt{INIT}, \{(\texttt{GEN}_t, \texttt{SCORE}_t, \kappa_t, \texttt{SELECT}_t, \texttt{UPDATE}_t)\}_{t<T}, \texttt{OUT})$ be a bounded transition sampler. If for every $t < T$:

- $\texttt{GEN}_t$ enumerates candidates by a Cypher pattern over $G$ and $\Sigma_t$ (Lemma 2 applies);
- $\texttt{SCORE}_t$ is a Cypher-expressible scalar/aggregate function of $(G, \Sigma_t, c)$ per candidate $c$;
- $\kappa_t$ is Cypher-expressible;
- $\texttt{SELECT}_t$ uses only filtering, ordering by $\xi \circ \kappa_t$ (read from `$xi`), `LIMIT`, list slicing, grouping, and arithmetic comparison against $\xi$-values;
- $\texttt{UPDATE}_t$ is built from Lemma 1 operations (concat, comprehension, group-aggregate) plus pattern lookups via Lemma 2;
- $\texttt{INIT}$ and $\texttt{OUT}$ are Cypher-expressible functions of $G$ and $\Sigma_T$ respectively;

then there exists a read-only Cypher query $Q_A$ such that for every finite $G$ and every injective tape $\xi$ (passed as parameter `$xi`),
$$
Q_A(G, \texttt{\$xi}) \equiv_{\text{ms}} O = \texttt{OUT}(A(G, \xi)).
$$

### Proof

Induction on $t \in \{0, \ldots, T\}$ with invariant $\Sigma_t^{\text{Cy}} \equiv_{\text{ms}} \Sigma_t$ (componentwise on each state field).

**Base.** $\texttt{INIT}$ Cypher-expressible ⇒ a Cypher fragment computes $\Sigma_0$'s components as lists $L_1, \ldots, L_m$. By Lemma 1, $L_i \equiv_{\text{ms}} R_i$ for each component. Binding them as `WITH {n1: L1, ..., nm: Lm} AS state` gives $\Sigma_0^{\text{Cy}} \equiv_{\text{ms}} \Sigma_0$.

**Step.** Assume invariant at $t$.

*Candidates.* By Lemma 2 applied to $\texttt{GEN}_t$'s pattern over $\Sigma_t^{\text{Cy}}$ and $G$ (with `WITH state, collect(...) AS candidates` carrying state forward), Cypher produces $C_t^{\text{Cy}} \equiv_{\text{ms}} C_t$.

*Scores.* $\texttt{SCORE}_t$ is a per-candidate Cypher-expressible scalar/aggregate, so for every $c \in C_t$, $\texttt{SCORE}_t(G, \Sigma_t, c) = \texttt{SCORE}_t(G, \Sigma_t^{\text{Cy}}, c)$ (Cypher closed under scalar/aggregate over equivalent inputs). Apply via:
```cypher
WITH state, [c IN candidates | {c: c, w: SCORE_t(G, state, c)}] AS W
```
Yields $W_t^{\text{Cy}} \equiv_{\text{ms}} W_t$.

*Selection.* Bind keys via $\kappa_t$ then look up tape:
```cypher
WITH state, [w IN W | {c: w.c, w: w.w, r: $xi[kappa_t(w.c)]}] AS Wk
```
Injectivity of $\xi$ on the (finite) candidate key set ensures all `r` values are distinct, so `ORDER BY r` is total — Cypher's `ORDER BY` is deterministic on distinct keys, hence `LIMIT k` is well-defined with no tie-break ambiguity. $\texttt{SELECT}_t$'s composition of filter / sort / limit / slice / group is built from operations closed under multiset equivalence (Lemma 1) and produces the same multiset $P_t^{\text{Cy}}$ from $W_t^{\text{Cy}}$ that abstract $\texttt{SELECT}_t$ produces from $W_t$ given the same $\xi$. Hence $P_t^{\text{Cy}} \equiv_{\text{ms}} P_t$.

*Update.* $\texttt{UPDATE}_t$ is a finite composition of Lemma 1 and Lemma 2 operations. By Lemma 3 (closure under composition), the resulting fragment is Cypher-expressible; by closure of $\equiv_{\text{ms}}$ under each constituent operation, $\Sigma_{t+1}^{\text{Cy}} \equiv_{\text{ms}} \Sigma_{t+1}$. Invariant at $t+1$.

By induction, $\Sigma_T^{\text{Cy}} \equiv_{\text{ms}} \Sigma_T$. $\texttt{OUT}$ Cypher-expressible ⇒ apply to `state` and return:
```cypher
RETURN OUT_expr(state) AS output
```
giving $Q_A(G, \texttt{\$xi}) \equiv_{\text{ms}} \texttt{OUT}(\Sigma_T) = O$. □

**State size.** Each step's `candidates` is bounded by $|\Sigma_t| \cdot d_{\max}^{q_t}$ where $d_{\max}$ is max degree and $q_t$ the pattern arity; `SELECT` only shrinks. State grows by at most a polynomial in $|G|$ per step, total bounded by $\text{poly}(|G|, T)$ — finite, so the Cypher list values remain well-defined.

### Not covered (explicit limits)

The theorem does **not** capture:

1. **Unbounded $T$** — fixed-point or convergence-driven samplers (e.g. run until subgraph size $\geq s$ with input-dependent stop). Bounded $T$ baked into hypothesis.
2. **Schema growth.** Tuple arity / number of state components fixed; algorithms that maintain unbounded memory per walker (e.g. full self-avoiding history with no bound) escape.
3. **Non-Cypher-expressible primitives inside $\texttt{SCORE}$.** Eigenvectors, METIS clustering, $k$-means, gradient-derived weights: must be precomputed and stored as node/edge properties, else outside class.
4. **Reproducibility / seeded randomness.** The formal claim uses an external tape $\xi$ to make the equivalence deterministic. Cypher's `rand()` works for in-engine sampling but is not seedable per query in standard Cypher, so deterministic replay across runs requires the tape route (or `apoc.util.random` / similar engine-specific seeded primitives).

### Corollary (tightened)

Each named sampler instantiates one fixed schema + transition components. Tape consumed wherever the $\texttt{SELECT}_t$ column is non-trivial (i.e. anything but pure top-$k$ over deterministic scores); $\kappa_t$ is the candidate identifier unless noted.

| Sampler | $\Sigma_t$ schemas | $\texttt{GEN}_t$ pattern | $\texttt{SCORE}_t$ | $\texttt{SELECT}_t$ ($\xi$?) | $\texttt{UPDATE}_t$ |
|---|---|---|---|---|---|
| Uniform node | $(v)$ | `MATCH (v)` | $1$ | top-$k$ by $\xi$ (yes) | $\Sigma_{t+1} := P_t$ (replace) |
| Uniform edge | $(e,u,v)$ | `MATCH (u)-[e]-(v)` | $1$ | top-$m$ by $\xi$ (yes) | $\Sigma_{t+1} := P_t$ (replace) |
| GraphSAGE | frontier $(p, \ell)$, sampled $(p,c,e,\ell)$ | `UNWIND F MATCH (p)-[e]-(c)` | $1$ | per-$p$ top-$k$ by $\xi$ (yes) | append $P_t$ to sampled; new frontier $:= \{(c, \ell{+}1)\}$ |
| FastGCN | layer nodes $(v,\ell)$ | `MATCH (v)` | $q(v)$ from stored prop or `MATCH` aggregate | weighted top-$k$ (yes, via $\xi/q$) | layer $\ell{+}1 := P_t$ |
| LADIES | upper frontier $(v,\ell)$, sampled $(v,\ell)$ | `UNWIND F MATCH (v)-[]-(u)` | adj-mass aggregate | weighted top-$k$ (yes) | sampled $\mathrel{+}= P_t$; frontier $:= P_t$ at $\ell{+}1$ |
| Random walk (uniform) | $(w, v, s, prev)$ | `UNWIND W MATCH (v)-[]-(u) WHERE u≠prev` | $1$ | per-$w$ pick 1 by $\xi$ (yes) | $v \leftarrow u$, $prev \leftarrow v$, $s \leftarrow s{+}1$ |
| Random walk (node2rec/PinSAGE) | as above | as above | precomputed transition prob (on edge prop) | per-$w$ weighted pick 1 (yes) | as above |
| PinSAGE | $(seed, w, v, s)$ + visits $(seed, v, c)$ | walk + group-count | $c$ | per-seed top-$k$ (no — deterministic) | walk step + visit-count increment via `UNWIND … MERGE/aggregate` |
| GraphSAINT N/E ($T{=}1$) | $(v)$ or $(e,u,v)$ | `MATCH` global | uniform or precomp. weight | weighted top-$k$ (yes) | $\Sigma_{t+1} := P_t$ (single-shot) |
| Cluster-GCN | $(v, cid)$ | `MATCH (v) WHERE v.cid ∈ S` | $1$ | random subset of cluster ids (yes, on $S$) | $\Sigma_{t+1} := P_t$ |

Notes: Cluster-GCN covered only with $cid$ pre-stored; METIS itself outside. Non-uniform random walks require transition probabilities pre-stored as edge properties — node2vec $(p, q)$ biased walks need precomputation, raw $(p, q)$ inside Cypher is outside the fragment.

**$\texttt{UPDATE}_t$ in Cypher.** The table elides the Cypher form because every $\texttt{UPDATE}_t$ above is a Lemma 1 list-op on $P_t$ and possibly $\Sigma_t^{\text{Cy}}$. Five patterns suffice:

1. **Replace** (uniform, GraphSAINT, Cluster-GCN, FastGCN layer-step):
    ```cypher
    WITH {nodes: P} AS state
    ```
    Rebind `state` with the selected list as its sole component.

2. **Append / multiset union** (GraphSAGE sampled, LADIES sampled):
    ```cypher
    WITH {frontier: state.frontier, sampled: state.sampled + P} AS state
    ```
    List concat = multiset union (Lemma 1); other components carried unchanged.

3. **Layer advance** (GraphSAGE frontier, LADIES frontier):
    ```cypher
    WITH {frontier: [x IN P | {p: x.c, l: x.l + 1}],
          sampled:  state.sampled} AS state
    ```
    List comprehension reprojects selected children to the next layer's frontier schema.

4. **Per-row state mutation** (random walk):
    ```cypher
    WITH {walks: [x IN P | {w: x.w, v: x.u, s: x.s + 1, prev: x.v}]} AS state
    ```
    Each chosen neighbour $u$ becomes the new $v$; old $v$ becomes `prev`.

5. **Group-aggregate increment** (PinSAGE visit counts):
    ```cypher
    UNWIND state.visits + [x IN P | {seed: x.seed, v: x.v, c: 1}] AS r
    WITH state, r.seed AS seed, r.v AS v, sum(r.c) AS c
    WITH {walks: state.walks, visits: collect({seed: seed, v: v, c: c})} AS state
    ```
    `UNWIND … sum(…) … collect(…)` realises multiset grouping (Lemma 1).

All five are read-only Cypher in the allowed fragment, hence Cypher-expressible. Each produces a new `state` map, so composing steps (Lemma 3) is a chain of `WITH {…} AS state` bindings that yields a single query for the whole algorithm.

### Thesis statement (tightened)

> Every bounded graph sampling algorithm whose state is a fixed-arity tuple of finite multisets over Cypher-representable schemas, and whose per-step transition decomposes into Cypher-expressible candidate generation, scoring, selection (against an external random tape), and update, is computable by a read-only Cypher query up to multiset equivalence. This class strictly includes uniform node/edge sampling, fixed-fanout neighborhood sampling, layer-wise importance sampling, LADIES-style dependent sampling, fixed-length random-walk sampling, and pre-clustered subgraph sampling.

---
