"""Kern: räumliche Zufallsgraphen auf gegebenen Standorten und Kennzahlen zum Vergleich mit einem Straßennetz.

Modelle auf denselben Punkten: geometrischer Zufallsgraph (Kante bei Abstand höchstens Radius), k nächste Nachbarn (vereinigt), Delaunay-Kanten (zufällig oder nach Länge gekürzt),
Gabriel-Graph, Waxman-Auswahl aus den Delaunay-Kanten (Wahrscheinlichkeit fällt mit der Länge) und der Erdős–Rényi-Graph G(n, m) ohne Raumbezug.
Kennzahlen: Riesenkomponente, Transitivität, mittlere Weglänge, Zahl der Streckenkreuzungen, Gradverteilung, Kantenlänge."""

from dataclasses import dataclass

import numpy as np
from scipy import sparse
from scipy.sparse import csgraph
from scipy.spatial import Delaunay, cKDTree


def normalize(edges):
    """Kanten als sortierte Paare (u < v), ohne Selbstschleifen und Doppelte, aufsteigend sortiert."""
    return sorted({(min(int(u), int(v)), max(int(u), int(v))) for u, v in edges if int(u) != int(v)})


def edge_lengths(pos, edges):
    """Euklidische Länge jeder Kante."""
    pos = np.asarray(pos, dtype=float)
    e = np.asarray(edges, dtype=int).reshape(-1, 2)
    return np.hypot(pos[e[:, 0], 0] - pos[e[:, 1], 0], pos[e[:, 0], 1] - pos[e[:, 1], 1])


# --- Modelle auf gegebenen Standorten ------------------------------------------------------------------------------------------------------


def delaunay_edges(pos):
    """Kanten der Delaunay-Triangulierung (`scipy.spatial.Delaunay`), normiert."""
    tri = Delaunay(np.asarray(pos, dtype=float))
    e = set()
    for s in tri.simplices:
        for a, b in ((0, 1), (1, 2), (0, 2)):
            e.add((min(int(s[a]), int(s[b])), max(int(s[a]), int(s[b]))))
    return sorted(e)


def gabriel_edges(pos, candidates=None):
    """Gabriel-Kanten: (u, v) bleibt, wenn kein anderer Punkt im offenen Kreis mit dem Durchmesser uv liegt. Der Gabriel-Graph ist ein Teilgraph der Delaunay-Triangulierung,
    deshalb genügen deren Kanten als Kandidaten."""
    pos = np.asarray(pos, dtype=float)
    cand = delaunay_edges(pos) if candidates is None else list(candidates)
    tree = cKDTree(pos)
    out = []
    for u, v in cand:
        c = (pos[u] + pos[v]) / 2.0
        r = float(np.hypot(*(pos[u] - pos[v]))) / 2.0
        inside = [w for w in tree.query_ball_point(c, r * (1.0 - 1e-9)) if w != u and w != v]
        if not inside:
            out.append((u, v))
    return out


def radius_for_edges(pos, m: int) -> float:
    """Kleinster Radius, bei dem der geometrische Zufallsgraph mindestens m Kanten hat (der m-kleinste Paarabstand)."""
    pos = np.asarray(pos, dtype=float)
    tree = cKDTree(pos)
    hi = float(np.median(cKDTree(pos).query(pos, k=2)[0][:, 1])) * 2.0 + 1e-9
    while True:
        pairs = tree.query_pairs(hi, output_type="ndarray")
        if len(pairs) >= m:
            d = np.hypot(pos[pairs[:, 0], 0] - pos[pairs[:, 1], 0], pos[pairs[:, 0], 1] - pos[pairs[:, 1], 1])
            return float(np.partition(d, m - 1)[m - 1])
        hi *= 2.0


RADIUS_SLACK = 1e-12      # relative Zugabe: der Baum rechnet den Abstand minimal anders als `edge_lengths`, ein Paar genau auf dem Radius darf nicht herausfallen


def geometric_edges(pos, radius: float):
    """Kanten zwischen allen Punktpaaren mit Abstand höchstens `radius` (geometrischer Zufallsgraph; mit der Zugabe RADIUS_SLACK, damit der m-kleinste Abstand sicher mitzählt)."""
    pairs = cKDTree(np.asarray(pos, dtype=float)).query_pairs(radius * (1.0 + RADIUS_SLACK), output_type="ndarray")
    return normalize(map(tuple, pairs))


def knn_edges(pos, k: int):
    """Vereinigung der Kanten zu den k nächsten Nachbarn jedes Punktes (ungerichtet)."""
    pos = np.asarray(pos, dtype=float)
    k = min(k, len(pos) - 1)
    _, idx = cKDTree(pos).query(pos, k=k + 1)
    return normalize((i, int(j)) for i in range(len(pos)) for j in idx[i, 1:])


def thin_random(edges, m: int, rng: np.random.Generator):
    """m zufällig gewählte Kanten aus `edges` (ohne Zurücklegen)."""
    edges = list(edges)
    m = min(m, len(edges))
    return normalize(edges[i] for i in rng.choice(len(edges), size=m, replace=False))


def thin_shortest(pos, edges, m: int):
    """Die m kürzesten Kanten aus `edges` (bei gleicher Länge die mit dem kleineren Index)."""
    edges = list(edges)
    order = np.argsort(edge_lengths(pos, edges), kind="stable")
    return normalize(edges[i] for i in order[:m])


def waxman_keep_probabilities(lengths, scale: float, m: int):
    """Behaltewahrscheinlichkeit je Kandidat: min(1, c · exp(−Länge / scale)); c so gewählt, dass die erwartete Kantenzahl m ist (Bisektion; höchstens alle Kandidaten)."""
    lengths = np.asarray(lengths, dtype=float)
    base = np.exp(-lengths / scale)
    if m >= len(lengths):
        return np.ones(len(lengths))
    lo, hi = 0.0, 1.0
    while np.minimum(1.0, hi * base).sum() < m:
        hi *= 2.0
    for _ in range(80):
        mid = (lo + hi) / 2.0
        if np.minimum(1.0, mid * base).sum() < m:
            lo = mid
        else:
            hi = mid
    return np.minimum(1.0, hi * base)


def waxman_edges(pos, candidates, scale: float, m: int, rng: np.random.Generator):
    """Jede Kandidatenkante bleibt mit der Wahrscheinlichkeit aus `waxman_keep_probabilities` (erwartet m Kanten)."""
    candidates = list(candidates)
    p = waxman_keep_probabilities(edge_lengths(pos, candidates), scale, m)
    keep = rng.random(len(candidates)) < p
    return [candidates[i] for i in np.flatnonzero(keep)]


def gnm_edges(n: int, m: int, rng: np.random.Generator):
    """m verschiedene Kanten, gleichverteilt unter allen Paaren (Erdős–Rényi G(n, m), ohne Raumbezug)."""
    total = n * (n - 1) // 2
    m = min(m, total)
    out = set()
    while len(out) < m:
        need = m - len(out)
        a = rng.integers(0, n, size=2 * need + 8)
        b = rng.integers(0, n, size=2 * need + 8)
        for u, v in zip(a, b):
            if u != v:
                out.add((min(int(u), int(v)), max(int(u), int(v))))
                if len(out) == m:
                    break
    return sorted(out)


# --- Kennzahlen ------------------------------------------------------------------------------------------------------------------------------


def adjacency(n: int, edges):
    e = np.asarray(edges, dtype=int).reshape(-1, 2)
    a = sparse.coo_matrix((np.ones(len(e)), (e[:, 0], e[:, 1])), shape=(n, n))
    return (a + a.T).tocsr()


def degrees(n: int, edges):
    e = np.asarray(edges, dtype=int).reshape(-1, 2)
    return np.bincount(e.ravel(), minlength=n)


def giant_nodes(n: int, edges):
    """Knoten der größten Zusammenhangskomponente (bei Gleichstand die mit dem kleinsten Knoten)."""
    _, labels = csgraph.connected_components(adjacency(n, edges), directed=False)
    sizes = np.bincount(labels)
    return np.flatnonzero(labels == int(np.argmax(sizes)))


def transitivity(n: int, edges) -> float:
    """Anteil geschlossener Dreier unter allen verbundenen Dreiern: 3 · Dreiecke / Zahl der Zweierwege (null, wenn es keinen Zweierweg gibt)."""
    a = adjacency(n, edges)
    deg = np.asarray(a.sum(axis=1)).ravel()
    wedges = float((deg * (deg - 1) / 2.0).sum())
    if wedges == 0:
        return 0.0
    triangles = float(a.multiply(a @ a).sum()) / 6.0
    return 3.0 * triangles / wedges


def mean_hops(n: int, edges, sources) -> float:
    """Mittlere Zahl der Kanten auf kürzesten Wegen von den Quellen zu allen anderen erreichbaren Knoten (die Quelle selbst zählt nicht)."""
    sources = np.asarray(sources, dtype=int)
    d = csgraph.shortest_path(adjacency(n, edges), method="D", directed=False, unweighted=True, indices=sources)
    d[np.arange(len(sources)), sources] = np.inf
    finite = np.isfinite(d)
    return float(d[finite].sum() / finite.sum())


def _orient(ax, ay, bx, by, cx, cy):
    return (bx - ax) * (cy - ay) - (by - ay) * (cx - ax)


def segments_cross(p, q, r, s) -> bool:
    """Kreuzen sich die Strecken pq und rs echt (innere Schnittpunkt, keine gemeinsamen Endpunkte, keine Berührung)?"""
    d1 = _orient(*r, *s, *p)
    d2 = _orient(*r, *s, *q)
    d3 = _orient(*p, *q, *r)
    d4 = _orient(*p, *q, *s)
    return ((d1 > 0) != (d2 > 0)) and ((d3 > 0) != (d4 > 0)) and d1 != 0 and d2 != 0 and d3 != 0 and d4 != 0


def crossings(pos, edges, limit: int | None = None) -> int:
    """Zahl der Kantenpaare, deren Strecken sich echt kreuzen (Kanten mit gemeinsamem Endpunkt zählen nicht). Sweep über den kleinsten x-Wert mit Prüfung der Umrandungen;
    mit `limit` bricht die Zählung ab, sobald sie erreicht ist (Rückgabe: limit)."""
    pos = np.asarray(pos, dtype=float)
    e = list(edges)
    xs0 = np.array([min(pos[u][0], pos[v][0]) for u, v in e])
    xs1 = np.array([max(pos[u][0], pos[v][0]) for u, v in e])
    ys0 = np.array([min(pos[u][1], pos[v][1]) for u, v in e])
    ys1 = np.array([max(pos[u][1], pos[v][1]) for u, v in e])
    order = np.argsort(xs0, kind="stable")
    count = 0
    for ii in range(len(order)):
        i = int(order[ii])
        a, b = e[i]
        for jj in range(ii + 1, len(order)):
            j = int(order[jj])
            if xs0[j] > xs1[i]:
                break
            if ys0[j] > ys1[i] or ys1[j] < ys0[i]:
                continue
            c, d = e[j]
            if len({a, b, c, d}) == 4 and segments_cross(pos[a], pos[b], pos[c], pos[d]):
                count += 1
                if limit is not None and count >= limit:
                    return limit
    return count


def degree_shares(n: int, edges, kmax: int = 9):
    """Anteil der Knoten mit Grad 0, 1, …, kmax − 1 und, im letzten Eintrag, mit Grad ≥ kmax."""
    deg = np.bincount(np.minimum(degrees(n, edges), kmax), minlength=kmax + 1)
    return deg / n


def total_variation(p, q) -> float:
    """Totalvariationsabstand zweier Verteilungen auf denselben Werten: ½ Σ |p − q|."""
    return 0.5 * float(np.abs(np.asarray(p, dtype=float) - np.asarray(q, dtype=float)).sum())


@dataclass(eq=False)
class NetStats:
    m: int
    giant: float
    transitivity: float
    hops: float
    crossings: int
    kmax: int
    shares: np.ndarray
    len_median: float
    len_p95: float


def net_stats(n: int, edges, pos, sources, count_crossings: bool = True) -> NetStats:
    """Alle Kennzahlen eines Netzes; `sources`: Quellen für die mittlere Weglänge (innerhalb der größten Komponente). Ohne `count_crossings` steht bei den Kreuzungen −1
    (für Graphen ohne Raumbezug, deren Kanten über die ganze Karte laufen und deren Zählung quadratisch dauert)."""
    edges = list(edges)
    giant = giant_nodes(n, edges)
    src = np.asarray(sources, dtype=int)
    src = src[np.isin(src, giant)]
    if len(src) == 0:
        src = giant[:1]
    lens = edge_lengths(pos, edges) if edges else np.zeros(1)
    return NetStats(len(edges), len(giant) / n, transitivity(n, edges), mean_hops(n, edges, src), crossings(pos, edges) if count_crossings else -1, int(degrees(n, edges).max()),
                    degree_shares(n, edges), float(np.median(lens)), float(np.percentile(lens, 95)))


def pick_sources(n: int, k: int, rng: np.random.Generator):
    """k verschiedene Knoten als Quellen der Weglängenmessung."""
    return np.sort(rng.choice(n, size=min(k, n), replace=False))

