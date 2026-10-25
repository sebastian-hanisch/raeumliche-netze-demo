"""Orakel-Tests mit unabhängigem Rechenweg (ORACLE-PLAYBOOK): Brute-Force-Aufzählung (leere Umkreise für Delaunay, leere Kreise für Gabriel, alle Paare für Radius und Nachbarn),
`networkx` für Transitivität, Weglängen und Komponenten, Aufzählung mit exakten Brüchen für Streckenkreuzungen, Chi-Quadrat für die Gleichverteilung von G(n, m)."""

import itertools
import math
from fractions import Fraction

import numpy as np
import pytest
from scipy import stats

import rnz_algorithm as A
import rnz_scenario as S


def random_points(n, seed):
    return np.random.default_rng(seed).random((n, 2)) * 100.0


def circumcircle(a, b, c):
    """Mittelpunkt und Radius des Umkreises (None bei Kollinearität)."""
    ax, ay = a
    bx, by = b
    cx, cy = c
    d = 2 * (ax * (by - cy) + bx * (cy - ay) + cx * (ay - by))
    if abs(d) < 1e-12:
        return None
    ux = ((ax**2 + ay**2) * (by - cy) + (bx**2 + by**2) * (cy - ay) + (cx**2 + cy**2) * (ay - by)) / d
    uy = ((ax**2 + ay**2) * (cx - bx) + (bx**2 + by**2) * (ax - cx) + (cx**2 + cy**2) * (bx - ax)) / d
    return (ux, uy), math.hypot(ax - ux, ay - uy)


def brute_delaunay(pos):
    """Alle Dreiecke mit leerem Umkreis (kein weiterer Punkt strikt im Inneren) und ihre Kanten."""
    n = len(pos)
    edges = set()
    for i, j, k in itertools.combinations(range(n), 3):
        cc = circumcircle(pos[i], pos[j], pos[k])
        if cc is None:
            continue
        (cx, cy), r = cc
        if all(math.hypot(pos[w][0] - cx, pos[w][1] - cy) >= r * (1 - 1e-9) for w in range(n) if w not in (i, j, k)):
            edges |= {(i, j), (i, k), (j, k)}
    return sorted(edges)


def test_delaunay_equals_the_empty_circumcircle_enumeration():
    for seed in (1, 2, 3):
        pos = random_points(40, seed)
        assert A.delaunay_edges(pos) == brute_delaunay(pos)


def test_gabriel_equals_the_empty_disc_enumeration_over_all_pairs():
    for seed in (4, 5):
        pos = random_points(60, seed)
        brute = []
        for u, v in itertools.combinations(range(len(pos)), 2):
            c = (pos[u] + pos[v]) / 2
            r = math.hypot(*(pos[u] - pos[v])) / 2
            if all(math.hypot(*(pos[w] - c)) >= r * (1 - 1e-9) for w in range(len(pos)) if w not in (u, v)):
                brute.append((u, v))
        got = A.gabriel_edges(pos)
        assert got == brute
        assert set(got) <= set(A.delaunay_edges(pos))              # der Gabriel-Graph ist ein Teilgraph der Delaunay-Triangulierung


def test_geometric_graph_and_radius_equal_the_enumeration_of_all_pairs():
    pos = random_points(80, 6)
    d = {(u, v): math.hypot(*(pos[u] - pos[v])) for u, v in itertools.combinations(range(80), 2)}
    for m in (10, 100, 500):
        r = A.radius_for_edges(pos, m)
        assert r == pytest.approx(sorted(d.values())[m - 1])
        assert A.geometric_edges(pos, r) == sorted(e for e, x in d.items() if x <= r * (1 + A.RADIUS_SLACK))
        assert len(A.geometric_edges(pos, r)) == m          # genau m Kanten (keine Gleichstände bei Zufallspunkten)


def test_knn_equals_the_sorted_distance_enumeration():
    pos = random_points(70, 7)
    for k in (1, 2, 3):
        brute = set()
        for u in range(70):
            near = sorted((math.hypot(*(pos[u] - pos[v])), v) for v in range(70) if v != u)[:k]
            brute |= {(min(u, v), max(u, v)) for _, v in near}
        assert A.knn_edges(pos, k) == sorted(brute)


def random_graph(n, m, seed):
    return A.gnm_edges(n, m, np.random.default_rng(seed))


def test_transitivity_equals_networkx():
    nx = pytest.importorskip("networkx")
    for seed, (n, m) in enumerate([(30, 60), (60, 90), (100, 400), (200, 250)]):
        edges = random_graph(n, m, seed)
        g = nx.Graph()
        g.add_nodes_from(range(n))
        g.add_edges_from(edges)
        assert A.transitivity(n, edges) == pytest.approx(nx.transitivity(g))


def test_mean_hops_and_giant_equal_networkx():
    nx = pytest.importorskip("networkx")
    n = 120
    edges = random_graph(n, 150, 11)
    g = nx.Graph()
    g.add_nodes_from(range(n))
    g.add_edges_from(edges)
    giant = max(nx.connected_components(g), key=len)
    assert A.giant_nodes(n, edges).tolist() == sorted(giant)
    src = sorted(giant)[:10]
    total, count = 0, 0
    for s in src:
        for t, d in nx.single_source_shortest_path_length(g, s).items():
            if t != s:
                total += d
                count += 1
    assert A.mean_hops(n, edges, src) == pytest.approx(total / count)


def exact_cross(p, q, r, s):
    """Echte Kreuzung zweier Strecken mit ganzzahligen Koordinaten, mit exakten Brüchen (Parameter t, u des Schnittpunkts strikt zwischen 0 und 1)."""
    (x1, y1), (x2, y2), (x3, y3), (x4, y4) = p, q, r, s
    den = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
    if den == 0:
        return False
    t = Fraction((x1 - x3) * (y3 - y4) - (y1 - y3) * (x3 - x4), den)
    u = Fraction((x1 - x3) * (y1 - y2) - (y1 - y3) * (x1 - x2), den)
    return 0 < t < 1 and 0 < u < 1


def test_crossings_equal_the_exact_enumeration():
    rng = np.random.default_rng(12)
    pos = rng.integers(0, 40, size=(25, 2))
    edges = A.gnm_edges(25, 45, rng)
    brute = 0
    for (a, b), (c, d) in itertools.combinations(edges, 2):
        if len({a, b, c, d}) == 4 and exact_cross(tuple(pos[a]), tuple(pos[b]), tuple(pos[c]), tuple(pos[d])):
            brute += 1
    assert brute > 5
    assert A.crossings(pos.astype(float), edges) == brute


def test_gnm_is_uniform_over_all_edge_sets():
    n, m, draws = 5, 3, 6000                                 # C(10, 3) = 120 mögliche Kantenmengen
    rng = np.random.default_rng(13)
    counts = {}
    for _ in range(draws):
        e = tuple(A.gnm_edges(n, m, rng))
        counts[e] = counts.get(e, 0) + 1
    assert len(counts) == 120
    assert stats.chisquare(list(counts.values())).pvalue > 0.001


def test_waxman_expected_edge_count_is_m_and_short_edges_stay_more_often():
    pos = random_points(300, 14)
    cand = A.delaunay_edges(pos)
    m = len(cand) // 2
    lens = A.edge_lengths(pos, cand)
    runs = [A.waxman_edges(pos, cand, float(np.median(lens)), m, np.random.default_rng(100 + j)) for j in range(40)]
    sizes = np.array([len(r) for r in runs])
    assert abs(sizes.mean() - m) < 4 * math.sqrt(m) / math.sqrt(40) + 1
    kept = {e: sum(e in set(r) for r in runs) for e in cand}
    short = [kept[e] for e, x in zip(cand, lens) if x < np.percentile(lens, 25)]
    long = [kept[e] for e, x in zip(cand, lens) if x > np.percentile(lens, 75)]
    assert np.mean(short) > 1.5 * np.mean(long)


def test_the_map_data_has_the_documented_size():
    pos, edges = S.make_map()
    assert len(pos) == 5072 and len(edges) == 7621
    assert A.giant_nodes(len(pos), edges).size == 5072
    lens = A.edge_lengths(pos, edges)
    assert float(np.median(lens)) == pytest.approx(13, abs=2) and A.crossings(pos, edges) == 125


def test_the_section_is_a_connected_ball_around_the_middle():
    for n in (300, 1000):
        pos, edges = S.make_map(n)
        assert len(pos) == n and A.giant_nodes(n, edges).size == n
        assert all(0 <= u < v < n for u, v in edges)
        centre = np.median(S.make_map()[0], axis=0)
        assert np.linalg.norm(pos - centre, axis=1).min() < 60


def test_radius_equals_the_sorted_pair_distance_for_every_edge_count():
    pos = random_points(40, 21)
    d = sorted(math.hypot(*(pos[u] - pos[v])) for u, v in itertools.combinations(range(40), 2))
    for m in range(1, 120):
        assert A.radius_for_edges(pos, m) == pytest.approx(d[m - 1]), m


def test_crossing_test_does_not_depend_on_the_scale_of_the_coordinates():
    """Mit Koordinaten unter 1 sind die Orientierungswerte kleiner als 1: die Vorzeichenprüfung darf nicht an eine Schranke gebunden sein."""
    rng = np.random.default_rng(22)
    for scale in (1.0, 0.5, 0.125, 2.0 ** -10):          # Zweierpotenzen: die Skalierung ist in Gleitkommazahlen exakt, Berührungen bleiben Berührungen
        trials = crossing = 0
        for _ in range(400):
            p, q, r, s = (tuple(int(x) for x in rng.integers(0, 12, size=2)) for _ in range(4))
            if len({p, q, r, s}) < 4:
                continue
            want = exact_cross(p, q, r, s)
            assert A.segments_cross(*(tuple(c * scale for c in pt) for pt in (p, q, r, s))) == want, (scale, p, q, r, s)
            trials += 1
            crossing += want
        assert trials > 300 and crossing > 20           # genug echte Kreuzungen, damit der Vergleich etwas prüft


def test_crossing_test_by_hand_with_small_coordinates():
    assert A.segments_cross((0, 0), (0.5, 0.5), (0, 0.5), (0.5, 0))            # Kreuz im Quadrat der Seitenlänge ½
    assert not A.segments_cross((0, 0), (0.25, 0.25), (0, 0.5), (0.5, 0))      # die Strecken erreichen sich nicht
