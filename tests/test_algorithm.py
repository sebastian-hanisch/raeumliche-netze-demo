"""Kern: jede Einheit an einer Mini-Instanz von Hand (DEMO-PLAYBOOK §1b)."""

import numpy as np
import pytest

import rnz_algorithm as A

FLAT = np.array([[0.0, 0.0], [4.0, 0.0], [2.0, 0.5]])                          # flaches Dreieck: C liegt im Kreis über AB
KITE = np.array([[0.0, 0.0], [4.0, 0.0], [2.0, 1.0], [2.0, -3.0]])             # Drachenviereck: die Delaunay-Diagonale ist CD, nicht AB


def test_normalize_by_hand():
    assert A.normalize([(3, 1), (1, 3), (2, 2), (0, 1)]) == [(0, 1), (1, 3)]       # sortiert, ohne Doppelte und Schleifen


def test_edge_lengths_by_hand(path4):
    pos, edges = path4
    assert A.edge_lengths(pos, edges).tolist() == [1.0, 2.0, 3.0]
    assert A.edge_lengths(np.array([[0.0, 0.0], [3.0, 4.0]]), [(0, 1)]).tolist() == [5.0]


def test_delaunay_edges_by_hand():
    assert A.delaunay_edges(FLAT) == [(0, 1), (0, 2), (1, 2)]
    # Umkreis von A, B, C hat den Mittelpunkt (2, −1,5) und Radius 2,5; D (Abstand 1,5) liegt darin: AB ist keine Delaunay-Kante, CD schon
    assert A.delaunay_edges(KITE) == [(0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]


def test_gabriel_edges_by_hand():
    assert A.gabriel_edges(FLAT) == [(0, 2), (1, 2)]               # C liegt im Kreis mit Durchmesser AB (Abstand 0,5 vom Mittelpunkt, Radius 2)
    assert A.gabriel_edges(KITE) == [(0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]
    assert A.gabriel_edges(FLAT, candidates=[(0, 1)]) == []        # mit gegebenen Kandidaten


def test_radius_for_edges_by_hand(path4):
    pos, _ = path4              # Paarabstände 1, 2, 3, 3, 5, 6
    assert A.radius_for_edges(pos, 1) == 1.0 and A.radius_for_edges(pos, 2) == 2.0 and A.radius_for_edges(pos, 3) == 3.0
    assert A.radius_for_edges(pos, 5) == 5.0 and A.radius_for_edges(pos, 6) == 6.0       # größer als der erste Suchradius


def test_geometric_edges_by_hand(path4):
    pos, _ = path4
    assert A.geometric_edges(pos, 2.0) == [(0, 1), (1, 2)]
    assert A.geometric_edges(pos, 3.0) == [(0, 1), (0, 2), (1, 2), (2, 3)]         # Abstand genau gleich dem Radius zählt
    assert A.geometric_edges(pos, 0.5) == []


def test_knn_edges_by_hand(path4):
    pos, _ = path4
    assert A.knn_edges(pos, 1) == [(0, 1), (1, 2), (2, 3)]         # Punkt 2: nächster ist 1 (Abstand 2) vor 3 (Abstand 3)
    assert len(A.knn_edges(pos, 3)) == 6 and len(A.knn_edges(pos, 10)) == 6          # k wird auf n − 1 begrenzt


def test_thinning_by_hand(path4):
    pos, _ = path4
    every = [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]       # Längen 1, 3, 6, 2, 5, 3
    assert A.thin_shortest(pos, every, 3) == [(0, 1), (0, 2), (1, 2)]               # bei Gleichstand (0,2) und (2,3) die vorn stehende
    assert A.thin_shortest(pos, every, 10) == every
    a = A.thin_random(every, 3, np.random.default_rng(1))
    assert len(a) == 3 and set(a) <= set(every) and a == A.thin_random(every, 3, np.random.default_rng(1))
    assert A.thin_random(every, 10, np.random.default_rng(1)) == every


def test_waxman_probabilities_by_hand():
    assert A.waxman_keep_probabilities([1, 1, 1, 1], 5.0, 2).tolist() == pytest.approx([0.5] * 4)       # gleiche Längen, erwartet 2 von 4
    assert A.waxman_keep_probabilities([1, 2], 5.0, 5).tolist() == [1.0, 1.0]                            # mehr gewünscht als Kandidaten
    p = A.waxman_keep_probabilities([0.0, 100.0], 1.0, 1.5)                                              # die kurze Kante wird auf 1 gekappt, der Rest geht an die lange
    assert p.tolist() == pytest.approx([1.0, 0.5])
    p = A.waxman_keep_probabilities([1.0, 2.0, 3.0], 1.0, 1.0)
    assert p.sum() == pytest.approx(1.0) and p[0] > p[1] > p[2]


def test_waxman_edges_keep_the_candidates_independently(path4):
    pos, edges = path4
    kept = A.waxman_edges(pos, edges, 5.0, 10, np.random.default_rng(1))       # m über der Zahl der Kandidaten: alle bleiben
    assert kept == edges


def test_gnm_edges_by_hand():
    e = A.gnm_edges(5, 4, np.random.default_rng(1))
    assert len(e) == len(set(e)) == 4 and all(0 <= u < v < 5 for u, v in e) and e == A.gnm_edges(5, 4, np.random.default_rng(1))
    assert A.gnm_edges(4, 6, np.random.default_rng(2)) == [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]       # alle Paare
    assert len(A.gnm_edges(4, 100, np.random.default_rng(2))) == 6                                                  # m wird auf alle Paare begrenzt


def test_degrees_and_giant_by_hand():
    edges = [(0, 1), (1, 2), (3, 4)]
    assert A.degrees(6, edges).tolist() == [1, 2, 1, 1, 1, 0]
    assert A.giant_nodes(6, edges).tolist() == [0, 1, 2]
    assert A.giant_nodes(4, [(0, 1), (2, 3)]).tolist() == [0, 1]                      # Gleichstand: die Komponente mit dem kleinsten Knoten


def test_transitivity_by_hand():
    assert A.transitivity(4, [(0, 1), (1, 2), (0, 2), (2, 3)]) == pytest.approx(3 * 1 / 5)       # ein Dreieck, Zweierwege 1 + 1 + 3 = 5
    assert A.transitivity(4, [(0, 1), (0, 2), (0, 3)]) == 0.0                                    # Stern
    assert A.transitivity(4, [(i, j) for i in range(4) for j in range(i + 1, 4)]) == pytest.approx(1.0)
    assert A.transitivity(3, [(0, 1)]) == 0.0                                                     # kein Zweierweg


def test_mean_hops_by_hand():
    path = [(0, 1), (1, 2), (2, 3)]
    assert A.mean_hops(4, path, [0]) == 2.0                    # Abstände 1, 2, 3
    assert A.mean_hops(4, path, [0, 3]) == 2.0
    assert A.mean_hops(4, path, [1]) == pytest.approx((1 + 1 + 2) / 3)
    assert A.mean_hops(4, [(0, 1), (1, 2)], [0]) == 1.5       # der isolierte Knoten zählt nicht


def test_segments_cross_by_hand():
    assert A.segments_cross((0, 0), (2, 2), (0, 2), (2, 0))                # Kreuz
    assert not A.segments_cross((0, 0), (2, 2), (2, 2), (3, 0))            # gemeinsamer Endpunkt
    assert not A.segments_cross((0, 0), (2, 0), (1, 0), (1, 2))            # T: eine Strecke endet auf der anderen
    assert not A.segments_cross((0, 0), (2, 0), (0, 1), (2, 1))            # parallel
    assert not A.segments_cross((0, 0), (2, 0), (1, 0), (3, 0))            # auf einer Geraden, überlappend
    assert not A.segments_cross((0, 0), (1, 1), (2, 0), (3, 1))            # weit auseinander


def test_crossings_by_hand(square):
    assert A.crossings(square, [(0, 2), (1, 3)]) == 1                       # die beiden Diagonalen
    assert A.crossings(square, [(0, 1), (1, 2), (2, 3), (0, 3)]) == 0       # Seiten teilen Endpunkte oder liegen weit auseinander
    assert A.crossings(square, [(0, 2), (1, 3), (0, 1)]) == 1
    assert A.crossings(square, [(0, 2), (1, 3), (0, 4)]) == 1               # (0,4) liegt auf der Diagonale (0,2) und teilt den Endpunkt 0
    assert A.crossings(square, [(0, 2), (1, 3)], limit=1) == 1
    assert A.crossings(square, []) == 0


def test_degree_shares_and_total_variation_by_hand():
    s = A.degree_shares(4, [(0, 1), (1, 2), (2, 3)], kmax=3)       # Grade 1, 2, 2, 1
    assert s.tolist() == [0.0, 0.5, 0.5, 0.0]
    s2 = A.degree_shares(5, [(0, i) for i in range(1, 5)], kmax=3)  # Stern: Grad 4 (≥ 3) und viermal Grad 1
    assert s2.tolist() == [0.0, 0.8, 0.0, 0.2]
    assert A.total_variation(s, s2) == pytest.approx(0.5 * (0.3 + 0.5 + 0.2))
    assert A.total_variation([0.5, 0.5], [0.5, 0.5]) == 0.0 and A.total_variation([1, 0], [0, 1]) == 1.0


def test_net_stats_by_hand(path4):
    pos, edges = path4
    st = A.net_stats(4, edges, pos, [0, 3])
    assert (st.m, st.giant, st.transitivity, st.hops, st.crossings, st.kmax) == (3, 1.0, 0.0, 2.0, 0, 2)
    assert st.len_median == 2.0 and st.len_p95 == pytest.approx(2.9)
    assert A.net_stats(4, edges, pos, [0], count_crossings=False).crossings == -1
    assert A.net_stats(4, [(0, 1), (2, 3)], pos, [2]).giant == 0.5                       # die Quelle liegt in der größten Komponente


def test_net_stats_falls_back_to_a_node_of_the_giant_component_when_no_source_lies_in_it(path4):
    pos, _ = path4
    st = A.net_stats(4, [(0, 1), (1, 2)], pos, [3])          # die einzige Quelle ist isoliert
    assert st.hops == 1.5


def test_pick_sources_are_sorted_and_distinct():
    s = A.pick_sources(10, 4, np.random.default_rng(1))
    assert s.tolist() == sorted(set(s.tolist())) and len(s) == 4 and len(A.pick_sources(3, 10, np.random.default_rng(1))) == 3
