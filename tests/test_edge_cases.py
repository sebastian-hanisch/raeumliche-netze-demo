"""Randfälle: kleinste und größte Ausschnitte für jedes Modell, kollineare und doppelte Punkte, leere Kantenlisten, Gleichstände."""

import numpy as np
import pytest

import rnz_algorithm as A
import rnz_constants as C
import rnz_evaluation as ev
import rnz_scenario as S
import rnz_visualization as V


def test_every_model_runs_at_the_smallest_and_largest_section():
    for model in C.MODELS:
        for n in (C.N_MIN, C.N_MAX):
            a = ev.analyse(ev.Settings(model=model, n=n, k=C.K_MAX, scale=C.SCALE_MIN, seed=1))
            assert len(a.pos) == n and 0.0 <= a.stats.giant <= 1.0 and a.passed in range(5), (model, n)


def test_k_and_scale_at_their_bounds():
    for k in (C.K_MIN, C.K_MAX):
        a = ev.analyse(ev.Settings(model="knn", n=500, k=k, seed=1))
        assert a.stats.m >= 500 * k / 2                                       # jeder Punkt hat mindestens k Nachbarn, jede Kante zählt höchstens zweimal
    for scale in (C.SCALE_MIN, C.SCALE_MAX):
        assert ev.analyse(ev.Settings(model="waxman", n=500, scale=scale, seed=1)).stats.m > 0


def test_a_map_without_edges_and_a_model_without_edges():
    pos = np.array([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]])
    assert A.transitivity(3, []) == 0.0 and A.degrees(3, []).tolist() == [0, 0, 0] and A.crossings(pos, []) == 0
    assert A.giant_nodes(3, []).tolist() == [0]


def test_collinear_points_have_no_triangles_and_no_crossings():
    pos = np.array([[float(i), 0.0] for i in range(6)])
    edges = [(i, i + 1) for i in range(5)] + [(0, 2), (1, 3)]
    assert A.crossings(pos, edges) == 0 and A.transitivity(6, edges) > 0       # die Dreier (0,1,2) und (1,2,3) schließen sich über die langen Kanten


def test_equal_lengths_are_cut_in_a_fixed_order():
    pos = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])           # Quadrat: vier Kanten der Länge 1, zwei Diagonalen
    edges = [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]
    assert A.thin_shortest(pos, edges, 2) == [(0, 1), (0, 2)]                  # bei Gleichstand die vorn stehenden
    assert A.thin_shortest(pos, edges, 2) == A.thin_shortest(pos, list(edges), 2)


def test_the_map_figure_caps_the_number_of_drawn_edges():
    edges = [(i, i + 1) for i in range(20000)]
    assert len(V.select_edges(edges)) == V.MAX_DRAWN_EDGES and V.select_edges(edges)[0] == (0, 1) and V.select_edges(edges)[-1] == (19999, 20000)
    assert V.select_edges(edges[:10]) == edges[:10]
    pos = np.array([[float(i), 0.0] for i in range(20001)])
    fig = V.build_map(pos, edges, "#000", "x")
    assert fig.data[0].x.count(None) == V.MAX_DRAWN_EDGES


def test_the_map_figure_survives_an_empty_edge_list():
    fig = V.build_map(np.array([[0.0, 0.0], [5.0, 3.0]]), [], "#000", "x")
    assert len(fig.data[0].x) == 0 and fig.layout.yaxis.scaleanchor == "x"


def test_unknown_model_raises():
    with pytest.raises(ValueError):
        S.build_model("x", np.zeros((3, 2)), 2, 2, 30, np.random.default_rng(1))


def test_waxman_with_nearly_all_edges_wanted_keeps_nearly_all_candidates():
    pos = np.random.default_rng(1).random((50, 2)) * 100
    cand = A.delaunay_edges(pos)
    p = A.waxman_keep_probabilities(A.edge_lengths(pos, cand), 10.0, len(cand) - 1)
    assert p.sum() == pytest.approx(len(cand) - 1) and p.max() <= 1.0
