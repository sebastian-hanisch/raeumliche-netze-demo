"""Szenario und Auswertung: Modelle auf der Karte, Kantenzahlen, Determinismus, Messreihe (Struktur und Richtung), feste Pläne."""

import numpy as np
import pytest

import rnz_algorithm as A
import rnz_constants as C
import rnz_evaluation as ev
import rnz_scenario as S


@pytest.fixture(scope="module")
def sec():
    return S.make_map(600)


def test_build_model_edge_counts(sec):
    pos, edges = sec
    m = len(edges)
    rng = lambda: np.random.default_rng(1)       # noqa: E731
    for model in ("geometric", "delaunay_random", "delaunay_short", "gnm"):
        assert len(S.build_model(model, pos, m, 2, 30, rng())) == m, model          # diese Modelle bekommen genau die Kantenzahl der Karte
    assert abs(len(S.build_model("waxman", pos, m, 2, 30, rng())) - m) < 6 * np.sqrt(m)       # erwartet m, zufällig
    gab = S.build_model("gabriel", pos, m, 2, 30, rng())
    assert set(gab) <= set(A.delaunay_edges(pos)) and len(gab) > m * 0.9


def test_build_model_k_and_scale_reach_the_models(sec):
    pos, edges = sec
    rng = lambda: np.random.default_rng(1)       # noqa: E731
    k1, k3 = S.build_model("knn", pos, len(edges), 1, 30, rng()), S.build_model("knn", pos, len(edges), 3, 30, rng())
    assert len(k1) < len(k3) and set(k1) <= set(k3)
    w10, w200 = S.build_model("waxman", pos, len(edges), 2, 10, rng()), S.build_model("waxman", pos, len(edges), 2, 200, rng())
    assert np.median(A.edge_lengths(pos, w10)) < np.median(A.edge_lengths(pos, w200))      # kleine Skala: kürzere Kanten


def test_build_model_is_deterministic_and_rejects_unknown(sec):
    pos, edges = sec
    for model in C.MODELS:
        assert S.build_model(model, pos, len(edges), 2, 30, np.random.default_rng(3)) == S.build_model(model, pos, len(edges), 2, 30, np.random.default_rng(3))
    with pytest.raises(ValueError):
        S.build_model("zufall", pos, 10, 2, 30, np.random.default_rng(1))


def test_the_map_is_cached_and_the_section_is_nested_in_size():
    a, b = S.make_map(300), S.make_map(300)
    assert (a[0] == b[0]).all() and a[1] == b[1]
    assert len(S.make_map(300)[1]) < len(S.make_map(900)[1])


def test_analyse_is_deterministic_and_consistent():
    s = ev.Settings(model="gabriel", n=500, seed=3)
    a, b = ev.analyse(s), ev.analyse(s)
    assert a.model_edges == b.model_edges and a.ref.m == len(a.ref_edges) == a.ref.m and a.stats.m == len(a.model_edges)
    assert a.tv == pytest.approx(A.total_variation(a.stats.shares, a.ref.shares)) and a.passed == sum(a.checks.values()) and set(a.checks) == set(ev.MEASURES)
    assert a.stats.crossings == 0 and a.ref.crossings > 0                       # der Gabriel-Graph ist kreuzungsfrei, die Karte hat Streckenkreuzungen


def test_analyse_skips_the_crossing_count_without_space():
    a = ev.analyse(ev.Settings(model="gnm", n=500, seed=1))
    assert a.stats.crossings == -1 and a.stats.m == a.ref.m


def test_the_random_seed_changes_only_the_random_models():
    same = [ev.analyse(ev.Settings(model="gabriel", n=500, seed=s)).model_edges for s in (1, 2)]
    assert same[0] == same[1]
    diff = [ev.analyse(ev.Settings(model="gnm", n=500, seed=s)).model_edges for s in (1, 2)]
    assert diff[0] != diff[1]


def test_series_structure_and_direction():
    rows = ev.series_models(ns=(400, None), runs=2, models=("gabriel", "gnm", "geometric"))
    assert [r["requested"] for r in rows] == [400] * 4 + [None] * 4 and [r["model"] for r in rows[:4]] == ["map", "gabriel", "gnm", "geometric"]
    full = {r["model"]: r for r in rows if r["requested"] is None}
    assert full["map"]["n"] == 5072 and full["map"]["m"] == 7621 and full["map"]["crossings"] == 125 and full["map"]["passed"] == 4
    assert full["gabriel"]["crossings"] == 0 and full["gnm"]["crossings"] == -1 and full["geometric"]["giant"] < 0.1
    assert full["gabriel"]["giant"] == 1.0 and full["gabriel"]["transitivity"] > 3 * full["map"]["transitivity"] and full["gnm"]["transitivity"] < full["map"]["transitivity"] / 5
    assert all(set(r["checks"]) == set(ev.MEASURES) for r in rows if r["model"] != "map") and all(0 <= r["passed"] <= 4 for r in rows)


def test_series_uses_one_run_for_deterministic_models(monkeypatch):
    calls = []
    real = A.net_stats

    def spy(*a, **kw):
        calls.append(1)
        return real(*a, **kw)

    monkeypatch.setattr(A, "net_stats", spy)
    ev.series_models(ns=(300,), runs=3, models=("gabriel", "gnm"))
    assert len(calls) == 1 + 1 + 3                     # Karte, Gabriel einmal, G(n, m) dreimal


def test_the_fixed_plans_and_defaults_are_the_documented_ones():
    assert C.SERIES_SEED == 100000 and C.SERIES_RUNS == 3 and C.SERIES_NS == (500, 1500, 3000, None) and C.HOP_SOURCES == 40
    s = ev.Settings()
    assert (s.model, s.n, s.k, s.scale, s.seed) == ("geometric", 2000, 2, 30, 35) and set(C.MODELS) == set(ev.RANDOM_MODELS) | {"geometric", "knn", "delaunay_short", "gabriel"}
