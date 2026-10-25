"""Jede Zahl der README wird hier nachgerechnet (DEMO-PLAYBOOK §4): über dieselben Auswertungsfunktionen wie in der App, mit den festen Plänen.

Die Zufallszahlen eines numpy-Generators sind über Versionen nicht stabil: die Messwerte der zufälligen Modelle gelten als Bänder um die in der README genannten Zahlen.
Die Karte und die deterministischen Modelle (k nächste Nachbarn, Delaunay nach Länge, Gabriel, geometrischer Zufallsgraph) sind bei gegebenen Standorten eindeutig."""

import pytest

import rnz_constants as C
import rnz_evaluation as ev


@pytest.fixture(scope="module")
def series():
    return ev.series_models()


@pytest.fixture(scope="module")
def full(series):
    return {r["model"]: r for r in series if r["requested"] is C.FULL_MAP}


@pytest.fixture(scope="module")
def presets():
    return {name: ev.analyse(ev.Settings(**p)) for name, p in C.PRESETS.items()}


def test_the_series_use_the_documented_fixed_plans():
    assert C.SERIES_SEED == 100000 and C.SERIES_RUNS == 3 and C.SERIES_NS == (500, 1500, 3000, None) and C.HOP_SOURCES == 40
    assert (C.N_MIN, C.N_MAX, C.DEFAULT_N) == (500, 5000, 2000)


def test_the_map(full):
    ref = full["map"]
    assert (ref["n"], ref["m"], ref["crossings"], ref["kmax"]) == (5072, 7621, 125, 8)
    assert ref["giant"] == 1.0 and ref["transitivity"] == pytest.approx(0.033, abs=0.002) and ref["hops"] == pytest.approx(30.9, abs=3.0)
    assert ref["len_median"] == pytest.approx(13, abs=1) and ref["len_p95"] == pytest.approx(89, abs=4)


def test_the_delaunay_triangulation_of_the_map_has_15193_edges():
    import rnz_algorithm as A
    import rnz_scenario as S

    pos, _ = S.make_map()
    assert len(A.delaunay_edges(pos)) == 15193


def test_the_geometric_graph_falls_apart_and_has_many_triangles(full):
    g = full["geometric"]
    assert g["m"] == pytest.approx(7621, abs=10) and g["giant"] <= 0.03 and g["transitivity"] == pytest.approx(0.73, abs=0.05)
    assert g["crossings"] == pytest.approx(5336, rel=0.1) and g["kmax"] == 18 and g["tv"] == pytest.approx(0.52, abs=0.05) and g["passed"] == 0
    assert g["len_p95"] == pytest.approx(14, abs=1.5) and g["len_p95"] < full["map"]["len_p95"] / 5


def test_the_nearest_neighbour_graph_falls_apart(full):
    k = full["knn"]
    assert k["m"] == 6482 and k["giant"] <= 0.03 and k["transitivity"] == pytest.approx(0.40, abs=0.04) and k["tv"] == pytest.approx(0.53, abs=0.05) and k["passed"] == 0


def test_the_delaunay_models(full):
    r, s = full["delaunay_random"], full["delaunay_short"]
    assert r["crossings"] == 0 and r["giant"] == pytest.approx(0.97, abs=0.02) and r["transitivity"] == pytest.approx(0.184, abs=0.02) and r["hops"] == pytest.approx(29.4, abs=3.0) and r["tv"] == pytest.approx(0.36, abs=0.05)
    assert s["crossings"] == 0 and s["giant"] <= 0.05 and s["transitivity"] == pytest.approx(0.474, abs=0.04) and s["passed"] == 0
    assert r["passed"] == 2 and r["checks"]["giant"] and r["checks"]["hops"] and not r["checks"]["transitivity"] and not r["checks"]["degrees"]


def test_the_gabriel_graph_matches_the_path_length_but_not_the_triangles(full):
    g = full["gabriel"]
    assert g["m"] == 9712 and g["giant"] == 1.0 and g["crossings"] == 0 and g["hops"] == pytest.approx(34.2, abs=3.0)
    assert g["transitivity"] == pytest.approx(0.274, abs=0.02) and g["transitivity"] > 5 * full["map"]["transitivity"] and g["tv"] == pytest.approx(0.317, abs=0.03)
    assert g["passed"] == 2 and g["checks"]["giant"] and g["checks"]["hops"] and not g["checks"]["transitivity"] and not g["checks"]["degrees"]
    assert g["m"] > full["map"]["m"] * 1.2                          # etwa 27 % mehr Kanten als die Karte


def test_waxman_and_the_random_graph(full):
    w, e = full["waxman"], full["gnm"]
    assert w["m"] == pytest.approx(7621, abs=300) and w["giant"] == pytest.approx(0.69, abs=0.12) and w["transitivity"] == pytest.approx(0.288, abs=0.03) and w["hops"] > 1.5 * full["map"]["hops"] and w["passed"] == 0
    assert e["crossings"] == -1 and e["giant"] == pytest.approx(0.94, abs=0.02) and e["transitivity"] < 0.003 and e["hops"] == pytest.approx(7.8, abs=1.0) and e["tv"] == pytest.approx(0.41, abs=0.05) and e["passed"] == 0
    assert e["len_median"] > 50 * full["map"]["len_median"]         # die Kanten sind hunderte Meter lang


def test_no_model_hits_more_than_two_of_four_in_any_size(series):
    best = {}
    for r in series:
        if r["model"] != "map":
            best[r["n"]] = max(best.get(r["n"], 0), r["passed"])
    assert set(best) == {500, 1500, 3000, 5072} and max(best.values()) == 2 and best[5072] == 2
    assert all(r["passed"] <= 2 for r in series if r["model"] != "map")


def test_every_model_misses_the_degree_distribution_and_the_triangles(series):
    for r in series:
        if r["model"] == "map":
            continue
        assert not r["checks"]["degrees"] and not r["checks"]["transitivity"], (r["n"], r["model"])
        assert r["tv"] >= 0.25                                            # kleinster Abstand der Gradverteilung über alle Modelle und Größen
    spatial = [r for r in series if r["requested"] is C.FULL_MAP and r["model"] not in ("map", "gnm")]
    assert min(r["transitivity"] for r in spatial) > 5 * next(r for r in series if r["requested"] is C.FULL_MAP and r["model"] == "map")["transitivity"]


def test_presets(presets):
    geo, knn, gab, short, er = (presets[n] for n in C.PRESETS)
    assert len(geo.pos) == 2000 and geo.passed == 0 and geo.stats.giant <= 0.06 and geo.stats.crossings == pytest.approx(1245, rel=0.15) and geo.ref.crossings == pytest.approx(58, abs=8)
    assert len(knn.pos) == 1500 and knn.passed == 0 and knn.stats.m == 1932 and knn.stats.transitivity == pytest.approx(0.425, abs=0.04)
    assert gab.passed == 2 and gab.stats.m == 2756 and gab.stats.crossings == 0 and gab.stats.hops == pytest.approx(20.9, abs=2.5)
    assert short.passed == 0 and short.stats.m == 2093 and short.stats.giant <= 0.1 and short.stats.crossings == 0
    assert er.passed == 0 and er.stats.crossings == -1 and er.stats.transitivity < 0.005 and er.stats.giant == pytest.approx(0.926, abs=0.02)


def test_the_verdict_texts_match_the_series(series, presets):
    kind, text = ev.verdict_series(series)
    assert kind == "info" and "ganzen Karte (5072 Knoten) trifft das beste Modell" in text and "2 von 4 Kennzahlen" in text and "kein Modell trifft in irgendeiner der Größen alle vier" in text
    kind, text = ev.verdict_model(presets["Gabriel-Graph"])
    assert kind == "info" and "trifft 2 von 4 Kennzahlen" in text and "Transitivität 0,255 (Karte 0,045)" in text and "Kreuzungen: 0 (Karte 54)" in text
