"""Vergleichsregeln und Verdikte (reine Funktionen): Schwellen an dyadischen Werten, jede Zahl im Text an handgesetzten Werten."""

import numpy as np

import rnz_algorithm as A
import rnz_constants as C
import rnz_evaluation as ev
from rnz_evaluation import Analysis, Settings


def stats(m=100, giant=1.0, trans=0.25, hops=4.0, cross=0, kmax=6, shares=None, med=12.0, p95=80.0):
    return A.NetStats(m, giant, trans, hops, cross, kmax, np.array(shares if shares is not None else [0.0, 0.25, 0.25, 0.25, 0.25]), med, p95)


def make(ref=None, st=None, tv=0.05, model="geometric"):
    ref, st = ref or stats(), st or stats()
    return Analysis(Settings(model=model), np.zeros((2, 2)), [], [], ref, st, tv, ev.matches(ref, st, tv))


def test_matches_giant_boundary(monkeypatch):
    monkeypatch.setattr(C, "GIANT_TOL", 0.0625)                          # dyadisch: der Rand ist exakt darstellbar
    ref = stats(giant=1.0)
    assert ev.matches(ref, stats(giant=0.9375), 0.0)["giant"] is True and ev.matches(ref, stats(giant=0.9374), 0.0)["giant"] is False


def test_matches_transitivity_boundaries_and_zero_reference():
    ref = stats(trans=0.25)
    assert ev.matches(ref, stats(trans=0.5), 0.0)["transitivity"] and ev.matches(ref, stats(trans=0.125), 0.0)["transitivity"]       # Verhältnis genau 2 und genau ½
    assert not ev.matches(ref, stats(trans=0.5001), 0.0)["transitivity"] and not ev.matches(ref, stats(trans=0.1249), 0.0)["transitivity"]
    assert not ev.matches(stats(trans=0.0), stats(trans=0.0), 0.0)["transitivity"]                                                  # Karte ohne Dreiecke: kein Verhältnis


def test_matches_hops_boundaries_and_the_giant_condition():
    ref = stats(hops=4.0)
    assert ev.matches(ref, stats(hops=3.0), 0.0)["hops"] and not ev.matches(ref, stats(hops=2.99), 0.0)["hops"]                       # ¾
    ref3 = stats(hops=3.0)
    assert ev.matches(ref3, stats(hops=4.0), 0.0)["hops"] and not ev.matches(ref3, stats(hops=4.01), 0.0)["hops"]                    # 4/3
    assert ev.matches(ref, stats(hops=4.0, giant=0.5), 0.0)["hops"] and not ev.matches(ref, stats(hops=4.0, giant=0.4999), 0.0)["hops"]    # ohne große Komponente zählt die Weglänge nicht


def test_matches_degree_distance_boundary(monkeypatch):
    monkeypatch.setattr(C, "TV_TOL", 0.125)
    assert ev.matches(stats(), stats(), 0.125)["degrees"] and not ev.matches(stats(), stats(), 0.1251)["degrees"]


def test_the_documented_thresholds():
    assert (C.GIANT_TOL, C.TRANS_RATIO, C.HOPS_RATIO, C.TV_TOL, C.HOPS_MIN_GIANT) == (0.05, (0.5, 2.0), (0.75, 4 / 3), 0.1, 0.5)


def test_verdict_kinds_follow_the_number_of_matches():
    ref = stats(giant=1.0, trans=0.25, hops=4.0)
    four = make(ref, stats(giant=1.0, trans=0.3, hops=4.0), tv=0.05)
    assert four.passed == 4 and ev.verdict_model(four)[0] == "success"
    three = make(ref, stats(giant=1.0, trans=0.3, hops=4.0), tv=0.5)
    assert three.passed == 3 and ev.verdict_model(three)[0] == "info"
    two = make(ref, stats(giant=1.0, trans=0.9, hops=4.0), tv=0.5)
    assert two.passed == 2 and ev.verdict_model(two)[0] == "info"
    zero = make(ref, stats(giant=0.1, trans=0.9, hops=20.0), tv=0.5)
    assert zero.passed == 0 and ev.verdict_model(zero)[0] == "warning"


def test_verdict_one_match_is_a_warning():
    ref = stats(giant=1.0, trans=0.25, hops=4.0)
    a = make(ref, stats(giant=1.0, trans=0.9, hops=20.0), tv=0.5)
    assert a.passed == 1 and ev.verdict_model(a)[0] == "warning"


def test_verdict_text_numbers():
    ref = stats(m=7621, giant=1.0, trans=0.033, hops=30.9, cross=125, kmax=8)
    st = stats(m=7626, giant=0.015, trans=0.733, hops=7.2, cross=5321, kmax=18)
    kind, text = ev.verdict_model(make(ref, st, tv=0.52))
    assert kind == "warning" and "trifft 0 von 4 Kennzahlen" in text and "Riesenkomponente 2 % (Karte 100 %)" in text and "Transitivität 0,733 (Karte 0,033)" in text
    assert "mittlere Weglänge 7,2 (Karte 30,9)" in text and "Gradverteilung im Abstand 0,52" in text and "Kreuzungen: 5321 (Karte 125)" in text and "Tabelle" in text
    kind, text = ev.verdict_model(make(ref, stats(cross=-1), tv=0.4))
    assert "Kreuzungen: ohne Raumbezug nicht gezählt (Karte 125)" in text
    kind, text = ev.verdict_model(make(stats(cross=0), stats(cross=0, trans=0.25), tv=0.0))
    assert kind == "success" and "Tabelle" not in text


def test_comparison_rows_by_hand():
    ref = stats(m=7621, giant=1.0, trans=0.033, hops=30.9, cross=125, kmax=8, med=13.0, p95=89.0)
    st = stats(m=9712, giant=1.0, trans=0.274, hops=34.2, cross=0, kmax=8, med=14.0, p95=68.0)
    rows = ev.comparison_rows(make(ref, st, tv=0.317))
    by = {r["Kennzahl"]: r for r in rows}
    assert by["Kanten"] == {"Kennzahl": "Kanten", "Karte": "7621", "Modell": "9712", "Trifft": "–"}
    assert by["Riesenkomponente (Anteil der Knoten)"]["Karte"] == "100,0 %" and by["Riesenkomponente (Anteil der Knoten)"]["Trifft"] == "ja"
    assert by["Transitivität"]["Karte"] == "0,033" and by["Transitivität"]["Modell"] == "0,274" and by["Transitivität"]["Trifft"] == "nein"
    assert by["mittlere Weglänge (Kanten, in der größten Komponente)"]["Trifft"] == "ja"
    assert by["Abstand der Gradverteilung (Totalvariation)"]["Modell"] == "0,32" and by["Abstand der Gradverteilung (Totalvariation)"]["Trifft"] == "nein"
    assert by["Kantenlänge Median / 95. Perzentil (m)"]["Karte"] == "13 / 89" and by["Kantenlänge Median / 95. Perzentil (m)"]["Modell"] == "14 / 68"
    assert ev.comparison_rows(make(ref, stats(cross=-1)))[6]["Modell"] == "nicht gezählt"


def series_rows():
    def row(n, requested, model, name, passed, giant=1.0, trans=0.25, hops=10.0):
        return dict(n=n, requested=requested, model=model, name=name, m=100, giant=giant, transitivity=trans, hops=hops, crossings=0 if model != "gnm" else -1, kmax=6, tv=0.2, len_median=10.0, len_p95=50.0, checks={}, passed=passed)

    rows = []
    for n, req in ((500, 500), (5072, None)):
        rows.append(row(n, req, "map", "Straßenkarte", 4))
        rows.append(row(n, req, "gabriel", "Gabriel-Graph", 2))
        rows.append(row(n, req, "gnm", "Erdős–Rényi G(n, m), ohne Raumbezug", 0))
    return rows


def test_series_verdict_and_table_by_hand():
    rows = series_rows()
    kind, text = ev.verdict_series(rows)
    assert kind == "info" and "ganzen Karte (5072 Knoten)" in text and "(Gabriel-Graph) 2 von 4 Kennzahlen" in text and "kein Modell trifft in irgendeiner der Größen alle vier" in text
    rows[-1]["passed"] = 4                       # ein Modell trifft bei der ganzen Karte alle vier
    assert "mindestens ein Modell trifft in einer Größe alle vier" in ev.verdict_series(rows)[1]
    table = ev.series_table(series_rows(), None)
    assert [t["Netz"] for t in table] == ["Straßenkarte", "Gabriel-Graph", "Erdős–Rényi G(n, m), ohne Raumbezug"]
    assert table[0]["Trifft von 4"] == "–" and table[0]["Abstand Gradverteilung"] == "–" and table[1]["Trifft von 4"] == "2" and table[1]["Riesenkomponente"] == "100,0 %"
    assert table[2]["Kreuzungen"] == "nicht gezählt" and table[1]["Weglänge"] == "10,0" and table[1]["Abstand Gradverteilung"] == "0,20"
    assert len(ev.series_table(series_rows(), 500)) == 3


def test_num_uses_a_decimal_comma():
    assert ev.num(3.14159, 2) == "3,14" and ev.num(10, 0) == "10"
