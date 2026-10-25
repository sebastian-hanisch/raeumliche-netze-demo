"""Kennzahlen, Messreihe und Verdikte (reine Funktionen: die App ruft st.<Art>(Text), die Tests prüfen jede Zahl im Text)."""

from dataclasses import dataclass
from statistics import median

import numpy as np

import rnz_algorithm as A
import rnz_constants as C
from rnz_scenario import build_model, make_map

RANDOM_MODELS = ("delaunay_random", "waxman", "gnm")        # Modelle mit Zufall; die übrigen sind bei gegebenen Standorten eindeutig
MEASURES = ("giant", "transitivity", "hops", "degrees")
MEASURE_NAMES = {"giant": "Riesenkomponente", "transitivity": "Transitivität", "hops": "mittlere Weglänge", "degrees": "Gradverteilung"}


def num(x: float, digits: int = 1) -> str:
    """Zahl mit Dezimalkomma (für Texte)."""
    return f"{x:.{digits}f}".replace(".", ",")


@dataclass(frozen=True)
class Settings:
    model: str = C.DEFAULT_MODEL
    n: int = C.DEFAULT_N
    k: int = C.DEFAULT_K
    scale: int = C.DEFAULT_SCALE
    seed: int = C.DEFAULT_SEED


@dataclass(eq=False)
class Analysis:
    settings: Settings
    pos: np.ndarray
    ref_edges: list
    model_edges: list
    ref: A.NetStats
    stats: A.NetStats
    tv: float
    checks: dict

    @property
    def passed(self) -> int:
        return sum(self.checks.values())


def matches(ref: A.NetStats, st: A.NetStats, tv: float) -> dict:
    """Trifft das Modell die Kennzahl der Karte? Riesenkomponente (Abstand), Transitivität und Weglänge (Verhältnis), Gradverteilung (Totalvariationsabstand).
    Die Weglänge zählt nur, wenn die größte Komponente mindestens die Hälfte der Knoten enthält."""
    tr = st.transitivity / ref.transitivity if ref.transitivity > 0 else float("inf")
    hr = st.hops / ref.hops
    return {
        "giant": abs(st.giant - ref.giant) <= C.GIANT_TOL,
        "transitivity": C.TRANS_RATIO[0] <= tr <= C.TRANS_RATIO[1],
        "hops": st.giant >= C.HOPS_MIN_GIANT and C.HOPS_RATIO[0] <= hr <= C.HOPS_RATIO[1],
        "degrees": tv <= C.TV_TOL,
    }


def analyse(s: Settings) -> Analysis:
    pos, edges = make_map(s.n)
    n = len(pos)
    rng = np.random.default_rng(s.seed)
    src = A.pick_sources(n, C.HOP_SOURCES, rng)
    ref = A.net_stats(n, edges, pos, src)
    model_edges = build_model(s.model, pos, len(edges), s.k, s.scale, rng)
    st = A.net_stats(n, model_edges, pos, src, count_crossings=s.model != "gnm")
    tv = A.total_variation(st.shares, ref.shares)
    return Analysis(s, pos, edges, model_edges, ref, st, tv, matches(ref, st, tv))


# --- Messreihe -------------------------------------------------------------------------------------------------------------------------------


def series_models(ns=C.SERIES_NS, runs: int = C.SERIES_RUNS, models=tuple(C.MODELS), k: int = C.DEFAULT_K, scale: int = C.DEFAULT_SCALE) -> list[dict]:
    """Je Kartenausschnitt und Modell die Kennzahlen (Median über `runs` Läufe bei den zufälligen Modellen, einmal bei den übrigen) und der Abstand zur Karte."""
    rows = []
    for n_req in ns:
        pos, edges = make_map(n_req)
        n = len(pos)
        src = A.pick_sources(n, C.HOP_SOURCES, np.random.default_rng(C.SERIES_SEED))
        ref = A.net_stats(n, edges, pos, src)
        rows.append(dict(n=n, requested=n_req, model="map", name="Straßenkarte", m=ref.m, giant=ref.giant, transitivity=ref.transitivity, hops=ref.hops, crossings=ref.crossings,
                         kmax=ref.kmax, tv=0.0, len_median=ref.len_median, len_p95=ref.len_p95, checks={}, passed=4))
        for model in models:
            count = runs if model in RANDOM_MODELS else 1
            stats = [A.net_stats(n, build_model(model, pos, len(edges), k, scale, np.random.default_rng(C.SERIES_SEED + j)), pos, src, count_crossings=model != "gnm") for j in range(count)]
            tvs = [A.total_variation(st.shares, ref.shares) for st in stats]
            med = lambda f: median(f(st) for st in stats)      # noqa: E731
            m_stats = A.NetStats(int(med(lambda x: x.m)), med(lambda x: x.giant), med(lambda x: x.transitivity), med(lambda x: x.hops), int(med(lambda x: x.crossings)), int(med(lambda x: x.kmax)),
                                 stats[0].shares, med(lambda x: x.len_median), med(lambda x: x.len_p95))
            tv = median(tvs)
            checks = matches(ref, m_stats, tv)
            rows.append(dict(n=n, requested=n_req, model=model, name=C.MODELS[model], m=m_stats.m, giant=m_stats.giant, transitivity=m_stats.transitivity, hops=m_stats.hops, crossings=m_stats.crossings,
                             kmax=m_stats.kmax, tv=tv, len_median=m_stats.len_median, len_p95=m_stats.len_p95, checks=checks, passed=sum(checks.values())))
    return rows


# --- Verdikte --------------------------------------------------------------------------------------------------------------------------------


def verdict_model(a: Analysis) -> tuple[str, str]:
    """Kernfrage der Hauptansicht: wie viele der vier Kennzahlen der Karte trifft das gewählte Modell?"""
    r, s = a.ref, a.stats
    parts = [
        f"Riesenkomponente {num(s.giant * 100, 0)} % (Karte {num(r.giant * 100, 0)} %)",
        f"Transitivität {num(s.transitivity, 3)} (Karte {num(r.transitivity, 3)})",
        f"mittlere Weglänge {num(s.hops, 1)} (Karte {num(r.hops, 1)})",
        f"Gradverteilung im Abstand {num(a.tv, 2)}",
    ]
    cross = "ohne Raumbezug nicht gezählt" if s.crossings < 0 else str(s.crossings)
    head = f"Das Modell trifft {a.passed} von 4 Kennzahlen der Karte: " + "; ".join(parts) + f". Kreuzungen: {cross} (Karte {r.crossings})."
    if a.passed == 4:
        return "success", head
    if a.passed >= 2:
        return "info", head + " Welche Kennzahlen fehlen, zeigt die Tabelle."
    return "warning", head + " Welche Kennzahlen fehlen, zeigt die Tabelle."


def verdict_series(rows: list[dict]) -> tuple[str, str]:
    """Messreihe: bei der ganzen Karte, wie viele Kennzahlen das beste Modell trifft und ob eines alle vier trifft."""
    full = [r for r in rows if r["requested"] is C.FULL_MAP and r["model"] != "map"]
    best = max(full, key=lambda r: r["passed"])
    sizes = sorted({r["n"] for r in rows})
    none_all = all(r["passed"] < 4 for r in rows if r["model"] != "map")
    text = (f"Bei der ganzen Karte ({max(sizes)} Knoten) trifft das beste Modell ({best['name']}) {best['passed']} von 4 Kennzahlen; "
            + ("kein Modell trifft in irgendeiner der Größen alle vier." if none_all else "mindestens ein Modell trifft in einer Größe alle vier."))
    return "info", text


def comparison_rows(a: Analysis) -> list[dict]:
    """Tabelle Karte gegen Modell: je Kennzahl der Wert der Karte, der Wert des Modells und ob das Modell sie trifft."""
    r, s = a.ref, a.stats
    yes = lambda k: "ja" if a.checks[k] else "nein"      # noqa: E731
    return [
        {"Kennzahl": "Kanten", "Karte": str(r.m), "Modell": str(s.m), "Trifft": "–"},
        {"Kennzahl": "Riesenkomponente (Anteil der Knoten)", "Karte": f"{num(r.giant * 100, 1)} %", "Modell": f"{num(s.giant * 100, 1)} %", "Trifft": yes("giant")},
        {"Kennzahl": "Transitivität", "Karte": num(r.transitivity, 3), "Modell": num(s.transitivity, 3), "Trifft": yes("transitivity")},
        {"Kennzahl": "mittlere Weglänge (Kanten, in der größten Komponente)", "Karte": num(r.hops, 1), "Modell": num(s.hops, 1), "Trifft": yes("hops")},
        {"Kennzahl": "Abstand der Gradverteilung (Totalvariation)", "Karte": "0", "Modell": num(a.tv, 2), "Trifft": yes("degrees")},
        {"Kennzahl": "größter Grad", "Karte": str(r.kmax), "Modell": str(s.kmax), "Trifft": "–"},
        {"Kennzahl": "Kreuzungen der Strecken", "Karte": str(r.crossings), "Modell": "nicht gezählt" if s.crossings < 0 else str(s.crossings), "Trifft": "–"},
        {"Kennzahl": "Kantenlänge Median / 95. Perzentil (m)", "Karte": f"{num(r.len_median, 0)} / {num(r.len_p95, 0)}", "Modell": f"{num(s.len_median, 0)} / {num(s.len_p95, 0)}", "Trifft": "–"},
    ]


def series_table(rows: list[dict], requested) -> list[dict]:
    """Tabelle der Messreihe für einen Ausschnitt: je Netz die Kennzahlen und wie viele der vier das Modell trifft."""
    out = []
    for r in (x for x in rows if x["requested"] == requested):
        out.append({
            "Netz": r["name"], "Kanten": str(r["m"]), "Riesenkomponente": f"{num(r['giant'] * 100, 1)} %", "Transitivität": num(r["transitivity"], 3), "Weglänge": num(r["hops"], 1),
            "Kreuzungen": "nicht gezählt" if r["crossings"] < 0 else str(r["crossings"]), "Abstand Gradverteilung": "–" if r["model"] == "map" else num(r["tv"], 2),
            "Trifft von 4": "–" if r["model"] == "map" else str(r["passed"]),
        })
    return out
