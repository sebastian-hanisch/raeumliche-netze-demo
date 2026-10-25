"""Konstanten: Regler-Grenzen, Schwellen des Vergleichs, feste Pläne der Messreihe, Farben und Presets."""

MODELS = {
    "geometric": "Geometrischer Zufallsgraph",
    "knn": "k nächste Nachbarn",
    "delaunay_random": "Delaunay, zufällig ausgedünnt",
    "delaunay_short": "Delaunay, kürzeste Kanten zuerst",
    "gabriel": "Gabriel-Graph",
    "waxman": "Waxman-Auswahl aus Delaunay",
    "gnm": "Erdős–Rényi G(n, m), ohne Raumbezug",
}
DEFAULT_MODEL = "geometric"

N_MIN, N_MAX, DEFAULT_N = 500, 5000, 2000          # Knoten des Kartenausschnitts (die Karte hat 5 072)
K_MIN, K_MAX, DEFAULT_K = 1, 4, 2                  # nächste Nachbarn
SCALE_MIN, SCALE_MAX, DEFAULT_SCALE = 10, 200, 30  # Längenskala der Waxman-Auswahl in Metern
SEED_MAX, DEFAULT_SEED = 999999, 35

FULL_MAP = None                                    # Kennzeichen „ganze Karte“ in den Messreihen
HOP_SOURCES = 40                                   # Quellen der mittleren Weglänge

# Schwellen: wann „trifft“ ein Modell eine Kennzahl der Karte
GIANT_TOL = 0.05              # Riesenkomponente: Abstand höchstens 0,05 (absolut)
TRANS_RATIO = (0.5, 2.0)      # Transitivität: Verhältnis Modell / Karte zwischen 1/2 und 2
HOPS_RATIO = (0.75, 4 / 3)    # mittlere Weglänge: Verhältnis zwischen 3/4 und 4/3
HOPS_MIN_GIANT = 0.5          # die Weglänge zählt nur, wenn die größte Komponente mindestens so viele Knoten enthält (sonst ist es die Weglänge eines kleinen Rests)
TV_TOL = 0.10                 # Gradverteilung: Totalvariationsabstand höchstens 0,1

# feste Pläne der Messreihe (nie ein einzelner Seed)
SERIES_SEED = 100000          # Lauf j der zufälligen Modelle hat den Seed SERIES_SEED + j
SERIES_RUNS = 3
SERIES_NS = (500, 1500, 3000, FULL_MAP)

COLOR_MEASURED = "#2F6B65"
COLOR_THEORY = "#4c78a8"
COLOR_THIRD = "#e8a33d"
COLOR_ALERT = "#d62728"
COLOR_LAYER = "rgba(120,120,120,0.45)"


def _p(model, n=1500, k=DEFAULT_K, scale=DEFAULT_SCALE):
    return {"model": model, "n": n, "k": k, "scale": scale, "seed": DEFAULT_SEED}


PRESETS = {
    "Geometrischer Zufallsgraph": _p("geometric", n=DEFAULT_N),
    "Zwei nächste Nachbarn": _p("knn", k=2),
    "Gabriel-Graph": _p("gabriel"),
    "Delaunay, kürzeste Kanten zuerst": _p("delaunay_short"),
    "Erdős–Rényi ohne Raumbezug": _p("gnm"),
}
PRESET_HELP = {
    "Geometrischer Zufallsgraph": "Alle Punktpaare unter einem Radius sind verbunden, der Radius so gewählt, dass die Kantenzahl der Karte erreicht wird: das Netz hat viele Dreiecke und zerfällt in kleine Teile.",
    "Zwei nächste Nachbarn": "Jeder Punkt verbindet sich mit seinen zwei nächsten Nachbarn: ähnlich viele Kanten wie die Karte, aber viel mehr Dreiecke und keine zusammenhängende Karte.",
    "Gabriel-Graph": "Teilgraph der Delaunay-Triangulierung mit der Eigenschaft, dass kein Punkt im Kreis über der Kante liegt: zusammenhängend und kreuzungsfrei, aber mit deutlich mehr Dreiecken als die Karte.",
    "Delaunay, kürzeste Kanten zuerst": "Aus der Triangulierung bleiben die kürzesten Kanten, bis die Kantenzahl der Karte erreicht ist: kreuzungsfrei, aber die Karte zerfällt in kleine Teile.",
    "Erdős–Rényi ohne Raumbezug": "Dieselbe Zahl von Kanten, aber gleichverteilt unter allen Paaren: die Gradverteilung ähnelt der Karte, doch die Kanten sind kilometerlang und es gibt fast keine Dreiecke.",
}
