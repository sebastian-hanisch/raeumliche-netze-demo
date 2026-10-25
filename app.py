"""Netze im Raum: sieht ein Straßennetz aus wie ein Zufallsmodell? - interaktive Konzept-Demo
Sebastian Hanisch - Operations Research und Machine Learning

Stück 6 der Linie „Netzmodelle“: die Straßenkarte von Toronto (OpenStreetMap, ODbL) gegen räumliche Zufallsgraphen auf denselben Standorten (geometrischer Zufallsgraph, k nächste Nachbarn,
Delaunay, Gabriel, Waxman) und gegen G(n, m) ohne Raumbezug. Der Kern liegt in rnz_algorithm.py.

Lauffähig mit: streamlit run app.py
"""

import time

import streamlit as st

import rnz_constants as C
import rnz_evaluation as ev
from rnz_evaluation import Settings, analyse, comparison_rows, num, verdict_model, verdict_series
from rnz_layout import dunkles_design, kopf_css
from rnz_presets import (
    bounds,
    init_session_state_defaults,
    load_permalink_settings,
    randomize_seed,
    sidebar_presets,
    sync_query_params,
)
from rnz_visualization import build_degrees, build_map, build_series

st.set_page_config(page_title="Netze im Raum – Sebastian Hanisch", layout="wide")


@st.cache_data(show_spinner=False)
def _analysis(settings):
    return analyse(settings)


@st.cache_data(show_spinner=False)
def _series():
    return ev.series_models()


# Fester Kopf: nur der Titel bleibt beim Scrollen oben (Stylesheet in rnz_layout.py).
st.html(kopf_css(dunkles_design()))
with st.container(key="kopf"):
    st.title("🗺️ Netze im Raum: sieht ein Straßennetz aus wie ein Zufallsmodell?")
st.markdown(
    """
Straßennetze liegen in der Ebene: wenige Kanten je Kreuzung, keine Kreuzungen ohne Knoten, kurze Kanten. Liegt es nahe, sie als **räumlichen Zufallsgraphen** zu beschreiben? Die Demo nimmt die Straßenkarte rund um den Campus der Universität Toronto
(5 072 Kreuzungen, 7 621 Straßenabschnitte) und baut auf **denselben Standorten** Netze nach sechs räumlichen Modellen mit etwa derselben Kantenzahl: einen geometrischen Zufallsgraphen, die k nächsten Nachbarn, die Delaunay-Triangulierung
(zufällig oder nach Länge gekürzt), den Gabriel-Graph und eine Waxman-Auswahl; dazu ein Erdős–Rényi-Netz ganz ohne Raumbezug. Verglichen werden **vier Kennzahlen**: Riesenkomponente, Transitivität, mittlere Weglänge und Gradverteilung.
Sechstes Stück der Linie „Netzmodelle“; **das Ergebnis ist ein Negativbefund**: keines der Modelle trifft alle vier Kennzahlen zugleich. Das sagt etwas über diese Modelle und diese Karte, nicht über Straßennetze im Allgemeinen.

Weiter unten: 🔬 **Messreihe** (vier Ausschnitte bis zur ganzen Karte), 🚧 **Wo die Annahmen enden** und der Expander **📐 Mathematische Formulierung**.
"""
)

with st.expander("So wird verglichen", expanded=True):
    st.markdown(
        """
1. **Karte:** die n Kreuzungen um die Mitte des Ausschnitts (Breitensuche) mit ihren Straßenabschnitten; Standorte in Metern.
2. **Modell:** ein Netz auf denselben Standorten. Alle Modelle außer „k nächste Nachbarn“ und „Gabriel“ bekommen **dieselbe Kantenzahl** wie die Karte; Gabriel und k nächste Nachbarn haben ihre eigene.
3. **Kennzahlen:** Anteil der Knoten in der größten Komponente, Transitivität (Anteil geschlossener Dreier), mittlere Weglänge in Kanten (von 40 Quellen, in der größten Komponente) und der Totalvariationsabstand der Gradverteilung.
4. **Trifft das Modell die Kennzahl?** Riesenkomponente: Abstand höchstens 0,05; Transitivität: Verhältnis zwischen ½ und 2; Weglänge: Verhältnis zwischen ¾ und 4⁄3, aber nur, wenn die größte Komponente mindestens die Hälfte der Knoten enthält; Gradverteilung: Abstand höchstens 0,1.
        """
    )

st.caption("🔗 Die Adresszeile oben spiegelt Ihre aktuelle Konfiguration wider – einfach kopieren, um ein Szenario zu teilen.")

load_permalink_settings()
init_session_state_defaults()

with st.sidebar:
    sidebar_presets()  # Schnellstart oben, das aktive Preset ist hervorgehoben
    st.header("⚙️ Einstellungen")
    model = st.selectbox("Modell", list(C.MODELS), format_func=C.MODELS.get, key="model_select", help="Welches Modell auf den Standorten der Karte gebaut wird.")
    n = st.slider("Knoten des Ausschnitts", *bounds("n_slider"), key="n_slider", step=100, help="Wie viele Kreuzungen um die Mitte der Karte (die ganze Karte hat 5 072; sie steht in der Messreihe).")
    k = st.slider("Nächste Nachbarn k", *bounds("k_slider"), key="k_slider", help="Nur für „k nächste Nachbarn“: jeder Punkt wird mit seinen k nächsten Nachbarn verbunden.")
    scale = st.slider("Längenskala der Waxman-Auswahl (m)", *bounds("scale_slider"), key="scale_slider", step=10, help="Nur für die Waxman-Auswahl: je kleiner, desto stärker werden lange Kanten aussortiert.")
    seed = st.number_input("Zufalls-Seed", *bounds("seed_input"), key="seed_input", step=1)
    st.button("🎲 Neues Modellnetz ziehen", width="stretch", on_click=randomize_seed)

sync_query_params({"model_select": model, "n_slider": int(n), "k_slider": int(k), "scale_slider": int(scale), "seed_input": int(seed)})

settings = Settings(model=model, n=int(n), k=int(k), scale=int(scale), seed=int(seed))
with st.spinner("Baue das Modell und messe …"):
    a = _analysis(settings)

# --- Ergebnis ------------------------------------------------------------------------------------------------------------------------------

st.markdown("## 🎯 Ergebnis")
m1, m2, m3, m4 = st.columns(4)
m1.metric("Kanten (Modell)", f"{a.stats.m}", delta=f"Karte {a.ref.m}", delta_color="off")
m2.metric("Riesenkomponente", f"{num(a.stats.giant * 100, 0)} %", delta=f"Karte {num(a.ref.giant * 100, 0)} %", delta_color="off")
m3.metric("Transitivität", num(a.stats.transitivity, 3), delta=f"Karte {num(a.ref.transitivity, 3)}", delta_color="off")
m4.metric("Weglänge (Kanten)", num(a.stats.hops, 1), delta=f"Karte {num(a.ref.hops, 1)}", delta_color="off")
c1, c2 = st.columns(2)
c1.markdown("**Straßenkarte**")
c1.plotly_chart(build_map(a.pos, a.ref_edges, C.COLOR_MEASURED, "Straßenkarte"), width="stretch", key="map_ref")
c2.markdown(f"**Modell: {C.MODELS[settings.model]}**")
c2.plotly_chart(build_map(a.pos, a.model_edges, C.COLOR_THIRD, "Modell"), width="stretch", key="map_model")
st.caption(
    "Beide Karten zeigen dieselben Standorte (Meter); die Kanten sind Strecken, kein Straßenverlauf. "
    + ("Das Erdős–Rényi-Netz hat Kanten über die ganze Karte; gezeigt wird eine gleichmäßige Auswahl von höchstens 6 000 Kanten. " if settings.model == "gnm" else "")
    + "Kartendaten: © OpenStreetMap-Mitwirkende, Open Database License (ODbL)."
)
d1, d2 = st.columns([1, 1])
d1.plotly_chart(build_degrees(a.ref.shares, a.stats.shares), width="stretch", key="degrees")
d2.table(comparison_rows(a))

# --- Kernfrage -----------------------------------------------------------------------------------------------------------------------------

st.markdown("## ❓ Kernfrage: Trifft dieses Modell die Straßenkarte?")
kind, text = verdict_model(a)
getattr(st, kind)(text)

st.markdown("---")

# --- Messreihe -----------------------------------------------------------------------------------------------------------------------------

st.subheader("🔬 Messreihe: alle Modelle bei vier Ausschnittsgrößen")
st.caption(
    f"Festes Programm, unabhängig von den Reglern: Ausschnitte mit {', '.join(str(x) for x in C.SERIES_NS if x)} Knoten und die ganze Karte, alle sieben Modelle; bei den zufälligen Modellen der Median über {C.SERIES_RUNS} Läufe; "
    "k = 2 und Waxman-Skala 30 m. Das Rechnen dauert einige Sekunden."
)
if st.button("Messreihe rechnen", key="series_button"):
    t0 = time.perf_counter()
    with st.spinner("Rechne die Messreihe (einige Sekunden) …"):
        _series()
    st.session_state["series_done"] = time.perf_counter() - t0
if "series_done" in st.session_state:
    rows = _series()
    sizes = {r["requested"]: r["n"] for r in rows}
    pick = st.selectbox("Ausschnitt", list(sizes), format_func=lambda x: f"{sizes[x]} Knoten" + (" (ganze Karte)" if x is C.FULL_MAP else ""), index=len(sizes) - 1, key="series_size")
    st.plotly_chart(build_series(rows, pick), width="stretch", key="series_chart")
    st.caption("Der Stern ist die Straßenkarte: kein Modell liegt in seiner Nähe. Offene Kreise: die größte Komponente enthält weniger als die Hälfte der Knoten, die Weglänge ist die eines kleinen Rests.")
    st.table(ev.series_table(rows, pick))
    kind, text = verdict_series(rows)
    getattr(st, kind)(text)
    st.caption(f"Rechenzeit dieser Sitzung: {num(st.session_state['series_done'], 1)} s (beim ersten Mal; danach liegt das Ergebnis im Zwischenspeicher).")

st.markdown("---")

# --- Grenzen -------------------------------------------------------------------------------------------------------------------------------

st.subheader("🚧 Wo die Annahmen enden")
st.markdown(
    """
| Annahme | Was passiert, wenn sie verletzt ist | Wer setzt an |
|---|---|---|
| **Eine Karte steht für Straßennetze** | Der Befund gilt für diesen Ausschnitt von Toronto (Campus und Umgebung, 1 km Umkreis); andere Städte, Gitter- oder Altstadtnetze können anders aussehen. | diese Demo |
| **Die Standorte sind gegeben** | Die Modelle bekommen die echten Kreuzungen als Punkte; ein Modell, das auch die Standorte erzeugt, ist schwerer und nicht Teil der Demo. | Netze im Raum (weitere Stücke) |
| **Vier Kennzahlen reichen** | Es gibt mehr Maße (Kantenlänge, Kreise, Betweenness); die Schwellen für „trifft“ sind eine Festlegung der Demo (Abstand 0,05, Verhältnis 2, ¾ bis 4⁄3, Abstand 0,1). | diese Demo |
| **Das Modell sieht einen Straßenverlauf** | Die Kanten sind Strecken zwischen den Punkten; krumme Straßen und Brücken sind nicht abgebildet, und die Karte hat 125 Kreuzungen von Strecken, die keine Kreuzungen im Straßennetz sind. | Planarität und Zeichnen (Vorgänger) |
| **Ein Modell soll alles zugleich treffen** | Die Modelle sind einfach; ein Modell mit mehreren Stellschrauben (Länge, Grad, Dreiecke) könnte besser passen. Die Demo zeigt nur, dass diese einfachen es nicht tun. | Modellwahl (Folgestück) |
"""
)

st.markdown("---")

with st.expander("📐 Mathematische Formulierung"):
    st.markdown(
        r"""
**Lesehilfe.** Gegeben sind $n$ Punkte $x_1, \dots, x_n$ in der Ebene (die Kreuzungen) und die Kantenmenge $E$ der Karte mit $m = |E|$ Kanten.

| Symbol | Bedeutung |
|---|---|
| $d(u, v)$ | euklidischer Abstand der Punkte $u$ und $v$ |
| $m$ | Zahl der Kanten der Karte |
| $C$ | Transitivität |

**Geometrischer Zufallsgraph.** Kante genau dann, wenn $d(u, v) \le \rho$; der Radius $\rho$ ist der $m$-kleinste Paarabstand, das Netz hat also $m$ Kanten.

**k nächste Nachbarn.** Kante $\{u, v\}$, wenn $v$ unter den $k$ nächsten Punkten von $u$ liegt oder umgekehrt.

**Delaunay-Triangulierung.** Kante $\{u, v\}$, wenn es einen Kreis durch $u$ und $v$ ohne weiteren Punkt im Inneren gibt. **Gabriel-Graph:** Kante, wenn der Kreis mit Durchmesser $uv$ keinen weiteren Punkt im Inneren enthält; der Gabriel-Graph ist ein Teilgraph der Delaunay-Triangulierung.

**Waxman-Auswahl.** Jede Delaunay-Kante $e$ bleibt mit Wahrscheinlichkeit $p_e = \min\{1, c \, e^{-d_e / s}\}$, wobei $c$ so gewählt ist, dass $\sum_e p_e = m$.

**Transitivität.** $C = \dfrac{3 \cdot \#\text{Dreiecke}}{\#\text{Zweierwege}}$, mit $\#\text{Zweierwege} = \sum_v \binom{\deg v}{2}$.

**Gradverteilung.** Abstand $\tfrac12 \sum_k |p_k - q_k|$ der Anteile der Knoten mit Grad $k$ (Grade ab 9 zusammengefasst).

**Literatur** (aus dem Gedächtnis, nicht im Original geprüft): Penrose, *Random Geometric Graphs* (2003); Waxman, *Routing of multipoint connections*, IEEE Journal on Selected Areas in Communications 6 (1988); Gabriel und Sokal, *A new statistical approach to geographic variation analysis*, Systematic Zoology 18 (1969); Barthélemy, *Spatial networks*, Physics Reports 499 (2011).
        """
    )

st.markdown("---")
st.caption(
    "Diese Demo ist Teil des Portfolios von [Sebastian Hanisch](https://sebastianhanisch.net) – "
    "Operations Research und Machine Learning ([Über mich](https://sebastianhanisch.net/ueber-mich.html)). "
    "Mehr zu den Demos: [Interaktive Demos](https://sebastianhanisch.net/demos.html)."
)
