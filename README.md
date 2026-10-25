# Netze im Raum: sieht ein Straßennetz aus wie ein Zufallsmodell? – Streamlit-Demo

<!-- TODO nach dem Deploy: direkt hierher den Live-Link setzen (Leerzeile davor und danach):
**[→ Demo live ausprobieren](https://sebastianhanisch-raeumliche-netze-demo.streamlit.app/)** -->

Sechstes Stück der **Netzmodelle-Linie** (Graphentheorie) in der „Konzepte“-Reihe für die Website „Sebastian Hanisch – Operations Research und Machine Learning“. Die Linie erzeugt Netze aus Zufallsmodellen und prüft sie gegen Theorie und Daten; dieses Stück
nimmt ein **echtes Straßennetz** als Vergleich. Vorgänger sind [navigierbarkeit-demo](https://github.com/sebastian-hanisch/navigierbarkeit-demo) (Suche mit bekannter Geometrie) und [planarisierung-demo](https://github.com/sebastian-hanisch/planarisierung-demo),
das dieselbe Karte als Zeichnung mit Streckenkreuzungen untersucht; die Karte stammt aus [graph-zeichnen-demo](https://github.com/sebastian-hanisch/graph-zeichnen-demo).

Die Demo nimmt die Straßenkarte rund um den Campus der Universität Toronto (5 072 Kreuzungen, 7 621 Straßenabschnitte, OpenStreetMap) und baut auf **denselben Standorten** Netze nach sechs räumlichen Modellen: **geometrischer Zufallsgraph**, **k nächste Nachbarn**,
**Delaunay-Triangulierung** (zufällig oder nach Länge gekürzt), **Gabriel-Graph** und **Waxman-Auswahl**; dazu ein **Erdős–Rényi-Netz G(n, m)** ganz ohne Raumbezug. Verglichen werden vier Kennzahlen: Riesenkomponente, Transitivität, mittlere Weglänge und Gradverteilung.
Der Kern ist eigener Code auf `scipy.spatial`; `networkx` kommt nur als Gegenprobe in die Tests.

## Warum dieses Problem

Straßennetze liegen in der Ebene und sehen auf den ersten Blick „räumlich zufällig“ aus: wenige Kanten je Kreuzung, kurze Kanten. Ob ein einfaches räumliches Modell sie beschreibt, ist eine messbare Frage.
Die Frage der Demo: **Trifft eines dieser Modelle die Karte, und wenn nicht, woran liegt es?** Das Ergebnis ist ein **Negativbefund**: keines trifft alle vier Kennzahlen, bei keiner der vier Ausschnittsgrößen. Das sagt etwas über diese sieben Modelle und diese Karte, nicht über Straßennetze im Allgemeinen.

## Modell

Alle Modelle bekommen die Standorte der Karte (Meter, aus OpenStreetMap-Koordinaten). **Geometrischer Zufallsgraph:** Kante, wenn der Abstand höchstens ρ ist; ρ ist der m-kleinste Paarabstand, das Netz hat also die Kantenzahl m der Karte.
**k nächste Nachbarn:** Kante, wenn ein Punkt unter den k nächsten des anderen liegt (k = 2). **Delaunay:** die Kanten der Triangulierung (15 193 bei der ganzen Karte), zufällig auf m oder auf die m kürzesten gekürzt. **Gabriel:** die Delaunay-Kanten, über denen der Kreis mit Durchmesser uv keinen Punkt enthält (eigene Kantenzahl).
**Waxman:** jede Delaunay-Kante bleibt mit Wahrscheinlichkeit min(1, c·e^(−Länge/Skala)), c so, dass m Kanten erwartet werden. **G(n, m):** m gleichverteilt gezogene Paare, ohne Bezug zu den Standorten.
Kennzahlen: Anteil der Knoten in der größten Komponente, **Transitivität** (3 · Dreiecke / Zweierwege), **mittlere Weglänge** in Kanten (von 40 Quellen, in der größten Komponente), **Totalvariationsabstand** der Gradverteilung (Grade ab 9 zusammengefasst) und die Zahl der Streckenkreuzungen.
Ein Modell „trifft“ eine Kennzahl, wenn die Riesenkomponente höchstens 0,05 abweicht, die Transitivität zwischen dem halben und doppelten Wert der Karte liegt, die Weglänge zwischen ¾ und 4⁄3 (nur, wenn die größte Komponente mindestens die Hälfte der Knoten enthält) und der Abstand der Gradverteilung höchstens 0,1 ist.
Formeln und Lesehilfe stehen im 📐-Expander der App. Die Literaturangaben dort stammen aus dem Gedächtnis und sind nicht im Original geprüft.

## Methodik

- **Kern** (`rnz_algorithm.py`): Delaunay, Gabriel, geometrischer Zufallsgraph und Radius, nächste Nachbarn, Kürzen, Waxman, G(n, m), Kennzahlen (Komponenten, Transitivität über die dünne Adjazenzmatrix, Weglänge über `scipy.sparse.csgraph`, Streckenkreuzungen mit Sweep und Orientierungstest).
- **Szenario und Auswertung** (`rnz_scenario.py`, `rnz_evaluation.py`): der Kartenausschnitt (Breitensuche um die Mitte), die Modelle auf seinen Standorten, die Vergleichsregeln, eine Messreihe mit festem Programm, alle Verdikte als reine Funktionen.
- **Oberfläche** (`app.py`): fünf Schnellstart-Szenarien, Regler für Modell, Ausschnitt, k und Waxman-Skala, Karte und Modell nebeneinander, Gradverteilung und Vergleichstabelle, Kernfrage, Messreihe auf Klick, Annahmen und Formeln.
- **Daten** (`data/toronto_campus.json`): Auszug aus OpenStreetMap, © OpenStreetMap-Mitwirkende, **Open Database License (ODbL)**; Kopie aus `graph-zeichnen-demo`, Lizenz und Namensnennung in `data/LICENSE-ODbL.md`.

## Befunde (gemessen, keine Behauptungen)

Karte und deterministische Modelle sind bei gegebenen Standorten eindeutig; bei den zufälligen Modellen (Delaunay zufällig, Waxman, G(n, m)) ist jede Zahl der **Median über drei Läufe** (Seeds ab 100000), die 40 Quellen der Weglänge sind fest gezogen. Die Schnellstart-Szenarien sind Einzelläufe mit dem Zufalls-Seed 35.
**Die Zufallszahlen eines `numpy`-Generators sind nicht über Versionen stabil**: die Tests prüfen deshalb Bänder um die Zahlen der zufälligen Modelle.

| Frage | Befund | Test |
|---|---|---|
| Stimmt der Kern? | Delaunay gleich der Aufzählung aller Dreiecke mit leerem Umkreis (drei Punktmengen mit je 40 Punkten); Gabriel gleich der Aufzählung aller Paare mit leerem Kreis (zwei Mengen mit je 60 Punkten) und Teilgraph der Delaunay-Triangulierung; Radius und geometrischer Zufallsgraph gleich der Aufzählung aller Paare; nächste Nachbarn gleich der Sortierung der Abstände; Transitivität, Weglänge und Komponenten gleich `networkx`; die Streckenkreuzungen gleich einer Zählung mit exakten Brüchen; G(n, m) gleichverteilt über alle 120 Kantenmengen (Chi-Quadrat); die Karte hat 5 072 Knoten, 7 621 Kanten und 125 Streckenkreuzungen. | `tests/test_oracle_raeumliche_netze.py` |
| Wie sieht die Karte aus? | 5 072 Knoten, 7 621 Kanten (mittlerer Grad 3,0), eine Komponente, Transitivität **0,033**, mittlere Weglänge **30,9** Kanten, größter Grad 8, **125** Streckenkreuzungen, Kantenlänge im Median **13 m**, 95. Perzentil **89 m**. | `test_claims.py::test_the_map` |
| Was zeigt der geometrische Zufallsgraph? | Bei gleicher Kantenzahl liegen nur **1,5 %** der Knoten in der größten Komponente, die Transitivität ist **0,73** (das 22-Fache der Karte), die Zahl der Streckenkreuzungen **5 336** gegen 125, 95 % der Kanten sind höchstens 14 m lang (Karte: 89 m). Die Gradverteilung liegt im Abstand **0,52**. | `test_claims.py::test_the_geometric_graph_falls_apart_and_has_many_triangles` |
| Was zeigen die nächsten Nachbarn und die gekürzte Triangulierung? | Zwei nächste Nachbarn: **6 482** Kanten, Riesenkomponente **1,6 %**, Transitivität **0,40**, Abstand der Gradverteilung **0,53**. Delaunay mit den kürzesten Kanten: Riesenkomponente **2,9 %**, Transitivität **0,47**. Beide treffen keine der vier Kennzahlen. | `test_claims.py::test_the_nearest_neighbour_graph_falls_apart`, `::test_the_delaunay_models` |
| Welche Modelle kommen am nächsten? | Delaunay zufällig auf 7 621 Kanten gekürzt: Riesenkomponente **97 %** und mittlere Weglänge **29,4** passen, aber Transitivität **0,18** und Gradverteilung (Abstand **0,36**) nicht. Gabriel-Graph (**9 712** Kanten, 27 % mehr als die Karte): Riesenkomponente 100 %, Weglänge **34,2** passen, Transitivität **0,27** (das 8-Fache) und Gradverteilung (**0,32**) nicht. Beide treffen **2 von 4**. | `test_claims.py::test_the_gabriel_graph_matches_the_path_length_but_not_the_triangles`, `::test_the_delaunay_models` |
| Und Waxman und das Netz ohne Raumbezug? | Waxman (Skala 30 m): Riesenkomponente **69 %**, Weglänge **61,7**, Transitivität **0,29**. G(n, m): Riesenkomponente **94 %**, Transitivität **0,001**, Weglänge **7,8**, Kanten im Median **961 m** lang. Beide treffen 0 von 4. | `test_claims.py::test_waxman_and_the_random_graph` |
| Hält der Befund bei kleineren Ausschnitten? | Ja: bei 500, 1 500, 3 000 Knoten und der ganzen Karte trifft kein Modell mehr als **2 von 4** Kennzahlen, und **keines trifft die Gradverteilung oder die Transitivität**; der kleinste Abstand der Gradverteilung über alle Modelle und Größen ist **0,26**. | `test_claims.py::test_no_model_hits_more_than_two_of_four_in_any_size`, `::test_every_model_misses_the_degree_distribution_and_the_triangles` |
| Was zeigen die Schnellstart-Szenarien (Zufalls-Seed 35)? | Geometrischer Zufallsgraph (2 000 Knoten): Riesenkomponente 3,7 % gegen 100 %, Transitivität 0,69 gegen 0,045, 1 245 Streckenkreuzungen gegen 58. Zwei nächste Nachbarn (1 500 Knoten): 1 932 Kanten gegen 2 093, Riesenkomponente 5,5 %. Gabriel-Graph: 2 756 Kanten gegen 2 093, 2 von 4. Delaunay nach Länge: Riesenkomponente 7,3 %, keine Kreuzung. G(n, m): Transitivität 0,002, Riesenkomponente 93 %. | `test_claims.py::test_presets` |

## Ehrliche Grenzen

- **Eine Karte, ein Ausschnitt.** Der Befund gilt für den Campus der Universität Toronto und einen Kilometer Umgebung (OpenStreetMap, abgefragt am 21.09.2026); andere Städte, Gitternetze oder Altstädte können anders aussehen. Die Messreihe zeigt vier Ausschnitte derselben Karte, nicht vier Städte.
- **Die Modelle sind einfach, die Schwellen eigene Festlegungen.** Abstand 0,05, Verhältnis 2, Verhältnis ¾ bis 4⁄3, Abstand 0,1 und die Mindestgröße der Riesenkomponente sind Setzungen der Demo, keine Sätze. Ein Modell mit mehreren Stellschrauben (Länge, Grad, Dreiecke) könnte besser passen; die Demo zeigt nur, dass diese sieben es nicht tun.
- **Die Weglänge ist schwach.** Sie wird von nur 40 Quellen gemessen; die Karte kam in der Vormessung auf 33,7, hier auf 30,9 (andere Quellen). Bei zerfallenden Modellen ist sie die Weglänge eines kleinen Rests (kleiner als 3 % der Knoten) und zählt deshalb nicht als Treffer.
- **Die Karte hat Streckenkreuzungen** (125), weil die Kanten gerade Strecken statt Straßenverläufen sind. Ob Brücken, Tunnel oder Krümmung dahinterstecken, ist mit diesen Daten nicht entscheidbar ([planarisierung-demo](https://github.com/sebastian-hanisch/planarisierung-demo)).
- **Kantenzahl gleich gesetzt.** Alle Modelle außer nächsten Nachbarn (k = 2) und Gabriel haben die Kantenzahl der Karte; bei der ganzen Karte weicht sie bei Gabriel (+27 %) und k = 2 (−15 %) davon ab, und der Vergleich der Kennzahlen hängt davon ein Stück weit ab.
- **Zufallsmodelle nur dreimal.** Median über drei Läufe; Standardabweichungen sind nicht ausgewiesen. Delaunay-Zufall, Waxman und G(n, m) schwanken, die übrigen Modelle sind bei festen Standorten eindeutig.
- **Datenlizenz.** Die Karte steht unter der ODbL (Namensnennung und Share-Alike für die Daten); die Lizenz gilt nur für `data/`, nicht für den Code.
- **Literatur** (Penrose, Waxman, Gabriel und Sokal, Barthélemy) aus dem Gedächtnis, nicht im Original geprüft.

## Befunde und Korrekturen gegenüber dem Plan

- **Port:** 9162 statt 9160 (9160 hält `potenzgesetz-test-demo`, 9161 `navigierbarkeit-demo`).
- **Vormessung bestätigt:** geometrischer Zufallsgraph 5 336 Kreuzungen und Riesenkomponente 1,5 % (Vormessung 5 283 und 1,5 %); Gabriel trifft die Weglänge, aber nicht die Transitivität (0,274 gegen 0,033, wie in der Vormessung).
- **Abstand der Gradverteilung:** die Vormessung nannte „mindestens 0,32“; die Demo findet beim Gabriel-Graph **0,317** und über alle Größen und Modelle mindestens 0,26. Die Aussage ist dieselbe (kein Modell trifft die Gradverteilung), die Zahl wurde nicht übernommen.
- **Weglänge zählt nur mit großer Komponente:** in der Vormessung stand bei zerfallenden Modellen (zum Beispiel Delaunay nach Länge, Riesenkomponente 2,9 %) eine Weglänge neben der Karte, die nach dem Verhältnis „passte“; das ist die Weglänge eines kleinen Rests und zählt deshalb nicht (Schwelle: größte Komponente mindestens die Hälfte).
- **Ein Fehler, den der Orakel-Test fand:** bei Gleichstand am Radius ließ der Suchbaum das Paar genau auf dem Radius aus, der geometrische Zufallsgraph hatte dann eine Kante weniger als verlangt; seither gibt `geometric_edges` eine relative Zugabe von 10⁻¹² (`RADIUS_SLACK`).
- **Messreihe in Sekunden:** die Messreihe rechnet in einigen Sekunden, nicht in Minuten; die Befunde aller Größen stehen deshalb in einer Reihe.

## Verifikation

- **Kern:** jede Einheit an einer Mini-Instanz von Hand (flaches Dreieck und Drachenviereck für Delaunay und Gabriel, Weg auf einer Geraden für Radius, Nachbarn und Kürzen, Dreieck mit Anhang für die Transitivität, Quadrat mit Diagonalen für die Kreuzungen, Stern für die Gradverteilung), Randfälle (kleinster und größter Ausschnitt für jedes Modell, kollineare Punkte, leere Kantenlisten, Gleichstände).
- **Orakel:** Aufzählung leerer Umkreise und Kreise, alle Paare für Radius und Nachbarn, `networkx`, exakte Brüche für Kreuzungen, Chi-Quadrat für G(n, m), Erwartung der Waxman-Kantenzahl.
- **Verdikte:** reine Funktionen, jede Schwelle an dyadischen Werten ohne Rundungsfehler, jede Zahl im Text an handgesetzten Werten (`tests/test_verdicts.py`).
- **Presets, Layout, Oberfläche:** wie in allen Demos (Presets vollständig und im Regelbereich und auf dem Raster der Regler, keines auf einem Reglerrand, Schnellstart oben, aktives Preset hervorgehoben, kein Preset entsteht allein durch einen Modellwechsel, AppTest jeder Button und jeder Regler an den Rändern).
- **Mutationsprüfung** (`tools/mutate_demo.py`): 138 Stellen im Kern `rnz_algorithm.py`, **106 gefangen** (9 davon über wortgleiche Zeilen in Nachbar-Demos), 25 als äquivalent festgehalten, 7 Zeitüberschreitungen (das Werkzeug zählt sie nicht). Der erste Lauf ließ 31 Mutanten überleben und deckte zwei echte Lücken auf: die Vorzeichenprüfung der Streckenkreuzung bei Koordinaten unter 1 (jetzt `test_crossing_test_does_not_depend_on_the_scale_of_the_coordinates`, mit Zweierpotenzen als Maßstab, damit die Skalierung exakt bleibt) und die Auswahl des m-kleinsten Abstands (jetzt `test_radius_equals_the_sorted_pair_distance_for_every_edge_count`). Die äquivalenten Stellen sind: die letzte Schleife und der Start bei `ii` im Kreuzungszähler (ein Paar mit sich selbst teilt alle Endpunkte), die Ränder der Umrandungs-Filter (Berühren zweier Umrandungen an einer Kante kann keine echte Kreuzung sein), `count >= limit` gegen `>`, das überflüssige `w != u and w != v` beim Gabriel-Graph (die Endpunkte liegen auf dem Rand des Kreises), die Vorbelegung und Größe der Ziehpuffer von G(n, m) (nur die Verteilung zählt, nicht die Zahl der Ziehungen), der Startwert des Suchradius, vier `>`-gegen-`>=`-Mutanten in der Vorzeichenprüfung (der Fall null ist schon ausgeschlossen), `<` gegen `<=` bei der Waxman-Ziehung (Gleichstand hat Wahrscheinlichkeit null) und drei Randvergleiche in der Bisektion der Waxman-Wahrscheinlichkeiten (gleiche Ergebnisse). Die 7 Zeitüberschreitungen sind Mutanten, die in Endlosschleifen laufen (Kantenzahl-Obergrenze und Ziehschleife von G(n, m), Abbruchbedingung der Radiussuche) oder die Rechnung ohne Umrandungs-Filter stark verlängern; sie wurden von Hand in einer Kopie geprüft: die Mutanten der Schleifen hängen oder werden gefangen, der ohne Filter ergibt dasselbe Ergebnis langsamer.

## Dateistruktur

| Datei | Zweck |
|---|---|
| `.github/workflows/tests.yml` | CI: Push, Pull Request, wöchentlicher Canary |
| `.gitignore` | venv, Caches, Streamlit-Secrets |
| `.streamlit/config.toml` | Streamlit-Einstellungen (keine Nutzungsstatistik) |
| `app.py` | Streamlit-App |
| `data/toronto_campus.json`, `data/LICENSE-ODbL.md` | Straßenkarte (OpenStreetMap, ODbL; Kopie aus `graph-zeichnen-demo`) |
| `requirements-dev.txt` | Laufzeit plus pytest und networkx (nur Gegenprobe) |
| `requirements.txt` | Laufzeit-Bibliotheken (ungepinnt; `scipy` rechnet Triangulierung, Suchbaum und Komponenten) |
| `rnz_algorithm.py` | Kern: räumliche Modelle und Kennzahlen |
| `rnz_constants.py` | Regler-Grenzen, Schwellen, feste Pläne, Farben, Presets |
| `rnz_evaluation.py` | Vergleich, Messreihe, Verdikte, Tabellen |
| `rnz_layout.py` | Fester Kopf (Stylesheet) |
| `rnz_presets.py` | Permalink (Begrenzen), Presets (Schnellstart in der Seitenleiste, aktives Preset hervorgehoben), Zufalls-Seed-Button |
| `rnz_scenario.py` | Kartenausschnitt und Modelle auf seinen Standorten |
| `rnz_visualization.py` | Plotly-Abbildungen (Achsen gesperrt) |
| `messen.py` | Einmalige Messung der Zahlen in dieser README (nicht Teil der App) |
| `tests/` | Mini-Instanzen, Kern, Orakel, Verdikte, Randfälle, Presets, Abbildungen, Layout, AppTest, Footer, Laufzeit-Importe, Claims |

## Bewusst nicht umgesetzt

- **Weitere Städte oder Karten hochladen:** würde Datenschutz- und Lizenzprüfung je Quelle verlangen; die Demo bleibt bei einer eingebetteten ODbL-Karte.
- **Modelle mit Stellschrauben und Anpassung** (zum Beispiel Länge und Grad gemeinsam): würde „keines passt“ in „eines passt, wenn man es anpasst“ verwandeln, ein eigenes Stück.
- **Weitere Kennzahlen** (Betweenness, Kreise, Kantenlängenverteilung als Vergleichsmaß): die Kantenlänge steht in der Tabelle, zählt aber nicht als Kennzahl.
- **Straßenverläufe statt Strecken:** die Kanten sind gerade Verbindungen.

## Lokal ausführen

```bash
python3 -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate

pip install -r requirements.txt
streamlit run app.py
```

## Tests ausführen

```bash
pip install -r requirements-dev.txt
pytest tests/ -v
```

Gebaut mit Streamlit, Plotly, NumPy und SciPy.

---

Diese Demo ist Teil des Portfolios von [Sebastian Hanisch](https://sebastianhanisch.net) – Operations Research und Machine Learning ([Über mich](https://sebastianhanisch.net/ueber-mich.html)). Mehr zu den Demos: [Interaktive Demos](https://sebastianhanisch.net/demos.html).
