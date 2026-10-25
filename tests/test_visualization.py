"""Abbildungen: Spuren je Figur, Werte, Farben, Achsen gesperrt."""

import numpy as np
import pytest

import rnz_constants as C
import rnz_evaluation as ev
import rnz_visualization as V


def _locked(fig):
    return fig.layout.xaxis.fixedrange is True and fig.layout.yaxis.fixedrange is True


@pytest.fixture(scope="module")
def analysis():
    return ev.analyse(ev.Settings(model="gabriel", n=500, seed=2))


def test_map_figure_has_one_segment_per_edge_and_equal_scales(analysis):
    fig = V.build_map(analysis.pos, analysis.ref_edges, C.COLOR_MEASURED, "Straßenkarte")
    assert fig.data[0].name == "Straßenkarte" and fig.data[0].x.count(None) == len(analysis.ref_edges) and _locked(fig)
    u, v = analysis.ref_edges[0]
    assert fig.data[0].x[:3] == (analysis.pos[u][0], analysis.pos[v][0], None) and fig.data[0].y[:3] == (analysis.pos[u][1], analysis.pos[v][1], None)
    assert fig.layout.yaxis.scaleanchor == "x" and fig.layout.xaxis.range is None
    assert fig.data[1].marker.opacity == 0 and fig.data[1].x == (analysis.pos[:, 0].min(), analysis.pos[:, 0].max())       # unsichtbare Eckpunkte


def test_degree_figure_compares_map_and_model(analysis):
    fig = V.build_degrees(analysis.ref.shares, analysis.stats.shares)
    assert [t.name for t in fig.data] == ["Straßenkarte", "Modell"] and _locked(fig)
    assert list(fig.data[0].x)[-1] == "≥ 9" and len(fig.data[0].x) == 10 and list(fig.data[0].x)[:3] == ["0", "1", "2"]
    assert list(fig.data[0].y) == pytest.approx([float(v) * 100 for v in analysis.ref.shares]) and sum(fig.data[1].y) == pytest.approx(100.0)


def test_series_figure_marks_the_map_and_open_circles_for_small_giants():
    def row(model, name, giant, trans, hops):
        return dict(requested=None, model=model, name=name, giant=giant, transitivity=trans, hops=hops)

    rows = [row("map", "Straßenkarte", 1.0, 0.03, 30.0), row("gabriel", "Gabriel-Graph", 1.0, 0.27, 34.0), row("geometric", "Geometrischer Zufallsgraph", 0.02, 0.73, 7.0), dict(row("gnm", "G(n, m)", 0.9, 0.001, 8.0), requested=500)]
    fig = V.build_series(rows, None)
    assert [t.name for t in fig.data] == ["Straßenkarte", "Gabriel-Graph", "Geometrischer Zufallsgraph"] and _locked(fig)
    assert fig.data[0].marker.symbol == "star" and (fig.data[0].x[0], fig.data[0].y[0]) == (30.0, 0.03)
    assert fig.data[1].marker.color != "rgba(0,0,0,0)" and fig.data[2].marker.color == "rgba(0,0,0,0)"        # offener Kreis: Riesenkomponente unter der Hälfte
    rows.append(dict(row("map", "Straßenkarte", 1.0, 0.04, 20.0), requested=500))
    small = V.build_series(rows, 500)
    assert [t.name for t in small.data] == ["Straßenkarte", "G(n, m)"]              # nur die Netze des gewählten Ausschnitts
