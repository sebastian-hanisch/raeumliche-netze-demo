"""Plotly-Abbildungen. Alle Achsen sind gesperrt (fixedrange), damit Touch-Geräte beim Scrollen nicht am Diagramm hängen bleiben."""

import numpy as np
import plotly.graph_objects as go

import rnz_constants as C

MAX_DRAWN_EDGES = 6000       # mehr Kanten werden gleichmäßig ausgewählt (nur für Netze ohne Raumbezug mit kilometerlangen Kanten)


def lock_axes(fig):
    fig.update_xaxes(fixedrange=True)
    fig.update_yaxes(fixedrange=True)
    return fig


def _base(fig, height=420):
    fig.update_layout(height=height, margin=dict(l=10, r=10, t=10, b=10), legend=dict(orientation="h", y=-0.2), plot_bgcolor="rgba(0,0,0,0)")
    return lock_axes(fig)


def select_edges(edges, limit: int = MAX_DRAWN_EDGES):
    """Alle Kanten, bei mehr als `limit` eine gleichmäßig gewählte Auswahl der Größe `limit`."""
    edges = list(edges)
    if len(edges) <= limit:
        return edges
    idx = np.linspace(0, len(edges) - 1, limit).round().astype(int)
    return [edges[i] for i in idx]


def build_map(pos, edges, color: str, name: str):
    """Karte der Kanten als Strecken (Meter); beide Achsen mit gleichem Maßstab (Autorange plus zwei unsichtbare Eckpunkte)."""
    pos = np.asarray(pos, dtype=float)
    shown = select_edges(edges)
    xs, ys = [], []
    for u, v in shown:
        xs += [pos[u][0], pos[v][0], None]
        ys += [pos[u][1], pos[v][1], None]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=xs, y=ys, mode="lines", line=dict(color=color, width=1), name=name, hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=[pos[:, 0].min(), pos[:, 0].max()], y=[pos[:, 1].min(), pos[:, 1].max()], mode="markers", marker=dict(size=1, opacity=0), hoverinfo="skip", showlegend=False))
    fig.update_xaxes(title_text="x (m)", showgrid=False)
    fig.update_yaxes(title_text="y (m)", scaleanchor="x", scaleratio=1, showgrid=False)
    return _base(fig, height=380)


def build_degrees(ref_shares, shares):
    """Anteil der Knoten je Grad: Karte und Modell nebeneinander; der letzte Balken fasst die Grade ab dem größten gezeigten zusammen."""
    k = len(ref_shares) - 1
    labels = [str(i) for i in range(k)] + [f"≥ {k}"]
    fig = go.Figure()
    fig.add_trace(go.Bar(x=labels, y=[float(v) * 100 for v in ref_shares], name="Straßenkarte", marker_color=C.COLOR_MEASURED))
    fig.add_trace(go.Bar(x=labels, y=[float(v) * 100 for v in shares], name="Modell", marker_color=C.COLOR_THIRD))
    fig.update_layout(barmode="group")
    fig.update_xaxes(title_text="Grad")
    fig.update_yaxes(title_text="Anteil der Knoten (%)")
    return _base(fig, height=340)


def build_series(rows, requested):
    """Transitivität über der mittleren Weglänge, ein Punkt je Netz; die Straßenkarte ist der Stern. Offene Kreise: die Weglänge ist die eines kleinen Rests (Riesenkomponente unter der Hälfte)."""
    sel = [r for r in rows if r["requested"] == requested]
    fig = go.Figure()
    ref = next(r for r in sel if r["model"] == "map")
    fig.add_trace(go.Scatter(x=[ref["hops"]], y=[ref["transitivity"]], mode="markers", marker=dict(size=18, color=C.COLOR_ALERT, symbol="star"), name=ref["name"],
                             hovertemplate="Straßenkarte: Weglänge %{x:.1f}, Transitivität %{y:.3f}<extra></extra>"))
    palette = [C.COLOR_MEASURED, C.COLOR_THEORY, C.COLOR_THIRD, "#7f7f7f", "#9467bd", "#8c564b", "#17becf"]
    for i, r in enumerate(x for x in sel if x["model"] != "map"):
        small = r["giant"] < C.HOPS_MIN_GIANT
        fig.add_trace(go.Scatter(x=[r["hops"]], y=[r["transitivity"]], mode="markers", name=r["name"],
                                 marker=dict(size=13, color=palette[i % len(palette)] if not small else "rgba(0,0,0,0)", symbol="circle", line=dict(color=palette[i % len(palette)], width=2)),
                                 hovertemplate=f"{r['name']}: Weglänge %{{x:.1f}}, Transitivität %{{y:.3f}}, Riesenkomponente {r['giant'] * 100:.0f} %<extra></extra>"))
    fig.update_xaxes(title_text="mittlere Weglänge (Kanten, in der größten Komponente)")
    fig.update_yaxes(title_text="Transitivität")
    return _base(fig)
