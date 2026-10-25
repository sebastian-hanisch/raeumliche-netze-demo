"""Szenario: die Straßenkarte (OpenStreetMap, ODbL) und das Modellnetz auf denselben Standorten."""

import json
from collections import deque
from functools import lru_cache
from pathlib import Path

import numpy as np

import rnz_algorithm as A

DATA = Path(__file__).resolve().parent / "data"


@lru_cache(maxsize=1)
def _raw():
    payload = json.loads((DATA / "toronto_campus.json").read_text(encoding="utf-8"))
    pos = np.array([(float(x), float(y)) for x, y in payload["nodes_xy_m"]])
    return pos, A.normalize((u, v) for u, v, _ in payload["arcs"])


def _adjacency(edges):
    adj: dict[int, list[int]] = {}
    for u, v in edges:
        adj.setdefault(u, []).append(v)
        adj.setdefault(v, []).append(u)
    return adj


def make_map(n: int | None = None):
    """Die Straßenkarte als (Standorte in Metern, Kanten). Ohne Angabe die größte Komponente um Knoten 0 (5 072 Knoten), sonst die n Knoten um die Mitte der Karte
    (Breitensuche ab dem Knoten nächst dem Median der Lage, Nachbarn aufsteigend), neu nummeriert."""
    pos, edges = _raw()
    adj = _adjacency(edges)
    if n is None:
        seen = {0}
        stack = [0]
        while stack:
            x = stack.pop()
            for y in adj.get(x, ()):
                if y not in seen:
                    seen.add(y)
                    stack.append(y)
        keep = sorted(seen)
    else:
        centre = np.median(pos, axis=0)
        start = int(np.argmin(np.linalg.norm(pos - centre, axis=1)))
        order, seen, queue = [start], {start}, deque([start])
        while queue and len(order) < n:
            x = queue.popleft()
            for y in sorted(adj.get(x, ())):
                if y not in seen and len(order) < n:
                    seen.add(y)
                    order.append(y)
                    queue.append(y)
        keep = sorted(order)
    index = {v: i for i, v in enumerate(keep)}
    return pos[keep], A.normalize((index[u], index[v]) for u, v in edges if u in index and v in index)


def build_model(model: str, pos, m: int, k: int, scale: float, rng: np.random.Generator):
    """Kanten des gewählten Modells auf den Standorten `pos`; alle Modelle außer k nächste Nachbarn und Gabriel-Graph erhalten die Kantenzahl m
    (Gabriel und k nächste Nachbarn haben ihre eigene Zahl)."""
    n = len(pos)
    if model == "geometric":
        return A.geometric_edges(pos, A.radius_for_edges(pos, m))
    if model == "knn":
        return A.knn_edges(pos, k)
    if model == "delaunay_random":
        return A.thin_random(A.delaunay_edges(pos), m, rng)
    if model == "delaunay_short":
        return A.thin_shortest(pos, A.delaunay_edges(pos), m)
    if model == "gabriel":
        return A.gabriel_edges(pos)
    if model == "waxman":
        return A.waxman_edges(pos, A.delaunay_edges(pos), scale, m, rng)
    if model == "gnm":
        return A.gnm_edges(n, m, rng)
    raise ValueError(f"unbekanntes Modell: {model}")
