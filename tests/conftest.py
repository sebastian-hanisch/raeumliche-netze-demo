"""Gemeinsame Fixtures: die Mini-Instanzen, an denen jede Einheit von Hand geprüft wird (DEMO-PLAYBOOK §1b)."""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


@pytest.fixture
def square():
    """Quadrat mit Seitenlänge 2 und Mittelpunkt: Punkte 0 (0,0), 1 (2,0), 2 (2,2), 3 (0,2), 4 (1,1). Delaunay: vier Seiten plus vier Kanten zum Mittelpunkt, die Diagonalen kreuzen sich im Mittelpunkt."""
    return np.array([[0.0, 0.0], [2.0, 0.0], [2.0, 2.0], [0.0, 2.0], [1.0, 1.0]])


@pytest.fixture
def path4():
    """Weg 0–1–2–3 auf einer Geraden mit Abständen 1, 2, 3 (Punkte bei x = 0, 1, 3, 6)."""
    return np.array([[0.0, 0.0], [1.0, 0.0], [3.0, 0.0], [6.0, 0.0]]), [(0, 1), (1, 2), (2, 3)]
