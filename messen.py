"""Einmalige Messung der README-Zahlen (nicht Teil der App): feste Pläne, Ausgabe auf stdout."""

import sys
import time

sys.path.insert(0, ".")
import rnz_constants as C
import rnz_evaluation as ev

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

t0 = time.time()
print("PRESETS")
for name, p in C.PRESETS.items():
    a = ev.analyse(ev.Settings(**p))
    s, r = a.stats, a.ref
    print(name, "n", len(a.pos), "Karte m", r.m, "giant", round(r.giant, 3), "trans", round(r.transitivity, 3), "hops", round(r.hops, 1), "cross", r.crossings, "| Modell m", s.m, "giant", round(s.giant, 3),
          "trans", round(s.transitivity, 3), "hops", round(s.hops, 1), "cross", s.crossings, "kmax", s.kmax, "tv", round(a.tv, 3), "trifft", a.passed, {k: int(v) for k, v in a.checks.items()})
print("SERIE")
for r in ev.series_models():
    print(r["n"], f"{r['name']:42s}", "m", r["m"], "giant", round(r["giant"], 3), "trans", round(r["transitivity"], 3), "hops", round(r["hops"], 1), "cross", r["crossings"], "kmax", r["kmax"], "tv", round(r["tv"], 3),
          "len", round(r["len_median"]), round(r["len_p95"]), "trifft", r["passed"], {k: int(v) for k, v in r["checks"].items()})
print("sec", round(time.time() - t0, 1))
