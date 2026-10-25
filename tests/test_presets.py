"""Presets: Vollständigkeit, gültige Werte, Standard-Preset = Standardeinstellungen, keiner auf einem Reglerrand und keiner, der nach einem Modellwechsel allein entsteht."""

import rnz_constants as C
import rnz_evaluation as ev
import rnz_presets as P


def test_every_preset_has_help_and_all_keys():
    assert set(C.PRESETS) == set(C.PRESET_HELP)
    for name, preset in C.PRESETS.items():
        assert set(preset) == set(P.PRESET_KEYS), name
        assert C.PRESET_HELP[name]


def test_preset_values_are_inside_the_slider_bounds():
    for name, p in C.PRESETS.items():
        assert p["model"] in C.MODELS and C.N_MIN <= p["n"] <= C.N_MAX and C.K_MIN <= p["k"] <= C.K_MAX and C.SCALE_MIN <= p["scale"] <= C.SCALE_MAX and 0 <= p["seed"] <= C.SEED_MAX, name


def test_default_preset_equals_the_default_settings():
    assert ev.Settings(**C.PRESETS["Geometrischer Zufallsgraph"]) == ev.Settings()


def test_setting_specs_cover_every_preset_key():
    assert set(P.PRESET_KEYS.values()) == set(P.SETTING_SPECS)


def test_no_preset_sits_on_a_slider_bound():
    """Die Layout-Tests setzen jeden Regler auf seinen Rand: ein Preset dort würde die Hervorhebung des Standard-Presets verfälschen."""
    bounds = {"n": (C.N_MIN, C.N_MAX), "k": (C.K_MIN, C.K_MAX), "scale": (C.SCALE_MIN, C.SCALE_MAX)}
    for name, preset in C.PRESETS.items():
        for key, (lo, hi) in bounds.items():
            assert preset[key] not in (lo, hi), (name, key)


def test_a_model_change_alone_never_produces_another_preset():
    """Wechselt man vom Standard nur das Modell, darf kein anderes Preset genau passen (sonst hebt der Layout-Test die Hervorhebung nicht auf)."""
    default = C.PRESETS["Geometrischer Zufallsgraph"]
    for name, p in C.PRESETS.items():
        if name != "Geometrischer Zufallsgraph":
            assert {k: v for k, v in p.items() if k != "model"} != {k: v for k, v in default.items() if k != "model"}, name


def test_presets_are_pairwise_different_and_cover_the_findings():
    values = [tuple(sorted(p.items())) for p in C.PRESETS.values()]
    assert len(set(values)) == len(values)
    assert {p["model"] for p in C.PRESETS.values()} == {"geometric", "knn", "gabriel", "delaunay_short", "gnm"}


def test_slider_steps_reach_the_presets():
    grid = {"n": (C.N_MIN, 100), "k": (C.K_MIN, 1), "scale": (C.SCALE_MIN, 10)}
    for name, p in C.PRESETS.items():
        for key, (lo, step) in grid.items():
            k = (p[key] - lo) / step
            assert abs(k - round(k)) < 1e-9, (name, key)
