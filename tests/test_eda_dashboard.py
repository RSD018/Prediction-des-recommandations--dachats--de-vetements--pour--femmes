"""Calculs statistiques du tableau de bord (bloc ajouté dans src/app.py), comparés à des valeurs calculées à la main."""
import numpy as np
import pandas as pd
import pytest
from scipy import stats

TARGET = "Recommended IND"


def test_wilson_interval_known_values(app_block):
    low, high = app_block["eda_wilson_interval"]([0, 5], [10, 10])
    # valeurs de référence de l'intervalle de Wilson à 95 % pour 0/10 et 5/10
    assert low[0] == pytest.approx(0.0, abs=1e-9) and high[0] == pytest.approx(0.2775, abs=1e-3)
    assert low[1] == pytest.approx(0.2366, abs=1e-3) and high[1] == pytest.approx(0.7634, abs=1e-3)


def test_wilson_interval_handles_empty_group_and_stays_in_unit_interval(app_block):
    low, high = app_block["eda_wilson_interval"]([0, 3], [0, 3])
    assert np.all((low >= 0) & (high <= 1) & (low <= high))


def test_tukey_summary_counts_outliers(app_block):
    s = pd.Series(list(range(1, 10)) + [100])
    out = app_block["eda_tukey_summary"](s)
    assert out["n"] == 10 and out["n_outliers"] == 1
    assert out["high"] == 9  # la moustache s'arrête à la dernière valeur dans la clôture
    assert out["q1"] < out["median"] < out["q3"]


def test_cramers_v_matches_manual_computation(app_block):
    table = pd.DataFrame([[30, 10], [20, 20], [10, 30]])  # 3 x 2 : pas de correction de Yates
    obs = table.to_numpy().astype(float)
    exp = obs.sum(1, keepdims=True) * obs.sum(0, keepdims=True) / obs.sum()
    chi2_manual = ((obs - exp) ** 2 / exp).sum()
    v_manual = np.sqrt(chi2_manual / (obs.sum() * (min(obs.shape) - 1)))
    chi2, p, v, dof = app_block["eda_cramers_v"](table)
    assert chi2 == pytest.approx(chi2_manual) and v == pytest.approx(v_manual) and dof == 2
    assert 0 < p < 1


def test_mann_whitney_matches_scipy_and_sign_of_effect(app_block):
    rng = np.random.default_rng(0)
    d = pd.DataFrame({TARGET: [1] * 60 + [0] * 40, "Age": np.r_[rng.normal(45, 8, 60), rng.normal(38, 8, 40)]})
    out = app_block["eda_mann_whitney"](d, "Age")
    u, p = stats.mannwhitneyu(d.loc[d[TARGET] == 1, "Age"], d.loc[d[TARGET] == 0, "Age"])
    assert out["U"] == pytest.approx(u) and out["p"] == pytest.approx(p)
    assert out["r"] > 0 and -1 <= out["r"] <= 1  # les « recommandé » ont des valeurs plus élevées
    assert out["n1"] == 60 and out["n0"] == 40


def test_mann_whitney_returns_none_when_a_group_is_too_small(app_block):
    d = pd.DataFrame({TARGET: [1, 1, 1, 0], "Age": [30, 31, 32, 33]})
    assert app_block["eda_mann_whitney"](d, "Age") is None


def test_fmt_p_never_displays_a_misleading_zero(app_block):
    assert app_block["eda_fmt_p"](0.0) == "< 1e-300"
    assert app_block["eda_fmt_p"](1.86e-6) == "1.86e-06"


@pytest.mark.parametrize("value,label", [(0.05, "négligeable"), (-0.2, "faible"), (0.4, "moyen"), (0.7, "fort")])
def test_effect_label_cohen_thresholds(app_block, value, label):
    assert app_block["eda_effect_label"](value) == label


def test_category_table_rates_and_totals(app_block):
    d = pd.DataFrame({"Dept": ["A"] * 4 + ["B"] * 6, TARGET: [1, 1, 1, 0, 1, 0, 0, 0, 0, 1]})
    t = app_block["eda_category_table"](d, "Dept").set_index("Dept")
    assert t["Effectif"].sum() == 10
    assert t.loc["A", "Taux (%)"] == 75.0 and t.loc["B", "Taux (%)"] == pytest.approx(33.3, abs=0.05)
    assert (t["IC95 bas (%)"] <= t["Taux (%)"]).all() and (t["Taux (%)"] <= t["IC95 haut (%)"]).all()


def test_prepare_adds_review_length_and_label(app_block):
    df = pd.DataFrame({"Review Text": ["one two three", ""], TARGET: [1, 0]})
    out = app_block["eda_prepare"](df)
    assert out["Review Length"].tolist() == [3, 0]
    assert out.shape[1] == df.shape[1] + 2
