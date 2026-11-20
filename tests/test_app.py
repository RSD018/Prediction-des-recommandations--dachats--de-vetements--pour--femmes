"""Test de fumée de l'application Streamlit : chaque page s'affiche sans exception."""
import pytest

st_testing = pytest.importorskip("streamlit.testing.v1")

from ml_utils import ROOT  # noqa: E402

APP = ROOT / "src" / "app.py"


def _run_page(index):
    at = st_testing.AppTest.from_file(str(APP), default_timeout=120).run()
    assert not at.exception, [e.value for e in at.exception]
    if index:
        radio = at.sidebar.radio[0]
        radio.set_value(radio.options[index]).run()
    return at


def _n_pages():
    at = st_testing.AppTest.from_file(str(APP), default_timeout=120).run()
    return len(at.sidebar.radio[0].options)


@pytest.mark.parametrize("index", range(_n_pages()))
def test_every_page_renders_without_exception(index):
    at = _run_page(index)
    assert not at.exception, [e.value for e in at.exception]


def _section_titles(at):
    return [m.value for m in at.markdown if 'class="section-title"' in m.value]


def _go_to(at, label_part):
    radio = at.sidebar.radio[0]
    option = next(o for o in radio.options if label_part in o)
    radio.set_value(option).run()
    assert not at.exception, [e.value for e in at.exception]
    return at


def test_every_page_shows_a_section_title():
    at = st_testing.AppTest.from_file(str(APP), default_timeout=120).run()
    for option in at.sidebar.radio[0].options:
        at.sidebar.radio[0].set_value(option).run()
        assert _section_titles(at), f"la page « {option} » n'affiche aucun titre de section"


def test_error_analysis_page_is_not_empty_and_not_shown_elsewhere():
    """Régression : deux blocs `if page == PAGES[7]` rendaient « Analyse des erreurs » vide
    et la collaient sous « À propos »."""
    at = st_testing.AppTest.from_file(str(APP), default_timeout=120).run()
    _go_to(at, "Analyse des erreurs")
    assert any("Analyse des erreurs" in t for t in _section_titles(at))
    _go_to(at, "À propos")
    assert not any("Analyse des erreurs" in t for t in _section_titles(at))


def test_eda_dashboard_has_interactive_filters():
    at = _run_page(3)  # page « Statistiques du Dataset »
    assert len(at.multiselect) >= 1 and len(at.slider) >= 2 and len(at.selectbox) >= 1


def test_prediction_form_returns_a_result_and_an_explanation():
    at = _run_page(1)  # page « Prédiction en Direct »
    at.button[0].click().run()  # bouton du formulaire
    assert not at.exception, [e.value for e in at.exception]
    assert not at.error, [e.value for e in at.error]
    assert any("Article" in m.value and "recommandé" in m.value for m in at.markdown)


def test_model_analyses_are_integrated_into_performance_page():
    at = st_testing.AppTest.from_file(str(APP), default_timeout=120).run()
    assert not any("Résultats complémentaires" in option for option in at.sidebar.radio[0].options)
    _go_to(at, "Performance des Modèles")
    assert any("Comparatif des Modèles Classiques" in title for title in _section_titles(at))
    assert any(slider.key and slider.key.startswith("thr_value_") for slider in at.slider)
    assert not at.exception, [e.value for e in at.exception]


def test_model_analysis_threshold_slider_changes_the_metrics():
    """Le curseur de seuil de l'onglet interactif doit réellement modifier la matrice de confusion affichée."""
    at = st_testing.AppTest.from_file(str(APP), default_timeout=120).run()
    _go_to(at, "Performance des Modèles")
    assert not at.exception, [e.value for e in at.exception]
    before = [m.value for m in at.metric]
    slider = next(s for s in at.slider if s.key and s.key.startswith("thr_value_"))
    slider.set_value(slider.max - 0.01).run()
    assert not at.exception, [e.value for e in at.exception]
    after = [m.value for m in at.metric]
    assert before and after and before != after
