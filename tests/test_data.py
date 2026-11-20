"""Tests des données : exigences du cahier des charges (>= 10 000 lignes, >= 10 attributs)."""
import pytest

from ml_utils import CLEAN_PATH, RAW_PATH, TARGET, load_clean_data, load_raw_data

pytestmark = pytest.mark.skipif(not RAW_PATH.exists(), reason="data/raw/reviews.csv absent")


def test_raw_dataset_meets_course_requirements():
    raw = load_raw_data()
    assert len(raw) >= 10_000, "le dataset brut doit avoir au moins 10 000 lignes"
    assert raw.shape[1] >= 10, "le dataset brut doit avoir au moins 10 attributs"


@pytest.mark.skipif(not CLEAN_PATH.exists(), reason="reviews_clean.csv absent")
def test_clean_target_is_binary_and_complete():
    clean = load_clean_data()
    assert set(clean[TARGET].unique()) == {0, 1}
    assert clean[TARGET].notna().all()


@pytest.mark.skipif(not CLEAN_PATH.exists(), reason="reviews_clean.csv absent")
def test_clean_dataset_has_no_duplicates_and_expected_columns():
    clean = load_clean_data()
    expected = {"Age", "Rating", "Title", "Review Text", "Division Name", "Department Name", "Class Name", TARGET}
    assert expected.issubset(clean.columns)
    assert not clean.duplicated().any()


@pytest.mark.skipif(not CLEAN_PATH.exists(), reason="reviews_clean.csv absent")
def test_class_imbalance_matches_documentation():
    """Le README annonce 82 % / 18 % : si les données changent, la doc doit suivre."""
    share = load_clean_data()[TARGET].mean()
    assert 0.80 <= share <= 0.84
