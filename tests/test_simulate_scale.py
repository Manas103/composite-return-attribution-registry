"""Measures the actual generated scale of the simulated book (the
"simulated 240-account, 3-year book" existence/scale claim)."""
from cra.simulate import generate_book


def test_full_book_scale(full_book):
    assert full_book.n_accounts == 240
    assert full_book.n_days == 756  # 3 x 252 trading days
    assert full_book.n_securities == 60
    assert len(full_book.composites) == 3
    assert len(full_book.positions) > 0
    n_months = len(full_book.month_boundaries())
    assert n_months >= 34  # roughly 36 calendar months across 756 trading days
