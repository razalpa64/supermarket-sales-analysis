"""
Unit tests for SQLite database table initialization, migration, and analytical queries.
"""

import tempfile
from pathlib import Path
import pandas as pd
import pytest

from src.database import SupermarketDatabase
from src.data_cleaning import clean_and_prepare_data


def test_sqlite_database_lifecycle():
    with tempfile.TemporaryDirectory() as tmpdir:
        test_db_path = Path(tmpdir) / "test_supermarket.db"
        db = SupermarketDatabase(db_path=test_db_path)
        db.initialize_tables()

        clean_df, daily_df = clean_and_prepare_data(save_outputs=False)

        # Import transactions
        n_trans = db.import_transactions_from_df(clean_df)
        assert n_trans == 1000

        # Import daily sales
        n_daily = db.import_daily_sales_from_df(daily_df)
        assert n_daily == 89

        # Run analytical queries
        top_prods = db.get_top_selling_products(limit=3)
        assert len(top_prods) == 3
        assert "product_line" in top_prods.columns

        branch_perf = db.get_branch_performance()
        assert len(branch_perf) == 3
        assert "branch" in branch_perf.columns

