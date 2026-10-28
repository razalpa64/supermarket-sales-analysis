"""
SQLite Database Integration Module for Supermarket Sales System.
Provides optional relational database support: schema creation, importing CSV data,
running analytical SQL queries, saving model metrics, and querying back into Pandas.
"""

import sys
import sqlite3
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple
import pandas as pd

# Add project root to sys.path
BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from src.config import (
    DATABASE_PATH,
    CLEANED_DATA_CSV,
    DAILY_SALES_FILE,
    REPORTS_DIR,
)


class SupermarketDatabase:
    """
    Manages SQLite database connection, table initialization, and analytical queries.
    """

    def __init__(self, db_path: Optional[Path] = None):
        self.db_path = db_path or DATABASE_PATH
        self.db_path.parent.mkdir(parents=True, exist_ok=True)

    def get_connection(self) -> sqlite3.Connection:
        """
        Creates and returns an active SQLite connection.
        """
        conn = sqlite3.connect(str(self.db_path))
        conn.row_factory = sqlite3.Row
        return conn

    def initialize_tables(self) -> None:
        """
        Creates transactions, daily_sales, and model_metrics tables if they do not exist.
        """
        conn = self.get_connection()
        try:
            cursor = conn.cursor()
            cursor.execute("""
            CREATE TABLE IF NOT EXISTS transactions (
                invoice_id TEXT PRIMARY KEY,
                branch TEXT NOT NULL,
                city TEXT NOT NULL,
                customer_type TEXT NOT NULL,
                gender TEXT NOT NULL,
                product_line TEXT NOT NULL,
                unit_price REAL NOT NULL,
                quantity INTEGER NOT NULL,
                tax_5 REAL NOT NULL,
                total REAL NOT NULL,
                date TEXT NOT NULL,
                time TEXT NOT NULL,
                payment TEXT NOT NULL,
                cogs REAL NOT NULL,
                gross_margin_pct REAL,
                gross_income REAL NOT NULL,
                rating REAL NOT NULL,
                hour INTEGER,
                minute INTEGER,
                day_of_week INTEGER,
                day_name TEXT,
                is_weekend INTEGER
            );
            """)

            cursor.execute("""
            CREATE TABLE IF NOT EXISTS daily_sales (
                date TEXT PRIMARY KEY,
                daily_revenue REAL NOT NULL,
                transaction_count INTEGER NOT NULL,
                total_quantity INTEGER,
                average_transaction_value REAL,
                day_of_week INTEGER,
                day_name TEXT,
                is_weekend INTEGER
            );
            """)

            cursor.execute("""
            CREATE TABLE IF NOT EXISTS model_metrics (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                scenario TEXT NOT NULL,
                algorithm TEXT NOT NULL,
                category TEXT,
                mae REAL,
                rmse REAL,
                r2 REAL,
                mape_pct REAL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
            """)

            cursor.execute("CREATE INDEX IF NOT EXISTS idx_trans_date ON transactions(date);")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_trans_branch ON transactions(branch);")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_trans_prod ON transactions(product_line);")
            conn.commit()
        finally:
            conn.close()

        print(f"[SUCCESS] Initialized SQLite tables and indices at: {self.db_path}")

    def import_transactions_from_df(self, df: pd.DataFrame, replace: bool = True) -> int:
        """
        Imports cleaned transactions into SQLite table.
        """
        self.initialize_tables()
        df_to_save = df.copy()

        col_mapping = {
            "Invoice ID": "invoice_id",
            "Branch": "branch",
            "City": "city",
            "Customer type": "customer_type",
            "Gender": "gender",
            "Product line": "product_line",
            "Unit price": "unit_price",
            "Quantity": "quantity",
            "Tax 5%": "tax_5",
            "Total": "total",
            "Date": "date",
            "Time": "time",
            "Payment": "payment",
            "cogs": "cogs",
            "gross margin percentage": "gross_margin_pct",
            "gross income": "gross_income",
            "Rating": "rating",
            "Hour": "hour",
            "Minute": "minute",
            "DayOfWeek": "day_of_week",
            "DayName": "day_name",
            "IsWeekend": "is_weekend",
        }
        df_to_save = df_to_save.rename(columns=col_mapping)
        if "date" in df_to_save.columns and pd.api.types.is_datetime64_any_dtype(df_to_save["date"]):
            df_to_save["date"] = df_to_save["date"].dt.strftime("%Y-%m-%d")

        valid_cols = [c for c in col_mapping.values() if c in df_to_save.columns]
        df_to_save = df_to_save[valid_cols]

        conn = self.get_connection()
        try:
            df_to_save.to_sql(
                "transactions",
                conn,
                if_exists="replace" if replace else "append",
                index=False,
            )
        finally:
            conn.close()
        print(f"[SUCCESS] Imported {len(df_to_save)} transaction records into SQLite 'transactions'.")
        return len(df_to_save)


    def import_daily_sales_from_df(self, daily_df: pd.DataFrame, replace: bool = True) -> int:
        """
        Imports daily sales summary into SQLite table.
        """
        self.initialize_tables()
        df_to_save = daily_df.copy()
        col_mapping = {
            "Date": "date",
            "Daily_Revenue": "daily_revenue",
            "Transaction_Count": "transaction_count",
            "Total_Quantity": "total_quantity",
            "Average_Transaction_Value": "average_transaction_value",
            "DayOfWeek": "day_of_week",
            "DayName": "day_name",
            "IsWeekend": "is_weekend",
        }
        df_to_save = df_to_save.rename(columns=col_mapping)
        if "date" in df_to_save.columns and pd.api.types.is_datetime64_any_dtype(df_to_save["date"]):
            df_to_save["date"] = df_to_save["date"].dt.strftime("%Y-%m-%d")

        valid_cols = [c for c in col_mapping.values() if c in df_to_save.columns]
        df_to_save = df_to_save[valid_cols]

        conn = self.get_connection()
        try:
            df_to_save.to_sql(
                "daily_sales",
                conn,
                if_exists="replace" if replace else "append",
                index=False,
            )
        finally:
            conn.close()
        print(f"[SUCCESS] Imported {len(df_to_save)} daily sales records into SQLite 'daily_sales'.")
        return len(df_to_save)

    def import_model_metrics_reports(self) -> int:
        """
        Imports CSV model evaluation reports into the model_metrics table.
        """
        self.initialize_tables()
        imported = 0
        conn = self.get_connection()
        try:
            # Scenario A
            rep_a = REPORTS_DIR / "scenario_a_model_comparison.csv"
            if rep_a.exists():
                df_a = pd.read_csv(rep_a)
                for _, row in df_a.iterrows():
                    conn.execute("""
                        INSERT INTO model_metrics (scenario, algorithm, category, mae, rmse, r2, mape_pct)
                        VALUES (?, ?, ?, ?, ?, ?, ?)
                    """, ("Scenario A", row.get("Algorithm"), row.get("Category"), row.get("Val_MAE"), row.get("Val_RMSE"), row.get("Val_R2"), row.get("Val_MAPE_Pct")))
                    imported += 1

            # Scenario B
            rep_b = REPORTS_DIR / "scenario_b_model_comparison.csv"
            if rep_b.exists():
                df_b = pd.read_csv(rep_b)
                for _, row in df_b.iterrows():
                    conn.execute("""
                        INSERT INTO model_metrics (scenario, algorithm, category, mae, rmse, r2, mape_pct)
                        VALUES (?, ?, ?, ?, ?, ?, ?)
                    """, ("Scenario B", row.get("Algorithm"), row.get("Category"), row.get("Val_MAE"), row.get("Val_RMSE"), row.get("Val_R2"), row.get("Val_MAPE_Pct")))
                    imported += 1

            conn.commit()
        finally:
            conn.close()
        print(f"[SUCCESS] Imported {imported} model evaluation metric records into SQLite.")
        return imported

    def query_to_dataframe(self, sql_query: str, params: Optional[Tuple] = None) -> pd.DataFrame:
        """
        Executes a custom SQL query and returns the result as a Pandas DataFrame.
        """
        conn = self.get_connection()
        try:
            df = pd.read_sql_query(sql_query, conn, params=params)
        finally:
            conn.close()
        return df


    # Analytical SQL queries
    def get_top_selling_products(self, limit: int = 6) -> pd.DataFrame:
        query = f"""
            SELECT 
                product_line,
                ROUND(SUM(total), 2) AS total_revenue,
                SUM(quantity) AS total_units_sold,
                COUNT(invoice_id) AS transaction_count,
                ROUND(AVG(total), 2) AS avg_transaction_value,
                ROUND(AVG(rating), 2) AS avg_rating
            FROM transactions
            GROUP BY product_line
            ORDER BY total_revenue DESC
            LIMIT {limit};
        """
        return self.query_to_dataframe(query)

    def get_branch_performance(self) -> pd.DataFrame:
        query = """
            SELECT 
                branch,
                city,
                ROUND(SUM(total), 2) AS total_revenue,
                COUNT(invoice_id) AS transactions,
                ROUND(AVG(total), 2) AS avg_ticket,
                ROUND(AVG(rating), 2) AS avg_rating
            FROM transactions
            GROUP BY branch, city
            ORDER BY total_revenue DESC;
        """
        return self.query_to_dataframe(query)

    def get_hourly_traffic(self) -> pd.DataFrame:
        query = """
            SELECT 
                hour,
                COUNT(invoice_id) AS transaction_count,
                ROUND(SUM(total), 2) AS total_revenue,
                ROUND(AVG(total), 2) AS avg_revenue
            FROM transactions
            WHERE hour IS NOT NULL
            GROUP BY hour
            ORDER BY hour ASC;
        """
        return self.query_to_dataframe(query)


def setup_and_seed_database() -> SupermarketDatabase:
    """
    Helper function to initialize the database and load current processed datasets.
    """
    db = SupermarketDatabase()
    db.initialize_tables()

    if CLEANED_DATA_CSV.exists():
        df_clean = pd.read_csv(CLEANED_DATA_CSV)
        db.import_transactions_from_df(df_clean)

    if DAILY_SALES_FILE.exists():
        df_daily = pd.read_csv(DAILY_SALES_FILE)
        db.import_daily_sales_from_df(df_daily)

    db.import_model_metrics_reports()
    return db


if __name__ == "__main__":
    db = setup_and_seed_database()
    print("\n--- Top Products via SQLite ---")
    print(db.get_top_selling_products())
    print("\n--- Branch Performance via SQLite ---")
    print(db.get_branch_performance())
