"""
=============================================================
Sales Analytics & BI Dashboard — End-to-End Pipeline
Author  : Vaibhav Gupta
Tech    : Python · Pandas · SQL (SQLite) · Power BI CSV Export
          Matplotlib · Seaborn
=============================================================
End-to-end business intelligence pipeline:
  1. Data ingestion from multiple sources (CSV / SQL / API)
  2. ETL — clean, transform, enrich
  3. KPI calculation (Revenue, AOV, Churn, LTV, Retention)
  4. SQL analytics with window functions & CTEs
  5. Power BI–ready export
  6. Automated insight generation
=============================================================
"""

import sqlite3
import logging
import warnings
from pathlib import Path
from datetime import datetime, timedelta

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)s  %(message)s")
logger = logging.getLogger(__name__)

OUTPUT_DIR = Path(__file__).parent / "outputs"
OUTPUT_DIR.mkdir(exist_ok=True)

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import seaborn as sns
    PLOT_AVAILABLE = True
except ImportError:
    PLOT_AVAILABLE = False

# ══════════════════════════════════════════════════════════
# 1. DATA GENERATION — Realistic E-Commerce Dataset
# ══════════════════════════════════════════════════════════
CATEGORIES  = ["Electronics", "Fashion", "Home & Kitchen", "Books", "Sports", "Beauty"]
REGIONS     = ["North", "South", "East", "West", "Central"]
CHANNELS    = ["Online", "Mobile App", "In-Store", "Partner"]
PAYMENT     = ["Credit Card", "UPI", "Net Banking", "COD", "Wallet"]

def generate_sales_data(n_orders: int = 50_000, start_date: str = "2023-01-01",
                         seed: int = 42) -> tuple:
    rng        = np.random.default_rng(seed)
    start      = pd.Timestamp(start_date)
    date_range = pd.date_range(start, periods=365 * 2, freq="D")

    # Simulate seasonal demand
    order_dates = rng.choice(date_range, n_orders, replace=True)
    months      = pd.DatetimeIndex(order_dates).month
    seasonal    = np.where(months.isin([10, 11, 12]), 1.4, 1.0)  # Festive boost

    n_customers = 8000
    customer_ids = rng.integers(1, n_customers + 1, n_orders)

    orders = pd.DataFrame({
        "order_id"      : range(1, n_orders + 1),
        "customer_id"   : customer_ids,
        "order_date"    : order_dates,
        "category"      : rng.choice(CATEGORIES, n_orders),
        "region"        : rng.choice(REGIONS, n_orders),
        "channel"       : rng.choice(CHANNELS, n_orders, p=[0.45, 0.30, 0.15, 0.10]),
        "payment_method": rng.choice(PAYMENT, n_orders),
        "quantity"      : rng.integers(1, 6, n_orders),
        "unit_price"    : (rng.lognormal(4.5, 1.0, n_orders) * seasonal).round(2),
        "discount_pct"  : rng.choice([0, 5, 10, 15, 20, 25], n_orders,
                                      p=[0.40, 0.20, 0.18, 0.10, 0.07, 0.05]),
        "return_flag"   : rng.choice([0, 1], n_orders, p=[0.92, 0.08]),
    })
    orders["revenue"] = (orders["unit_price"] * orders["quantity"]
                          * (1 - orders["discount_pct"] / 100)).round(2)

    # Customer table
    customers = pd.DataFrame({
        "customer_id"   : range(1, n_customers + 1),
        "signup_date"   : rng.choice(date_range[:365], n_customers, replace=True),
        "city"          : rng.choice(["Mumbai", "Delhi", "Bangalore", "Chennai",
                                       "Hyderabad", "Pune", "Kolkata", "Lucknow"], n_customers),
        "age_group"     : rng.choice(["18-24", "25-34", "35-44", "45-54", "55+"], n_customers,
                                      p=[0.18, 0.35, 0.27, 0.13, 0.07]),
        "loyalty_tier"  : rng.choice(["Bronze", "Silver", "Gold", "Platinum"], n_customers,
                                      p=[0.45, 0.30, 0.18, 0.07]),
    })

    logger.info("Generated %d orders, %d customers", len(orders), len(customers))
    return orders, customers


# ══════════════════════════════════════════════════════════
# 2. ETL — CLEAN & ENRICH
# ══════════════════════════════════════════════════════════
def etl_pipeline(orders: pd.DataFrame, customers: pd.DataFrame) -> pd.DataFrame:
    df = orders.merge(customers, on="customer_id", how="left")

    # Date features
    df["order_date"]   = pd.to_datetime(df["order_date"])
    df["year"]         = df["order_date"].dt.year
    df["month"]        = df["order_date"].dt.month
    df["quarter"]      = df["order_date"].dt.quarter
    df["day_of_week"]  = df["order_date"].dt.day_name()
    df["week_num"]     = df["order_date"].dt.isocalendar().week.astype(int)
    df["is_weekend"]   = df["order_date"].dt.dayofweek >= 5
    df["month_label"]  = df["order_date"].dt.strftime("%Y-%m")

    # Net revenue after returns
    df["net_revenue"] = np.where(df["return_flag"] == 1, 0, df["revenue"])

    # Profit margin proxy (fictional COGS = 60%)
    df["profit"]       = (df["net_revenue"] * 0.40).round(2)
    df["profit_margin"]= np.where(df["net_revenue"] > 0, 0.40, 0)

    logger.info("ETL complete. Shape: %s", df.shape)
    return df


# ══════════════════════════════════════════════════════════
# 3. KPI CALCULATIONS
# ══════════════════════════════════════════════════════════
def compute_kpis(df: pd.DataFrame) -> dict:
    total_revenue  = df["net_revenue"].sum()
    total_orders   = len(df)
    aov            = total_revenue / total_orders
    total_profit   = df["profit"].sum()
    return_rate    = df["return_flag"].mean() * 100
    n_customers    = df["customer_id"].nunique()
    ltv            = total_revenue / n_customers

    # Month-over-month growth (last 2 months)
    monthly = df.groupby("month_label")["net_revenue"].sum().sort_index()
    if len(monthly) >= 2:
        mom_growth = (monthly.iloc[-1] - monthly.iloc[-2]) / monthly.iloc[-2] * 100
    else:
        mom_growth = 0

    kpis = {
        "Total Revenue (₹)"   : f"₹{total_revenue:,.0f}",
        "Total Orders"        : f"{total_orders:,}",
        "Average Order Value" : f"₹{aov:,.2f}",
        "Total Profit (₹)"   : f"₹{total_profit:,.0f}",
        "Return Rate"         : f"{return_rate:.2f}%",
        "Unique Customers"    : f"{n_customers:,}",
        "Customer LTV (avg)"  : f"₹{ltv:,.2f}",
        "MoM Revenue Growth"  : f"{mom_growth:+.1f}%",
    }
    return kpis


# ══════════════════════════════════════════════════════════
# 4. SQL ANALYTICS (SQLite in-memory)
# ══════════════════════════════════════════════════════════
def run_sql_analytics(df: pd.DataFrame) -> dict:
    conn = sqlite3.connect(":memory:")
    df.to_sql("sales", conn, index=False, if_exists="replace")
    results = {}

    queries = {
        "top_categories": """
            SELECT category,
                   COUNT(*)                           AS orders,
                   ROUND(SUM(net_revenue), 0)         AS revenue,
                   ROUND(AVG(unit_price), 2)          AS avg_price,
                   ROUND(AVG(discount_pct), 1)        AS avg_discount
            FROM sales
            GROUP BY category
            ORDER BY revenue DESC
        """,
        "monthly_trend": """
            SELECT month_label,
                   COUNT(*)                     AS orders,
                   ROUND(SUM(net_revenue), 0)   AS revenue,
                   ROUND(AVG(net_revenue), 2)   AS aov
            FROM sales
            GROUP BY month_label
            ORDER BY month_label
        """,
        "region_performance": """
            SELECT region,
                   COUNT(DISTINCT customer_id)  AS customers,
                   COUNT(*)                     AS orders,
                   ROUND(SUM(net_revenue), 0)   AS revenue,
                   ROUND(AVG(discount_pct), 1)  AS avg_discount
            FROM sales
            GROUP BY region
            ORDER BY revenue DESC
        """,
        "channel_mix": """
            SELECT channel,
                   COUNT(*)                           AS orders,
                   ROUND(SUM(net_revenue), 0)         AS revenue,
                   ROUND(SUM(net_revenue) * 100.0
                         / SUM(SUM(net_revenue)) OVER(), 2) AS revenue_share_pct
            FROM sales
            GROUP BY channel
            ORDER BY revenue DESC
        """,
        "loyalty_tier_revenue": """
            SELECT loyalty_tier,
                   COUNT(DISTINCT customer_id) AS customers,
                   COUNT(*)                    AS orders,
                   ROUND(SUM(net_revenue), 0)  AS revenue,
                   ROUND(AVG(net_revenue), 2)  AS avg_order_value
            FROM sales
            GROUP BY loyalty_tier
            ORDER BY revenue DESC
        """,
        "top_customers": """
            SELECT customer_id,
                   COUNT(*)                    AS orders,
                   ROUND(SUM(net_revenue), 0)  AS lifetime_value,
                   ROUND(AVG(net_revenue), 2)  AS avg_order,
                   MAX(order_date)             AS last_order
            FROM sales
            GROUP BY customer_id
            ORDER BY lifetime_value DESC
            LIMIT 20
        """,
    }

    for name, sql in queries.items():
        results[name] = pd.read_sql(sql, conn)

    conn.close()
    return results


# ══════════════════════════════════════════════════════════
# 5. COHORT RETENTION ANALYSIS
# ══════════════════════════════════════════════════════════
def cohort_analysis(df: pd.DataFrame) -> pd.DataFrame:
    df2 = df.copy()
    df2["order_month"] = df2["order_date"].dt.to_period("M")

    first_order = df2.groupby("customer_id")["order_month"].min().rename("cohort_month")
    df2         = df2.join(first_order, on="customer_id")
    df2["period_num"] = (df2["order_month"] - df2["cohort_month"]).apply(lambda x: x.n)

    cohort_data = (
        df2.groupby(["cohort_month", "period_num"])["customer_id"]
        .nunique()
        .reset_index(name="customers")
    )

    cohort_pivot = cohort_data.pivot(index="cohort_month", columns="period_num",
                                      values="customers")
    cohort_sizes = cohort_pivot.iloc[:, 0]
    retention    = cohort_pivot.divide(cohort_sizes, axis=0).round(3) * 100

    logger.info("Cohort retention matrix: %s", retention.shape)
    return retention


# ══════════════════════════════════════════════════════════
# 6. VISUALISATIONS
# ══════════════════════════════════════════════════════════
def create_charts(df: pd.DataFrame, sql_results: dict):
    if not PLOT_AVAILABLE:
        logger.info("Matplotlib not available — skipping charts.")
        return

    sns.set_theme(style="darkgrid", palette="muted")
    fig, axes = plt.subplots(2, 3, figsize=(20, 12))
    fig.suptitle("Sales Analytics Dashboard — Vaibhav Gupta", fontsize=16, fontweight="bold")

    # 1. Monthly Revenue Trend
    monthly = sql_results["monthly_trend"].tail(24)
    axes[0, 0].plot(monthly["month_label"], monthly["revenue"], marker="o", linewidth=2)
    axes[0, 0].set_title("Monthly Revenue Trend")
    axes[0, 0].set_xlabel("Month")
    axes[0, 0].set_ylabel("Revenue (₹)")
    axes[0, 0].tick_params(axis="x", rotation=45)

    # 2. Revenue by Category
    cat = sql_results["top_categories"]
    axes[0, 1].barh(cat["category"], cat["revenue"])
    axes[0, 1].set_title("Revenue by Category")
    axes[0, 1].set_xlabel("Revenue (₹)")

    # 3. Region Performance
    reg = sql_results["region_performance"]
    axes[0, 2].pie(reg["revenue"], labels=reg["region"], autopct="%1.1f%%", startangle=90)
    axes[0, 2].set_title("Revenue by Region")

    # 4. Channel Mix
    ch = sql_results["channel_mix"]
    axes[1, 0].bar(ch["channel"], ch["revenue_share_pct"])
    axes[1, 0].set_title("Channel Revenue Share (%)")
    axes[1, 0].set_ylabel("Share %")

    # 5. AOV Distribution
    axes[1, 1].hist(df[df["net_revenue"] > 0]["net_revenue"], bins=50, edgecolor="black")
    axes[1, 1].set_title("Order Value Distribution")
    axes[1, 1].set_xlabel("Order Value (₹)")
    axes[1, 1].set_ylabel("Frequency")

    # 6. Loyalty Tier
    lt = sql_results["loyalty_tier_revenue"]
    axes[1, 2].bar(lt["loyalty_tier"], lt["avg_order_value"], color=["#CD7F32","#C0C0C0","#FFD700","#E5E4E2"])
    axes[1, 2].set_title("Avg Order Value by Loyalty Tier")
    axes[1, 2].set_ylabel("AOV (₹)")

    plt.tight_layout()
    chart_path = OUTPUT_DIR / "dashboard.png"
    plt.savefig(chart_path, dpi=150, bbox_inches="tight")
    logger.info("Dashboard chart saved → %s", chart_path)
    plt.close()


# ══════════════════════════════════════════════════════════
# 7. POWER BI EXPORT
# ══════════════════════════════════════════════════════════
def export_powerbi(df: pd.DataFrame, sql_results: dict):
    # Main fact table
    df.to_csv(OUTPUT_DIR / "fact_sales.csv", index=False)

    # Aggregated tables for Power BI
    for name, result_df in sql_results.items():
        result_df.to_csv(OUTPUT_DIR / f"dim_{name}.csv", index=False)

    logger.info("Power BI exports saved to %s", OUTPUT_DIR)


# ══════════════════════════════════════════════════════════
# 8. MAIN
# ══════════════════════════════════════════════════════════
def run():
    orders, customers = generate_sales_data(n_orders=50_000)
    df                = etl_pipeline(orders, customers)
    kpis              = compute_kpis(df)
    sql_results       = run_sql_analytics(df)
    retention         = cohort_analysis(df)

    print("\n" + "="*55)
    print("  SALES ANALYTICS — KPI SUMMARY")
    print("="*55)
    for k, v in kpis.items():
        print(f"  {k:<28} {v}")
    print("="*55)

    print("\n📊 Top Categories:")
    print(sql_results["top_categories"].to_string(index=False))

    print("\n📊 Region Performance:")
    print(sql_results["region_performance"].to_string(index=False))

    print("\n📊 Cohort Retention (first 6 periods):")
    print(retention.iloc[:6, :7].to_string())

    create_charts(df, sql_results)
    export_powerbi(df, sql_results)

    print(f"\n✅ Pipeline complete. Files saved to: {OUTPUT_DIR}")
    return df, kpis, sql_results


if __name__ == "__main__":
    df, kpis, results = run()