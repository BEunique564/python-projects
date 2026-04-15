# 📊 Sales Analytics & BI Dashboard
### End-to-End Business Intelligence Pipeline
**Author:** Vaibhav Gupta | AI Engineer @ Hanumant Technology, Lucknow

---

## 🚀 What This Project Does

A complete, production-ready **Business Intelligence pipeline** that:

1. **Generates** realistic e-commerce sales data (50,000 orders, 8,000 customers)
2. **Cleans & transforms** raw data (ETL pipeline)
3. **Calculates KPIs** — Revenue, AOV, LTV, Churn, MoM Growth
4. **Runs SQL analytics** using window functions & CTEs (SQLite in-memory)
5. **Builds cohort retention** analysis month-by-month
6. **Creates visual dashboard** with 6 charts (Matplotlib + Seaborn)
7. **Exports Power BI-ready CSVs** — fact table + dimension tables

---

## 🗂️ Project Structure

```
sales-analytics/
├── main.py              ← Entry point — run this
├── requirements.txt     ← All dependencies
├── README.md            ← You are here
└── outputs/             ← Auto-created on first run
    ├── dashboard.png         ← 6-panel visual dashboard
    ├── fact_sales.csv         ← Main fact table (all 50K orders)
    ├── dim_top_categories.csv
    ├── dim_monthly_trend.csv
    ├── dim_region_performance.csv
    ├── dim_channel_mix.csv
    ├── dim_loyalty_tier_revenue.csv
    └── dim_top_customers.csv
```

---

## ⚙️ Setup & Installation

### Step 1 — Clone / Download

```bash
git clone https://github.com/vaibhavgupta/sales-analytics.git
cd sales-analytics
```

### Step 2 — Create Virtual Environment (Recommended)

```bash
# Windows
python -m venv venv
venv\Scripts\activate

# Mac / Linux
python -m venv venv
source venv/bin/activate
```

### Step 3 — Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 4 — Run the Pipeline

```bash
python main.py
```

That's it. Sab kuch automatically run hoga. ✅

---

## 📦 Requirements

```
numpy>=1.24.0
pandas>=2.0.0
matplotlib>=3.7.0
seaborn>=0.12.0
```

> **Note:** `sqlite3` Python ke saath already aata hai — alag install nahi karna.

---

## 📈 KPIs Generated

Pipeline yeh business metrics calculate karta hai:

| KPI | Description |
|-----|-------------|
| **Total Revenue** | Net revenue after returns |
| **Total Orders** | Count of all transactions |
| **Average Order Value (AOV)** | Revenue / Total Orders |
| **Total Profit** | Net revenue × 40% margin |
| **Return Rate** | % of returned orders |
| **Unique Customers** | Distinct customer count |
| **Customer LTV** | Total Revenue / Unique Customers |
| **MoM Revenue Growth** | Month-over-month % change |

---

## 🗃️ Data Schema

### Orders Table

| Column | Type | Description |
|--------|------|-------------|
| `order_id` | int | Unique order identifier |
| `customer_id` | int | Linked to customer table |
| `order_date` | datetime | Date of purchase |
| `category` | str | Electronics, Fashion, etc. |
| `region` | str | North, South, East, West, Central |
| `channel` | str | Online, Mobile App, In-Store, Partner |
| `payment_method` | str | UPI, Credit Card, COD, etc. |
| `quantity` | int | Units ordered (1–5) |
| `unit_price` | float | Price per unit (lognormal dist) |
| `discount_pct` | int | Discount applied (0–25%) |
| `return_flag` | int | 1 = returned, 0 = kept |
| `revenue` | float | Gross revenue |
| `net_revenue` | float | Revenue after returns |
| `profit` | float | net_revenue × 0.40 |

### Customers Table

| Column | Type | Description |
|--------|------|-------------|
| `customer_id` | int | Unique customer ID |
| `signup_date` | datetime | When customer registered |
| `city` | str | Mumbai, Delhi, Lucknow, etc. |
| `age_group` | str | 18-24, 25-34, 35-44, 45-54, 55+ |
| `loyalty_tier` | str | Bronze, Silver, Gold, Platinum |

---

## 🔍 SQL Analytics (6 Queries)

All queries run on **SQLite in-memory** — no database setup needed.

### 1. Top Categories by Revenue
```sql
SELECT category,
       COUNT(*)                           AS orders,
       ROUND(SUM(net_revenue), 0)         AS revenue,
       ROUND(AVG(unit_price), 2)          AS avg_price,
       ROUND(AVG(discount_pct), 1)        AS avg_discount
FROM sales
GROUP BY category
ORDER BY revenue DESC
```

### 2. Monthly Revenue Trend
```sql
SELECT month_label,
       COUNT(*)                   AS orders,
       ROUND(SUM(net_revenue), 0) AS revenue,
       ROUND(AVG(net_revenue), 2) AS aov
FROM sales
GROUP BY month_label
ORDER BY month_label
```

### 3. Channel Mix with Window Function
```sql
SELECT channel,
       COUNT(*)                                              AS orders,
       ROUND(SUM(net_revenue), 0)                           AS revenue,
       ROUND(SUM(net_revenue) * 100.0 /
             SUM(SUM(net_revenue)) OVER(), 2)               AS revenue_share_pct
FROM sales
GROUP BY channel
ORDER BY revenue DESC
```

> **Window Function used:** `SUM() OVER()` for percentage of total — real-world SQL trick!

---

## 📊 Dashboard Charts

Running the script creates `outputs/dashboard.png` with 6 panels:

| Panel | Chart Type | What it shows |
|-------|-----------|---------------|
| Top-Left | Line chart | Monthly revenue trend (24 months) |
| Top-Center | Horizontal bar | Revenue by product category |
| Top-Right | Pie chart | Regional revenue distribution |
| Bottom-Left | Bar chart | Channel revenue share % |
| Bottom-Center | Histogram | Order value distribution |
| Bottom-Right | Bar chart | AOV by loyalty tier |

---

## 🔄 Cohort Retention Analysis

Month 0 = first purchase month. Shows what % of customers returned in subsequent months.

```
period_num     0       1       2       3       4       5
cohort_month
2023-01      100%    38.2%   22.1%   15.4%   11.2%   8.9%
2023-02      100%    37.8%   21.5%   14.9%   10.8%   ...
...
```

This is the same analysis used by **Flipkart, Amazon, Swiggy** to track customer loyalty.

---

## 📤 Power BI Integration

### Step 1 — Run the pipeline
```bash
python main.py
```

### Step 2 — Open Power BI Desktop
- Click **Get Data → Text/CSV**
- Import `outputs/fact_sales.csv` as the main fact table
- Import each `dim_*.csv` as dimension tables

### Step 3 — Create Relationships
```
fact_sales[customer_id] → dim_loyalty_tier_revenue[customer_id]
fact_sales[channel]     → dim_channel_mix[channel]
fact_sales[category]    → dim_top_categories[category]
```

### Step 4 — Build Your Dashboard
All KPIs and aggregations are pre-calculated — drag and drop to create visuals!

---

## 🛠️ Customization

### Change Data Size
```python
# In main.py → run() function
orders, customers = generate_sales_data(n_orders=100_000)  # Increase to 100K
```

### Change Date Range
```python
orders, customers = generate_sales_data(
    n_orders=50_000,
    start_date="2022-01-01"  # Start from 2022
)
```

### Use Real Data Instead
Replace `generate_sales_data()` with your own CSV:
```python
orders    = pd.read_csv("your_orders.csv")
customers = pd.read_csv("your_customers.csv")
df        = etl_pipeline(orders, customers)
```

### Change Profit Margin
```python
# In etl_pipeline() — line: df["profit"] = ...
df["profit"] = (df["net_revenue"] * 0.35).round(2)  # Change to 35% margin
```

---

## 🐛 Troubleshooting

| Error | Fix |
|-------|-----|
| `ModuleNotFoundError: pandas` | Run `pip install -r requirements.txt` |
| `ModuleNotFoundError: matplotlib` | Run `pip install matplotlib seaborn` |
| Charts not saving | Check write permission in project folder |
| Empty outputs folder | Make sure `python main.py` completed fully |
| Slow on low RAM | Reduce `n_orders` to 10,000 |

---

## 💡 Concepts Used (For Students)

This project covers real-world industry skills:

- **Pandas** — Data manipulation, merge, groupby, datetime features
- **NumPy** — Random data generation, vectorized operations
- **SQLite** — In-memory database, SQL window functions, CTEs
- **Matplotlib & Seaborn** — Multi-panel dashboards, chart customization
- **ETL Pipeline** — Extract → Transform → Load pattern
- **Cohort Analysis** — Period-over-period retention (pivot tables)
- **Business KPIs** — AOV, LTV, MoM growth, churn rate
- **Power BI Export** — Star schema fact + dimension tables

---

## 🎓 Learning Path

If you're a beginner — yeh project samajhne ke liye pehle yeh seekho:

1. **Week 1-2:** Python basics (loops, functions, lists)
2. **Week 3-4:** Pandas — read_csv, groupby, merge
3. **Week 5-6:** SQL — SELECT, GROUP BY, JOIN
4. **Week 7-8:** Matplotlib — basic charts
5. **Week 9:** Run this project — sab kuch click karega! ✅

---

## 📬 Connect

**Vaibhav Gupta** — Data Science Mentor & AI Engineer

- 🌐 Website: [vgmentorship.com](https://vgmentorship.com)
- 💼 LinkedIn: [linkedin.com/in/vaibhavgupta](https://linkedin.com/in/vaibhavgupta)
- 📍 Lucknow, Uttar Pradesh, India
- 📧 vaibhav@vgmentorship.com

> *"Coding talent se nahi, consistency se aati hai."*
> — Vaibhav Gupta

---

## 📄 License

MIT License — Free to use, modify, and share with attribution.

---

*Made with ❤️ in Lucknow 🏙️*
