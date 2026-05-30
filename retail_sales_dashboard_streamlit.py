from __future__ import annotations

# -*- coding: utf-8 -*-
"""
Retail Weekly Sales Forecasting Dashboard - Streamlit V6 No Top Gap

Business scenario:
- Weekly retail sales forecasting
- Inventory/restocking support
- Store and department performance monitoring
- Holiday and markdown/promotion effectiveness analysis
- Model evaluation and explainability for non-technical stakeholders

Required files in the same folder as this script:
1. scenario3_cleaned.csv
2. model_test_predictions.csv
3. random_forest_feature_importance.csv
4. comprehensive_evaluation_table.csv
5. worst_50_predictions_random_forest.csv

Run:
    pip install -r requirements_streamlit_dashboard_v6.txt
    streamlit run retail_sales_dashboard_streamlit_v6_no_top_gap.py
"""

from pathlib import Path
from html import escape
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# ============================================================
# Page settings
# ============================================================
st.set_page_config(
    page_title="Retail Sales Forecasting Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Color-blind friendly palette
BLUE = "#0072B2"
ORANGE = "#E69F00"
GREEN = "#009E73"
RED = "#D55E00"
PURPLE = "#7C3AED"
GRAY = "#6B7280"
TEXT = "#111827"
MUTED = "#4B5563"
BORDER = "#E5E7EB"
BG = "#F8FAFC"
CARD = "#FFFFFF"

# Force a readable light dashboard even when Streamlit/browser is in dark mode.
st.markdown(
    """
    <style>
    .stApp {
        background: #F8FAFC !important;
        color: #111827 !important;
    }

    /* Remove Streamlit default top toolbar/header that can appear as a dark gap. */
    [data-testid="stHeader"],
    header[data-testid="stHeader"],
    .stApp > header {
        background: rgba(0,0,0,0) !important;
        height: 0rem !important;
        min-height: 0rem !important;
        display: none !important;
        visibility: hidden !important;
    }
    [data-testid="stToolbar"],
    [data-testid="stDecoration"],
    #MainMenu,
    footer {
        display: none !important;
        visibility: hidden !important;
        height: 0rem !important;
    }
    [data-testid="stAppViewContainer"] {
        background: #F8FAFC !important;
    }
    html, body, [class*="css"], [data-testid="stMarkdownContainer"] {
        font-family: Inter, "Segoe UI", Arial, sans-serif !important;
    }
    .block-container {
        padding-top: 0.75rem !important;
        padding-bottom: 2rem !important;
        max-width: 1500px;
    }
    [data-testid="stSidebar"] {
        background: #FFFFFF !important;
        border-right: 1px solid #E5E7EB !important;
    }
    [data-testid="stSidebar"] * {
        color: #111827 !important;
    }
    /* Streamlit can inherit the browser/app dark theme for form controls.
       Force sidebar controls to a light readable style. */
    [data-testid="stSidebar"] div[data-baseweb="select"] > div,
    [data-testid="stSidebar"] div[data-baseweb="base-input"] > div,
    [data-testid="stSidebar"] input,
    [data-testid="stSidebar"] textarea {
        background-color: #FFFFFF !important;
        color: #111827 !important;
        border: 1px solid #CBD5E1 !important;
        border-radius: 10px !important;
    }
    [data-testid="stSidebar"] div[data-baseweb="select"] span,
    [data-testid="stSidebar"] div[data-baseweb="select"] div,
    [data-testid="stSidebar"] div[data-baseweb="select"] input,
    [data-testid="stSidebar"] div[data-baseweb="base-input"] input,
    [data-testid="stSidebar"] div[data-baseweb="base-input"] div {
        color: #111827 !important;
        -webkit-text-fill-color: #111827 !important;
    }
    [data-testid="stSidebar"] div[data-baseweb="select"] svg,
    [data-testid="stSidebar"] svg {
        fill: #111827 !important;
        color: #111827 !important;
    }
    [data-testid="stSidebar"] [data-baseweb="tag"] {
        background-color: #E0F2FE !important;
        color: #111827 !important;
        border-radius: 999px !important;
    }
    [data-testid="stSidebar"] [data-baseweb="tag"] span {
        color: #111827 !important;
        -webkit-text-fill-color: #111827 !important;
    }
    [data-testid="stSidebar"] .stDateInput input {
        background-color: #FFFFFF !important;
        color: #111827 !important;
        -webkit-text-fill-color: #111827 !important;
    }
    [data-testid="stSidebar"] .stSlider * {
        color: #111827 !important;
    }
    h1, h2, h3, h4, h5, h6, p, span, label, div {
        color: inherit;
    }
    .main-title {
        font-size: 2.2rem;
        font-weight: 900;
        letter-spacing: -0.03em;
        color: #111827 !important;
        margin-bottom: 0.15rem;
    }
    .subtitle {
        color: #4B5563 !important;
        font-size: 1.02rem;
        margin-bottom: 1rem;
        line-height: 1.55;
    }
    .small-note {
        color: #6B7280 !important;
        font-size: 0.88rem;
        margin-top: -0.35rem;
        margin-bottom: 0.7rem;
        line-height: 1.45;
    }
    .soft-divider {
        height: 1px;
        background: #E5E7EB;
        margin: 1.2rem 0;
    }
    div[data-testid="stMetric"] {
        background: #FFFFFF !important;
        border: 1px solid #E5E7EB !important;
        border-radius: 18px !important;
        padding: 1rem 1.1rem !important;
        box-shadow: 0 6px 20px rgba(15, 23, 42, 0.06) !important;
    }
    div[data-testid="stMetric"] * {
        color: #111827 !important;
    }
    div[data-testid="stMetricLabel"] {
        color: #4B5563 !important;
        font-weight: 750 !important;
    }
    div[data-testid="stMetricValue"] {
        color: #111827 !important;
        font-weight: 900 !important;
    }
    .card {
        background: #FFFFFF !important;
        color: #111827 !important;
        border: 1px solid #E5E7EB;
        border-radius: 18px;
        padding: 1rem 1.1rem;
        box-shadow: 0 6px 20px rgba(15, 23, 42, 0.06);
        min-height: 92px;
        margin-bottom: 0.75rem;
    }
    .card * { color: #111827 !important; }
    .card-title {
        font-weight: 850;
        font-size: 0.98rem;
        margin-bottom: 0.35rem;
    }
    .card-body {
        font-size: 0.92rem;
        color: #374151 !important;
        line-height: 1.45;
    }
    .tag {
        display: inline-block;
        border-radius: 999px;
        padding: 0.16rem 0.58rem;
        font-size: 0.72rem;
        font-weight: 800;
        margin-left: 0.35rem;
        white-space: nowrap;
        color: #111827 !important;
        background: #EEF2FF;
    }
    .alert-green { border-left: 8px solid #009E73; background: #F0FDF4 !important; }
    .alert-yellow { border-left: 8px solid #E69F00; background: #FFFBEB !important; }
    .alert-red { border-left: 8px solid #D55E00; background: #FFF7ED !important; }
    .scenario-ok { border-left: 8px solid #009E73; }
    .scenario-watch { border-left: 8px solid #E69F00; }
    .scenario-missing { border-left: 8px solid #D55E00; }
    .stDataFrame, [data-testid="stDataFrame"] {
        background: #FFFFFF !important;
        color: #111827 !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ============================================================
# Paths and model map
# ============================================================
BASE_DIR = Path(__file__).resolve().parent if "__file__" in globals() else Path.cwd()

DATA_FILES = {
    "Cleaned sales data": BASE_DIR / "scenario3_cleaned.csv",
    "Prediction data": BASE_DIR / "model_test_predictions.csv",
    "Feature importance": BASE_DIR / "random_forest_feature_importance.csv",
    "Model evaluation": BASE_DIR / "comprehensive_evaluation_table.csv",
    "Worst prediction errors": BASE_DIR / "worst_50_predictions_random_forest.csv",
}

MODEL_TO_COLUMN = {
    "Model 2 - Tuned Random Forest": "Random_Forest_Prediction",
    "Model 1 - Linear Regression": "Linear_Regression_Prediction",
    "Naive Baseline": "Naive_Prediction",
}

ACTUAL_COL = "Weekly_Sales_NonNegative"

# ============================================================
# Data loading and utilities
# ============================================================
@st.cache_data(show_spinner=False)
def load_csv(path: str) -> pd.DataFrame:
    file_path = Path(path)
    if not file_path.exists():
        return pd.DataFrame()
    df = pd.read_csv(file_path)
    if "Date" in df.columns:
        df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    return df


def add_store_size_category(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty or "Store_Size_Category" in df.columns:
        return df
    if {"Store", "Size"}.issubset(df.columns):
        out = df.copy()
        store_size = out[["Store", "Size"]].drop_duplicates().sort_values("Size")
        store_size["_rank"] = store_size["Size"].rank(method="first")
        try:
            store_size["Store_Size_Category"] = pd.qcut(
                store_size["_rank"], q=3, labels=["Small", "Medium", "Large"]
            ).astype(str)
        except ValueError:
            store_size["Store_Size_Category"] = "Unknown"
        mapper = store_size.set_index("Store")["Store_Size_Category"].to_dict()
        out["Store_Size_Category"] = out["Store"].map(mapper).fillna("Unknown")
        return out
    return df


def month_to_season(month: object) -> str:
    if pd.isna(month):
        return "Unknown"
    m = int(month)
    if m in (12, 1, 2):
        return "Winter"
    if m in (3, 4, 5):
        return "Spring"
    if m in (6, 7, 8):
        return "Summer"
    if m in (9, 10, 11):
        return "Autumn"
    return "Unknown"


def enrich_predictions(pred: pd.DataFrame, sales: pd.DataFrame) -> pd.DataFrame:
    if pred.empty:
        return pred
    out = pred.copy()
    if "Date" in out.columns:
        out["Date"] = pd.to_datetime(out["Date"], errors="coerce")
    out = add_store_size_category(out)

    if "Season" not in out.columns and "Month" in out.columns:
        out["Season"] = out["Month"].apply(month_to_season)

    # Bring promotion / holiday-period fields from cleaned data into the prediction table.
    merge_keys = ["Store", "Dept", "Date"]
    enrich_cols = [
        "Season", "Any_MarkDown", "Total_MarkDown", "MarkDown_Intensity",
        "Before_Holiday", "After_Holiday", "Holiday_Period", "Size",
    ]
    if not sales.empty and set(merge_keys).issubset(sales.columns):
        available = [c for c in enrich_cols if c in sales.columns and c not in out.columns]
        if available:
            small = sales[merge_keys + available].drop_duplicates(merge_keys).copy()
            small["Date"] = pd.to_datetime(small["Date"], errors="coerce")
            out = out.merge(small, on=merge_keys, how="left")

    if "Any_MarkDown" not in out.columns:
        out["Any_MarkDown"] = 0
    if "Total_MarkDown" not in out.columns:
        out["Total_MarkDown"] = 0.0
    if "Season" not in out.columns and "Month" in out.columns:
        out["Season"] = out["Month"].apply(month_to_season)
    return out


def money(value: float, unit: str = "auto") -> str:
    if pd.isna(value):
        return "N/A"
    value = float(value)
    if unit == "million":
        return f"US${value / 1_000_000:,.2f}M"
    if unit == "thousand":
        return f"US${value / 1_000:,.1f}K"
    if abs(value) >= 1_000_000:
        return f"US${value / 1_000_000:,.2f}M"
    if abs(value) >= 1_000:
        return f"US${value / 1_000:,.1f}K"
    return f"US${value:,.0f}"


def pct(value: float, decimals: int = 1) -> str:
    if pd.isna(value):
        return "N/A"
    return f"{float(value) * 100:.{decimals}f}%"


def safe_num(value: float, decimals: int = 0) -> str:
    if pd.isna(value):
        return "N/A"
    return f"{float(value):,.{decimals}f}"


def safe_r2(y_true: pd.Series, y_pred: pd.Series) -> float:
    valid = pd.DataFrame({"actual": y_true, "pred": y_pred}).replace([np.inf, -np.inf], np.nan).dropna()
    if len(valid) < 2:
        return np.nan
    ss_res = ((valid["actual"] - valid["pred"]) ** 2).sum()
    ss_tot = ((valid["actual"] - valid["actual"].mean()) ** 2).sum()
    if ss_tot == 0:
        return np.nan
    return 1 - ss_res / ss_tot


def clean_feature_name(name: object) -> str:
    text = str(name)
    for prefix in ("num__", "cat__", "remainder__"):
        text = text.replace(prefix, "")
    replacements = {
        "Lag_1_Week_Sales": "Previous Week Sales",
        "Lag_4_Week_Sales": "Sales 4 Weeks Ago",
        "Rolling_4_Week_Mean_Sales": "4-Week Rolling Average Sales",
        "Rolling_4_Week_Avg_Sales": "4-Week Rolling Average Sales",
        "Rolling_12_Week_Avg_Sales": "12-Week Rolling Average Sales",
        "Total_MarkDown": "Total Markdown",
        "CPI": "Consumer Price Index",
        "WeekOfYear": "Week Of Year",
        "Holiday_Week": "Holiday Week",
    }
    for old, new in replacements.items():
        text = text.replace(old, new)
    return text.replace("_", " ").title()


def chart_style(fig: go.Figure, height: int = 430) -> go.Figure:
    fig.update_layout(
        template="plotly_white",
        height=height,
        paper_bgcolor="#FFFFFF",
        plot_bgcolor="#FFFFFF",
        font=dict(color=TEXT, family="Inter, Segoe UI, Arial, sans-serif", size=13),
        title=dict(font=dict(color=TEXT, size=18), x=0.01),
        margin=dict(l=25, r=25, t=58, b=35),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
    )
    fig.update_xaxes(gridcolor="#E5E7EB", zerolinecolor="#E5E7EB", color=TEXT, title_font=dict(color=TEXT))
    fig.update_yaxes(gridcolor="#E5E7EB", zerolinecolor="#E5E7EB", color=TEXT, title_font=dict(color=TEXT))
    return fig


def card(title: str, body: str, tag: str | None = None, status: str | None = None) -> None:
    cls = "card"
    if status == "green":
        cls += " alert-green"
    elif status == "yellow":
        cls += " alert-yellow"
    elif status == "red":
        cls += " alert-red"
    tag_html = f"<span class='tag'>{escape(tag)}</span>" if tag else ""
    st.markdown(
        f"""
        <div class="{cls}" style="color:#111827 !important;">
            <div class="card-title" style="color:#111827 !important;">{escape(title)} {tag_html}</div>
            <div class="card-body" style="color:#374151 !important;">{body}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def scenario_card(title: str, body: str, ok: bool, partial: bool = False) -> None:
    cls = "scenario-ok" if ok else ("scenario-watch" if partial else "scenario-missing")
    status = "Supported" if ok else ("Partly supported" if partial else "Needs more data")
    st.markdown(
        f"""
        <div class="card {cls}" style="color:#111827 !important; min-height:125px;">
            <div class="card-title" style="color:#111827 !important;">{escape(title)} <span class='tag'>{status}</span></div>
            <div class="card-body" style="color:#374151 !important;">{body}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def apply_filters(
    df: pd.DataFrame,
    start_date,
    end_date,
    stores: list,
    depts: list,
    store_types: list,
    size_categories: list,
    seasons: list,
    holiday_filter: str,
    promo_filter: str,
) -> pd.DataFrame:
    if df.empty:
        return df
    out = df.copy()
    if "Date" in out.columns:
        out = out[(out["Date"] >= pd.to_datetime(start_date)) & (out["Date"] <= pd.to_datetime(end_date))]
    if stores and "Store" in out.columns:
        out = out[out["Store"].isin(stores)]
    if depts and "Dept" in out.columns:
        out = out[out["Dept"].isin(depts)]
    if store_types and "Type" in out.columns:
        out = out[out["Type"].isin(store_types)]
    if size_categories and "Store_Size_Category" in out.columns:
        out = out[out["Store_Size_Category"].isin(size_categories)]
    if seasons and "Season" in out.columns:
        out = out[out["Season"].isin(seasons)]
    if holiday_filter != "All" and "Holiday_Week" in out.columns:
        holiday_numeric = pd.to_numeric(out["Holiday_Week"], errors="coerce").fillna(0).astype(int)
        if holiday_filter == "Holiday weeks only":
            out = out[holiday_numeric == 1]
        elif holiday_filter == "Non-holiday weeks only":
            out = out[holiday_numeric == 0]
    if promo_filter != "All" and "Any_MarkDown" in out.columns:
        promo_numeric = pd.to_numeric(out["Any_MarkDown"], errors="coerce").fillna(0).astype(int)
        if promo_filter == "Markdown active":
            out = out[promo_numeric == 1]
        elif promo_filter == "No markdown":
            out = out[promo_numeric == 0]
    return out


def add_forecast_metrics(df: pd.DataFrame, pred_col: str, acceptable_pct: float) -> pd.DataFrame:
    out = df.copy()
    if out.empty or ACTUAL_COL not in out.columns or pred_col not in out.columns:
        return out
    out["Selected_Prediction"] = pd.to_numeric(out[pred_col], errors="coerce")
    out[ACTUAL_COL] = pd.to_numeric(out[ACTUAL_COL], errors="coerce")
    out["Prediction_Error"] = out["Selected_Prediction"] - out[ACTUAL_COL]
    out["Absolute_Error"] = out["Prediction_Error"].abs()

    positive_actual = out.loc[out[ACTUAL_COL] > 0, ACTUAL_COL]
    median_actual = positive_actual.median() if not positive_actual.empty else 1000.0
    denominator_floor = max(1000.0, float(median_actual) * 0.10)
    out["Business_Error_Denominator"] = np.maximum(out[ACTUAL_COL].fillna(0), denominator_floor)
    out["Business_APE"] = out["Absolute_Error"] / out["Business_Error_Denominator"]
    out["Raw_APE"] = np.where(out[ACTUAL_COL] > 0, out["Absolute_Error"] / out[ACTUAL_COL], np.nan)

    threshold = acceptable_pct / 100.0
    out["Forecast_Status"] = np.select(
        [out["Business_APE"] <= threshold, out["Business_APE"] <= threshold * 2],
        ["Good", "Watch"],
        default="Risk",
    )

    group_cols = [c for c in ["Store", "Dept"] if c in out.columns]
    if group_cols:
        group_mean = out.groupby(group_cols)["Selected_Prediction"].transform("mean")
        group_std = out.groupby(group_cols)["Selected_Prediction"].transform("std").fillna(0)
        out["High_Demand_Alert"] = out["Selected_Prediction"] > (group_mean + 0.75 * group_std)
    else:
        out["High_Demand_Alert"] = out["Selected_Prediction"] > out["Selected_Prediction"].quantile(0.85)

    # Business risk: large relative error plus a meaningful absolute error.
    abs_floor = max(1000.0, float(median_actual) * 0.05)
    out["Business_Risk_Record"] = (out["Business_APE"] > threshold * 2) & (out["Absolute_Error"] > abs_floor)
    out["Actual_Below_Forecast"] = out["Prediction_Error"] > abs_floor
    out["Actual_Above_Forecast"] = out["Prediction_Error"] < -abs_floor
    return out


def get_model_options(evaluation: pd.DataFrame, predictions: pd.DataFrame) -> list[tuple[str, str]]:
    available = [(model, col) for model, col in MODEL_TO_COLUMN.items() if col in predictions.columns]
    if not available:
        return []
    if not evaluation.empty and {"Model", "MAE"}.issubset(evaluation.columns):
        mae_map = evaluation.set_index("Model")["MAE"].to_dict()
        available = sorted(available, key=lambda x: mae_map.get(x[0], np.inf))
    else:
        priority = {"Model 2 - Tuned Random Forest": 0, "Model 1 - Linear Regression": 1, "Naive Baseline": 2}
        available = sorted(available, key=lambda x: priority.get(x[0], 99))
    return available


def build_recommendations(df: pd.DataFrame, sales_view: pd.DataFrame, threshold_pct: float) -> list[tuple[str, str, str, str]]:
    recs: list[tuple[str, str, str, str]] = []
    if df.empty or ACTUAL_COL not in df.columns:
        return [("No data", "No records match the current filters. Widen the date range or clear some filters.", "Data", "yellow")]

    total_actual = df[ACTUAL_COL].sum()
    wape = df["Absolute_Error"].sum() / total_actual if total_actual > 0 else np.nan
    bias = df["Prediction_Error"].mean()
    threshold = threshold_pct / 100.0

    if {"Store", "Dept"}.issubset(df.columns):
        demand = (
            df.groupby(["Store", "Dept"], as_index=False)
            .agg(
                Forecasted_Sales=("Selected_Prediction", "sum"),
                Actual_Sales=(ACTUAL_COL, "sum"),
                Demand_Spike_Weeks=("High_Demand_Alert", "sum"),
                Weeks=("Date", "nunique"),
            )
            .sort_values(["Forecasted_Sales", "Demand_Spike_Weeks"], ascending=False)
            .head(3)
        )
        top_items = "; ".join(
            f"Store {int(r.Store)} / Dept {int(r.Dept)} ({money(r.Forecasted_Sales)})"
            for _, r in demand.iterrows()
        )
        recs.append((
            "Inventory priority",
            f"Increase replenishment attention for {escape(top_items)}. These Store-Department combinations have the highest forecasted demand under the current filters.",
            "Restocking",
            "yellow",
        ))

        underperform = (
            df.groupby(["Store", "Dept"], as_index=False)
            .agg(
                Actual_Sales=(ACTUAL_COL, "sum"),
                Forecasted_Sales=("Selected_Prediction", "sum"),
                Abs_Error=("Absolute_Error", "sum"),
            )
        )
        underperform["Gap"] = underperform["Actual_Sales"] - underperform["Forecasted_Sales"]
        underperform["WAPE"] = np.where(underperform["Actual_Sales"] > 0, underperform["Abs_Error"] / underperform["Actual_Sales"], np.nan)
        weak = underperform.sort_values("Gap", ascending=True).head(3)
        weak_items = "; ".join(
            f"Store {int(r.Store)} / Dept {int(r.Dept)} ({money(abs(r.Gap))} below forecast)"
            for _, r in weak.iterrows()
        )
        recs.append((
            "Performance investigation",
            f"Review {escape(weak_items)}. Actual sales are below forecast, so managers should check local demand, shelf availability, promotion execution, or economic pressure.",
            "Underperformance",
            "red" if len(weak) else "yellow",
        ))

    if pd.notna(wape):
        if wape <= threshold:
            recs.append((
                "Forecast credibility",
                f"Weighted forecast deviation is {pct(wape)}, within the selected {threshold_pct:.0f}% tolerance. Use the forecast for routine planning, but still check high-risk groups.",
                "Model Monitoring",
                "green",
            ))
        elif wape <= threshold * 2:
            recs.append((
                "Forecast watch",
                f"Weighted forecast deviation is {pct(wape)}, above the target but not extreme. Use forecasts as a planning signal rather than an automatic ordering rule.",
                "Model Monitoring",
                "yellow",
            ))
        else:
            recs.append((
                "Forecast risk",
                f"Weighted forecast deviation is {pct(wape)}, which is high for operational ordering. Investigate errors before using this slice for inventory decisions.",
                "Model Risk",
                "red",
            ))

    if pd.notna(bias):
        if bias > 0:
            recs.append((
                "Bias control",
                f"The model over-forecasts by {money(bias)} per Store-Dept week on average. Avoid over-ordering unless recent sales momentum confirms the demand increase.",
                "Bias",
                "yellow",
            ))
        elif bias < 0:
            recs.append((
                "Stockout watch",
                f"The model under-forecasts by {money(abs(bias))} per Store-Dept week on average. Monitor fast-moving departments to reduce stockout risk.",
                "Stockout Risk",
                "yellow",
            ))

    if "Holiday_Week" in df.columns and df["Holiday_Week"].nunique(dropna=True) > 1:
        h = df.groupby("Holiday_Week")["Selected_Prediction"].mean()
        if 0 in h.index and 1 in h.index and h.loc[0] > 0:
            uplift = h.loc[1] / h.loc[0] - 1
            direction = "higher" if uplift >= 0 else "lower"
            recs.append((
                "Holiday planning",
                f"Holiday weeks are forecast to be {pct(abs(uplift))} {direction} than non-holiday weeks. Use this signal for holiday replenishment and staffing plans.",
                "Holiday",
                "yellow" if abs(uplift) >= 0.03 else "green",
            ))

    if not sales_view.empty and {"Any_MarkDown", ACTUAL_COL}.issubset(sales_view.columns):
        promo = sales_view.groupby("Any_MarkDown")[ACTUAL_COL].mean()
        if 0 in promo.index and 1 in promo.index and promo.loc[0] > 0:
            lift = promo.loc[1] / promo.loc[0] - 1
            recs.append((
                "Promotion check",
                f"Markdown weeks show an average sales lift of {pct(lift)} in the selected historical data. Use this to evaluate whether promotion intensity is producing enough demand response.",
                "Promotion",
                "green" if lift > 0.05 else "yellow",
            ))

    # Keep the panel readable.
    return recs[:6]

# ============================================================
# Load data
# ============================================================
with st.spinner("Loading dashboard files..."):
    sales = load_csv(str(DATA_FILES["Cleaned sales data"]))
    predictions = load_csv(str(DATA_FILES["Prediction data"]))
    feature_importance = load_csv(str(DATA_FILES["Feature importance"]))
    evaluation = load_csv(str(DATA_FILES["Model evaluation"]))
    worst_predictions = load_csv(str(DATA_FILES["Worst prediction errors"]))

sales = add_store_size_category(sales)
predictions = enrich_predictions(predictions, sales)

if predictions.empty:
    st.error("Prediction file is missing. Please place `model_test_predictions(2).csv` in the same folder as this script.")
    st.stop()
if ACTUAL_COL not in predictions.columns:
    st.error(f"Prediction data must contain `{ACTUAL_COL}`.")
    st.stop()

# ============================================================
# Header
# ============================================================
st.markdown('<div class="main-title">Retail Weekly Sales Forecasting Dashboard</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="subtitle">For store managers, supply chain managers, marketing managers, regional supervisors, and finance teams. '
    'The dashboard converts weekly sales forecasts into inventory, performance, promotion, and holiday planning actions.</div>',
    unsafe_allow_html=True,
)

# ============================================================
# Sidebar filters
# ============================================================
st.sidebar.header("Filters")
st.sidebar.caption("Drill down: whole business → store type → store → department → week.")

with st.sidebar.expander("Uploaded file check", expanded=False):
    for label, path in DATA_FILES.items():
        status = "✅ Found" if path.exists() else "❌ Missing"
        st.write(f"{status} **{label}**")
        st.caption(path.name)

model_options = get_model_options(evaluation, predictions)
if not model_options:
    st.error("No usable prediction column found. Expected Random_Forest_Prediction, Linear_Regression_Prediction, or Naive_Prediction.")
    st.stop()

model_labels = [m[0] for m in model_options]
model_label = st.sidebar.selectbox(
    "Forecasting model",
    options=model_labels,
    index=0,
    help="The default is selected from the lowest MAE in the uploaded evaluation table.",
)
pred_col = dict(model_options)[model_label]

min_date = predictions["Date"].min()
max_date = predictions["Date"].max()
selected_date_range = st.sidebar.date_input(
    "Date range",
    value=(min_date.date(), max_date.date()),
    min_value=min_date.date(),
    max_value=max_date.date(),
)
if isinstance(selected_date_range, tuple) and len(selected_date_range) == 2:
    start_date, end_date = selected_date_range
else:
    start_date, end_date = min_date.date(), max_date.date()

stores = sorted(predictions["Store"].dropna().unique().tolist()) if "Store" in predictions.columns else []
depts = sorted(predictions["Dept"].dropna().unique().tolist()) if "Dept" in predictions.columns else []
types = sorted(predictions["Type"].dropna().unique().tolist()) if "Type" in predictions.columns else []
sizes = sorted(predictions["Store_Size_Category"].dropna().unique().tolist()) if "Store_Size_Category" in predictions.columns else []
seasons = sorted(predictions["Season"].dropna().unique().tolist()) if "Season" in predictions.columns else []

selected_stores = st.sidebar.multiselect("Store", stores, default=[])
selected_depts = st.sidebar.multiselect("Department", depts, default=[])
selected_types = st.sidebar.multiselect("Store type", types, default=[])
selected_sizes = st.sidebar.multiselect("Store size category", sizes, default=[])
selected_seasons = st.sidebar.multiselect("Season", seasons, default=[])
holiday_filter = st.sidebar.selectbox("Holiday filter", ["All", "Holiday weeks only", "Non-holiday weeks only"])
promo_filter = st.sidebar.selectbox("Markdown / promotion filter", ["All", "Markdown active", "No markdown"])
acceptable_error_pct = st.sidebar.slider(
    "Acceptable forecast deviation threshold",
    min_value=5,
    max_value=60,
    value=30,
    step=5,
    help="Used for business status labels and risk alerts. WAPE is preferred over raw MAPE because many low-sales records distort percentage errors.",
)
top_n = st.sidebar.slider("Top N in ranking charts", 5, 25, 10, 1)

st.sidebar.markdown("---")
st.sidebar.subheader("Presentation display")
presentation_mode = st.sidebar.checkbox(
    "Use presentation-friendly main chart",
    value=True,
    help=(
        "Keeps the raw model metrics unchanged, but changes the main visual to a fairer business view: "
        "average per Store-Dept week rather than total company weekly sales."
    ),
)
main_chart_level = st.sidebar.selectbox(
    "Main chart level",
    options=[
        "Average weekly sales per Store-Dept",
        "Monthly total sales",
        "Weekly total sales",
    ],
    index=0,
    help=(
        "Weekly totals add thousands of prediction rows together, so small per-row errors look large. "
        "Average weekly sales per Store-Dept is usually better for presentations."
    ),
)
bias_adjust_display = st.sidebar.checkbox(
    "Show bias-adjusted forecast line",
    value=True,
    help=(
        "Presentation-only calibration: subtracts the selected model's average historical bias from the line. "
        "Do not report this as the raw model output. Raw metrics remain available in the dashboard."
    ),
)
main_axis_zero = st.sidebar.checkbox(
    "Start main sales axis at zero",
    value=True,
    help="Recommended for business users. It prevents the visual distance between actual and forecast from being exaggerated.",
)

# ============================================================
# Filter data and compute metrics
# ============================================================
filtered_pred = apply_filters(
    predictions,
    start_date,
    end_date,
    selected_stores,
    selected_depts,
    selected_types,
    selected_sizes,
    selected_seasons,
    holiday_filter,
    promo_filter,
)
filtered_sales = apply_filters(
    sales,
    start_date,
    end_date,
    selected_stores,
    selected_depts,
    selected_types,
    selected_sizes,
    selected_seasons,
    holiday_filter,
    promo_filter,
)
filtered_pred = add_forecast_metrics(filtered_pred, pred_col, acceptable_error_pct)

if filtered_pred.empty:
    st.warning("No records match the current filters. Please widen the date range or clear some filters.")
    st.stop()

# Raw model metrics. These are not changed by presentation display options.
total_actual = filtered_pred[ACTUAL_COL].sum()
total_forecast = filtered_pred["Selected_Prediction"].sum()
mae = filtered_pred["Absolute_Error"].mean()
rmse = float(np.sqrt(np.mean(filtered_pred["Prediction_Error"] ** 2)))
r2 = safe_r2(filtered_pred[ACTUAL_COL], filtered_pred["Selected_Prediction"])
wape = filtered_pred["Absolute_Error"].sum() / total_actual if total_actual > 0 else np.nan
bias = filtered_pred["Prediction_Error"].mean()
risk_count = int(filtered_pred["Business_Risk_Record"].sum()) if "Business_Risk_Record" in filtered_pred.columns else 0
watch_count = int((filtered_pred["Forecast_Status"] == "Watch").sum()) if "Forecast_Status" in filtered_pred.columns else 0
high_demand_count = int(filtered_pred["High_Demand_Alert"].sum()) if "High_Demand_Alert" in filtered_pred.columns else 0

# Presentation-only display forecast for the main line chart.
# This does not overwrite Selected_Prediction and does not change raw MAE/WAPE/R2.
filtered_pred["Display_Prediction"] = filtered_pred["Selected_Prediction"]
display_bias_correction = 0.0
if presentation_mode and bias_adjust_display and pd.notna(bias):
    display_bias_correction = float(bias)
    filtered_pred["Display_Prediction"] = (filtered_pred["Selected_Prediction"] - display_bias_correction).clip(lower=0)
filtered_pred["Display_Error"] = filtered_pred["Display_Prediction"] - filtered_pred[ACTUAL_COL]
filtered_pred["Display_Absolute_Error"] = filtered_pred["Display_Error"].abs()
display_wape = filtered_pred["Display_Absolute_Error"].sum() / total_actual if total_actual > 0 else np.nan

# ============================================================
# Scenario coverage check
# ============================================================
with st.expander("Does this dashboard meet the required business scenarios?", expanded=False):
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        scenario_card(
            "Use case 1: Monitor actual and forecasted sales",
            "Supported through KPI cards, main weekly time-series chart, prediction scatter plot, and drill-down table.",
            ok=True,
        )
    with c2:
        scenario_card(
            "Use case 2: Inventory and restocking decisions",
            "Partly supported with forecasted demand rankings and high-demand alerts. True next-week ordering requires a future forecast file and inventory stock data.",
            ok=False,
            partial=True,
        )
    with c3:
        scenario_card(
            "Use case 3: Identify weak stores/departments",
            "Supported through actual-vs-forecast gaps, risk groups, Store-Dept details, and ranking charts.",
            ok=True,
        )
    with c4:
        scenario_card(
            "Use case 4: Promotion and holiday impact",
            "Supported with holiday-week comparison, markdown comparison, promotion filters, and dynamic recommendations.",
            ok=True,
        )

# ============================================================
# KPI cards
# ============================================================
st.subheader("Executive KPI Overview")
k1, k2, k3, k4, k5, k6 = st.columns(6)
k1.metric("Actual sales", money(total_actual, "million"), help="Total actual weekly sales after filters")
k2.metric("Forecasted sales", money(total_forecast, "million"), help=f"Selected model: {model_label}")
k3.metric("MAE", money(mae), help="Average absolute forecast deviation per Store-Dept week")
k4.metric("Raw WAPE", pct(wape), help="Raw weighted absolute percentage error; better for business monitoring than raw MAPE")
k5.metric("Forecast credibility", pct(r2, 2), help="Filtered R² shown in business terms")
k6.metric("Risk records", f"{risk_count:,}", help=f"Meaningful error records above {acceptable_error_pct * 2:.0f}% business error")
if presentation_mode and bias_adjust_display and abs(display_bias_correction) > 1e-9:
    st.caption(
        f"Presentation mode is ON: main chart uses a labelled bias-adjusted display line. Raw model metrics are still shown in the KPI cards. "
        f"Raw WAPE: {pct(wape)} | Display-calibrated WAPE: {pct(display_wape)}."
    )

# ============================================================
# Main layout: alerts + main visualization
# ============================================================
st.markdown('<div class="soft-divider"></div>', unsafe_allow_html=True)
left, main = st.columns([0.95, 1.9], gap="large")

with left:
    st.subheader("Business Alerts")
    if pd.isna(wape):
        card("Forecast quality unavailable", "WAPE cannot be calculated because actual sales are unavailable or zero in this filtered view.", "Data", "yellow")
    elif wape <= acceptable_error_pct / 100.0:
        card("Forecast stable", f"Overall WAPE is {pct(wape)}, within the selected {acceptable_error_pct}% tolerance.", "Green", "green")
    elif wape <= acceptable_error_pct / 50.0:
        card("Forecast watch", f"Overall WAPE is {pct(wape)}. Use forecasts, but check high-error store-department groups.", "Yellow", "yellow")
    else:
        card("Forecast risk", f"Overall WAPE is {pct(wape)}. Do not use this slice for automatic ordering without review.", "Red", "red")

    if high_demand_count > 0:
        card("High demand watch", f"{high_demand_count:,} Store-Dept weekly records are above their normal forecast level. Prioritize replenishment review.", "Demand", "yellow")
    else:
        card("Demand pressure normal", "No strong high-demand spike is detected under the current filters.", "Demand", "green")

    if pd.notna(bias) and bias > 0:
        card("Over-forecast bias", f"Average forecast is {money(bias)} higher than actual per record. This may create over-ordering risk.", "Bias", "yellow")
    elif pd.notna(bias) and bias < 0:
        card("Under-forecast bias", f"Average forecast is {money(abs(bias))} lower than actual per record. This may create stockout risk.", "Bias", "yellow")
    else:
        card("Low average bias", "The selected filtered view has almost no average forecasting bias.", "Bias", "green")

with main:
    st.subheader("Main Visualization: Actual Sales vs Forecasted Sales")
    if presentation_mode:
        st.markdown(
            '<div class="small-note">Presentation-friendly view: the main chart uses the selected business aggregation level and can show a clearly labelled bias-adjusted display line. Raw model metrics are not overwritten.</div>',
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            '<div class="small-note">Audit view: the main chart uses the raw forecast line. Hover shows the absolute and percentage gap.</div>',
            unsafe_allow_html=True,
        )

    chart_df = filtered_pred.copy()
    if main_chart_level == "Monthly total sales":
        chart_df["Chart_Date"] = chart_df["Date"].dt.to_period("M").dt.to_timestamp()
        agg_method = "sum"
        scale = 1_000_000
        y_label = "Sales (USD millions)"
        unit_label = "USD million"
        y_suffix = "M"
        caption_level = "monthly total"
    elif main_chart_level == "Weekly total sales":
        chart_df["Chart_Date"] = chart_df["Date"]
        agg_method = "sum"
        scale = 1_000_000
        y_label = "Sales (USD millions)"
        unit_label = "USD million"
        y_suffix = "M"
        caption_level = "weekly total"
    else:
        chart_df["Chart_Date"] = chart_df["Date"]
        agg_method = "mean"
        scale = 1_000
        y_label = "Average sales per Store-Dept week (USD thousands)"
        unit_label = "USD thousand"
        y_suffix = "K"
        caption_level = "average Store-Dept weekly"

    forecast_for_chart = "Display_Prediction" if presentation_mode else "Selected_Prediction"
    if agg_method == "sum":
        time_df = (
            chart_df.groupby("Chart_Date", as_index=False)
            .agg(Actual_Sales=(ACTUAL_COL, "sum"), Forecasted_Sales=(forecast_for_chart, "sum"), Records=(ACTUAL_COL, "size"))
            .sort_values("Chart_Date")
        )
    else:
        time_df = (
            chart_df.groupby("Chart_Date", as_index=False)
            .agg(Actual_Sales=(ACTUAL_COL, "mean"), Forecasted_Sales=(forecast_for_chart, "mean"), Records=(ACTUAL_COL, "size"))
            .sort_values("Chart_Date")
        )

    time_df["Actual_Display"] = time_df["Actual_Sales"] / scale
    time_df["Forecast_Display"] = time_df["Forecasted_Sales"] / scale
    time_df["Gap_Display"] = time_df["Forecast_Display"] - time_df["Actual_Display"]
    time_df["Gap_Pct"] = np.where(time_df["Actual_Sales"] > 0, (time_df["Forecasted_Sales"] - time_df["Actual_Sales"]) / time_df["Actual_Sales"], np.nan)

    forecast_name = "Bias-adjusted forecast" if (presentation_mode and bias_adjust_display and abs(display_bias_correction) > 1e-9) else "Forecasted sales"
    fig_time = go.Figure()
    custom_actual = np.stack([time_df["Gap_Display"], time_df["Gap_Pct"], time_df["Records"]], axis=-1)
    fig_time.add_trace(go.Scatter(
        x=time_df["Chart_Date"], y=time_df["Actual_Display"], mode="lines+markers",
        name="Actual sales", line=dict(color=BLUE, width=3), marker=dict(size=6),
        customdata=custom_actual,
        hovertemplate=f"Date=%{{x|%Y-%m-%d}}<br>Actual=%{{y:,.2f}} {unit_label}<br>Forecast gap=%{{customdata[0]:+,.2f}} {unit_label}<br>Gap rate=%{{customdata[1]:+.1%}}<br>Records=%{{customdata[2]:,.0f}}<extra></extra>",
    ))
    fig_time.add_trace(go.Scatter(
        x=time_df["Chart_Date"], y=time_df["Forecast_Display"], mode="lines+markers",
        name=forecast_name, line=dict(color=ORANGE, width=3, dash="dash"), marker=dict(size=6),
        customdata=custom_actual,
        hovertemplate=f"Date=%{{x|%Y-%m-%d}}<br>Forecast=%{{y:,.2f}} {unit_label}<br>Forecast gap=%{{customdata[0]:+,.2f}} {unit_label}<br>Gap rate=%{{customdata[1]:+.1%}}<br>Records=%{{customdata[2]:,.0f}}<extra></extra>",
    ))
    fig_time.update_layout(xaxis_title="Date", yaxis_title=y_label, hovermode="x unified")
    fig_time.update_xaxes(rangeslider_visible=True)
    if main_axis_zero:
        y_max = float(np.nanmax([time_df["Actual_Display"].max(), time_df["Forecast_Display"].max()])) if not time_df.empty else 1.0
        fig_time.update_yaxes(tickformat=",.1f", range=[0, y_max * 1.12])
    else:
        fig_time.update_yaxes(tickformat=",.1f")
    st.plotly_chart(chart_style(fig_time, height=560), use_container_width=True, theme=None)

    max_gap = time_df["Gap_Display"].abs().max() if not time_df.empty else np.nan
    avg_gap_pct = time_df["Gap_Pct"].abs().mean() if not time_df.empty else np.nan
    if presentation_mode and bias_adjust_display and abs(display_bias_correction) > 1e-9:
        st.caption(
            f"Presentation check: this chart shows {caption_level} values and uses a labelled bias adjustment of {money(display_bias_correction)} per Store-Dept record. "
            f"The largest displayed gap is about {max_gap:,.2f} {y_suffix}, and the average displayed gap rate is {avg_gap_pct:.1%}. "
            f"Raw model WAPE is {pct(wape)}; display-calibrated WAPE is {pct(display_wape)}."
        )
    else:
        st.caption(
            f"Chart check: this chart shows {caption_level} values. The largest displayed gap is about {max_gap:,.2f} {y_suffix}, "
            f"and the average displayed gap rate is {avg_gap_pct:.1%}. Raw model WAPE is {pct(wape)}."
        )

# ============================================================
# Dynamic recommendations
# ============================================================
st.markdown('<div class="soft-divider"></div>', unsafe_allow_html=True)
st.subheader("Dynamic Recommendations Panel")
st.markdown(
    '<div class="small-note">Recommendations are generated from the selected filters, forecast deviation, demand pressure, holiday effect, and markdown activity.</div>',
    unsafe_allow_html=True,
)
rec_cols = st.columns(3)
for i, (title, body, tag, status) in enumerate(build_recommendations(filtered_pred, filtered_sales, acceptable_error_pct)):
    with rec_cols[i % 3]:
        card(title, body, tag, status)

# ============================================================
# Tabs by business scenario
# ============================================================
st.markdown('<div class="soft-divider"></div>', unsafe_allow_html=True)
tab_monitor, tab_inventory, tab_performance, tab_promo, tab_model, tab_tables = st.tabs([
    "1. Sales Monitoring",
    "2. Inventory Planning",
    "3. Store / Dept Performance",
    "4. Promotion & Holiday",
    "5. Model Explainability",
    "6. Drill-Down Tables",
])

# ------------------------------
# 1. Sales Monitoring
# ------------------------------
with tab_monitor:
    st.subheader("Actual Weekly Sales and Forecast Monitoring")
    a, b = st.columns([1.25, 1.0], gap="large")
    with a:
        error_trend = (
            filtered_pred.groupby("Date", as_index=False)
            .agg(Abs_Error=("Absolute_Error", "sum"), Actual=(ACTUAL_COL, "sum"))
            .sort_values("Date")
        )
        error_trend["WAPE"] = np.where(error_trend["Actual"] > 0, error_trend["Abs_Error"] / error_trend["Actual"], np.nan)
        fig_error = px.line(
            error_trend,
            x="Date",
            y="WAPE",
            markers=True,
            title="Forecast Deviation Trend (WAPE)",
            labels={"WAPE": "Weighted forecast deviation", "Date": "Date"},
        )
        fig_error.update_traces(line=dict(color=RED, width=3), marker=dict(size=7))
        fig_error.update_yaxes(tickformat=".0%")
        st.plotly_chart(chart_style(fig_error, height=430), use_container_width=True, theme=None)
    with b:
        status_counts = filtered_pred["Forecast_Status"].value_counts().reindex(["Good", "Watch", "Risk"]).fillna(0).reset_index()
        status_counts.columns = ["Status", "Records"]
        fig_status = px.bar(
            status_counts,
            x="Status",
            y="Records",
            title="Forecast Status Distribution",
            labels={"Records": "Number of Store-Dept weeks"},
            color="Status",
            color_discrete_map={"Good": GREEN, "Watch": ORANGE, "Risk": RED},
        )
        st.plotly_chart(chart_style(fig_status, height=430), use_container_width=True, theme=None)

    c, d = st.columns([1.25, 1.0], gap="large")
    with c:
        scatter_df = filtered_pred.copy()
        if len(scatter_df) > 20000:
            scatter_df = scatter_df.sample(20000, random_state=42)
        scatter_df["Actual_Thousand"] = scatter_df[ACTUAL_COL] / 1_000
        scatter_df["Forecast_Thousand"] = scatter_df["Selected_Prediction"] / 1_000
        max_axis = max(scatter_df["Actual_Thousand"].max(), scatter_df["Forecast_Thousand"].max())
        hover_cols = [c for c in ["Store", "Dept", "Date", "Type", "Store_Size_Category", "Holiday_Week", "Season", "Business_APE"] if c in scatter_df.columns]
        fig_scatter = px.scatter(
            scatter_df,
            x="Actual_Thousand",
            y="Forecast_Thousand",
            color="Forecast_Status",
            hover_data=hover_cols,
            title="Prediction Accuracy: Forecast vs Actual",
            labels={
                "Actual_Thousand": "Actual sales (USD thousands)",
                "Forecast_Thousand": "Forecasted sales (USD thousands)",
                "Forecast_Status": "Status",
            },
            color_discrete_map={"Good": GREEN, "Watch": ORANGE, "Risk": RED},
        )
        fig_scatter.add_trace(go.Scatter(
            x=[0, max_axis], y=[0, max_axis], mode="lines", name="Perfect prediction",
            line=dict(color=GRAY, width=2, dash="dot"), hoverinfo="skip",
        ))
        st.plotly_chart(chart_style(fig_scatter, height=500), use_container_width=True, theme=None)
    with d:
        st.markdown("#### Business interpretation")
        card("MAE", f"The average weekly Store-Dept forecast miss is <b>{money(mae)}</b>. This is easier for managers to understand than raw model loss.", "Metric", "green" if mae < 1500 else "yellow")
        card("WAPE", f"The selected view has <b>{pct(wape)}</b> weighted forecast deviation. This is more stable than MAPE when actual sales are close to zero.", "Metric", "green" if pd.notna(wape) and wape <= acceptable_error_pct / 100 else "yellow")
        card("R²", f"The model explains about <b>{pct(r2, 2)}</b> of sales variation in this filtered view.", "Metric", "green")

# ------------------------------
# 2. Inventory Planning
# ------------------------------
with tab_inventory:
    st.subheader("Inventory and Restocking Support")
    st.markdown(
        '<div class="small-note">This tab answers “which stores and departments need inventory attention?” using forecasted demand and demand-pressure alerts. For true next-week ordering, add a future forecast file and current inventory stock levels.</div>',
        unsafe_allow_html=True,
    )
    inv1, inv2 = st.columns([1.1, 1.0], gap="large")
    demand_group = (
        filtered_pred.groupby(["Store", "Dept"], as_index=False)
        .agg(
            Forecasted_Sales=("Selected_Prediction", "sum"),
            Actual_Sales=(ACTUAL_COL, "sum"),
            High_Demand_Weeks=("High_Demand_Alert", "sum"),
            Weeks=("Date", "nunique"),
            MAE=("Absolute_Error", "mean"),
        )
    )
    demand_group["Demand_Pressure_Score"] = demand_group["Forecasted_Sales"].rank(pct=True) * 70 + demand_group["High_Demand_Weeks"].rank(pct=True) * 30
    demand_top = demand_group.sort_values(["Demand_Pressure_Score", "Forecasted_Sales"], ascending=False).head(top_n).copy()
    demand_top["Store_Dept"] = demand_top.apply(lambda r: f"Store {int(r['Store'])} / Dept {int(r['Dept'])}", axis=1)
    demand_top = demand_top.sort_values("Forecasted_Sales", ascending=True)
    with inv1:
        fig_demand = px.bar(
            demand_top,
            x="Forecasted_Sales",
            y="Store_Dept",
            orientation="h",
            title="Restocking Priority by Forecasted Demand",
            labels={"Forecasted_Sales": "Forecasted sales (USD)", "Store_Dept": "Store / Department"},
            hover_data=["High_Demand_Weeks", "Weeks", "MAE"],
        )
        fig_demand.update_traces(marker_color=BLUE, hovertemplate="%{y}<br>Forecasted sales=US$%{x:,.0f}<extra></extra>")
        fig_demand.update_xaxes(tickprefix="US$", tickformat="~s")
        st.plotly_chart(chart_style(fig_demand, height=500), use_container_width=True, theme=None)
    with inv2:
        st.markdown("#### Restocking guidance")
        top3 = demand_group.sort_values("Forecasted_Sales", ascending=False).head(3)
        for _, row in top3.iterrows():
            card(
                f"Store {int(row.Store)} / Dept {int(row.Dept)}",
                f"Forecasted demand is <b>{money(row.Forecasted_Sales)}</b> across {int(row.Weeks)} selected week(s). High-demand weeks: <b>{int(row.High_Demand_Weeks)}</b>.",
                "Review",
                "yellow",
            )
        card(
            "Data limitation",
            "The uploaded files do not include current stock, safety stock, lead time, or future-week forecasts. Therefore, the dashboard can prioritize attention, but it cannot calculate exact order quantities yet.",
            "Need Data",
            "yellow",
        )

# ------------------------------
# 3. Store / Department Performance
# ------------------------------
with tab_performance:
    st.subheader("Store and Department Performance Review")
    p1, p2 = st.columns(2, gap="large")
    with p1:
        store_perf = (
            filtered_pred.groupby("Store", as_index=False)
            .agg(Actual_Sales=(ACTUAL_COL, "sum"), Forecasted_Sales=("Selected_Prediction", "sum"), Abs_Error=("Absolute_Error", "sum"))
        )
        store_perf["Gap"] = store_perf["Actual_Sales"] - store_perf["Forecasted_Sales"]
        store_perf["Store_Label"] = store_perf["Store"].apply(lambda x: f"Store {int(x)}" if pd.notna(x) else "Unknown")
        store_rank = store_perf.sort_values("Actual_Sales", ascending=False).head(top_n).sort_values("Actual_Sales", ascending=True)
        fig_store = px.bar(
            store_rank,
            x="Actual_Sales",
            y="Store_Label",
            orientation="h",
            title="Top Stores by Actual Sales",
            labels={"Actual_Sales": "Actual sales (USD)", "Store_Label": "Store"},
            hover_data=["Forecasted_Sales", "Gap"],
        )
        fig_store.update_traces(marker_color=BLUE, hovertemplate="%{y}<br>Actual sales=US$%{x:,.0f}<extra></extra>")
        fig_store.update_xaxes(tickprefix="US$", tickformat="~s")
        st.plotly_chart(chart_style(fig_store, height=470), use_container_width=True, theme=None)
    with p2:
        weak = store_perf.sort_values("Gap", ascending=True).head(top_n).sort_values("Gap", ascending=True).copy()
        weak["Store_Label"] = weak["Store"].apply(lambda x: f"Store {int(x)}" if pd.notna(x) else "Unknown")
        weak["Below_Forecast"] = -weak["Gap"]
        fig_weak = px.bar(
            weak,
            x="Below_Forecast",
            y="Store_Label",
            orientation="h",
            title="Stores Most Below Forecast",
            labels={"Below_Forecast": "Sales below forecast (USD)", "Store_Label": "Store"},
            hover_data=["Actual_Sales", "Forecasted_Sales"],
        )
        fig_weak.update_traces(marker_color=RED, hovertemplate="%{y}<br>Below forecast=US$%{x:,.0f}<extra></extra>")
        fig_weak.update_xaxes(tickprefix="US$", tickformat="~s")
        st.plotly_chart(chart_style(fig_weak, height=470), use_container_width=True, theme=None)

    p3, p4 = st.columns(2, gap="large")
    with p3:
        dept_error = (
            filtered_pred.groupby("Dept", as_index=False)
            .agg(Actual_Sales=(ACTUAL_COL, "sum"), Abs_Error=("Absolute_Error", "sum"), MAE=("Absolute_Error", "mean"))
        )
        dept_error["WAPE"] = np.where(dept_error["Actual_Sales"] > 0, dept_error["Abs_Error"] / dept_error["Actual_Sales"], np.nan)
        dept_error["Dept_Label"] = dept_error["Dept"].apply(lambda x: f"Dept {int(x)}" if pd.notna(x) else "Unknown")
        dept_top = dept_error.sort_values("WAPE", ascending=False).head(top_n).sort_values("WAPE", ascending=True)
        fig_dept = px.bar(
            dept_top,
            x="WAPE",
            y="Dept_Label",
            orientation="h",
            title="Departments with Highest Forecast Deviation",
            labels={"WAPE": "WAPE", "Dept_Label": "Department"},
            hover_data=["MAE", "Actual_Sales"],
        )
        fig_dept.update_traces(marker_color=ORANGE)
        fig_dept.update_xaxes(tickformat=".0%")
        st.plotly_chart(chart_style(fig_dept, height=450), use_container_width=True, theme=None)
    with p4:
        group_options = [c for c in ["Type", "Store_Size_Category", "Season", "Month"] if c in filtered_pred.columns]
        if group_options:
            group_dim = st.selectbox("Compare performance by", group_options, index=0)
            group_perf = (
                filtered_pred.groupby(group_dim, as_index=False)
                .agg(Actual=(ACTUAL_COL, "sum"), Forecast=("Selected_Prediction", "sum"))
            )
            group_perf["Group"] = group_perf[group_dim].astype(str)
            long = group_perf.melt(id_vars="Group", value_vars=["Actual", "Forecast"], var_name="Metric", value_name="Sales")
            fig_group = px.bar(
                long,
                x="Group",
                y="Sales",
                color="Metric",
                barmode="group",
                title=f"Actual vs Forecast by {group_dim}",
                labels={"Sales": "Sales (USD)", "Group": group_dim},
                color_discrete_map={"Actual": BLUE, "Forecast": ORANGE},
            )
            fig_group.update_yaxes(tickprefix="US$", tickformat="~s")
            st.plotly_chart(chart_style(fig_group, height=450), use_container_width=True, theme=None)

# ------------------------------
# 4. Promotion & Holiday
# ------------------------------
with tab_promo:
    st.subheader("Promotion and Holiday Effectiveness")
    h1, h2 = st.columns(2, gap="large")
    with h1:
        if "Holiday_Week" in filtered_pred.columns:
            holiday_df = (
                filtered_pred.groupby("Holiday_Week", as_index=False)
                .agg(Actual=(ACTUAL_COL, "mean"), Forecast=("Selected_Prediction", "mean"))
            )
            holiday_df["Week_Type"] = holiday_df["Holiday_Week"].map({0: "Non-holiday", 1: "Holiday"}).fillna(holiday_df["Holiday_Week"].astype(str))
            long_h = holiday_df.melt(id_vars="Week_Type", value_vars=["Actual", "Forecast"], var_name="Metric", value_name="Average_Sales")
            fig_h = px.bar(
                long_h,
                x="Week_Type",
                y="Average_Sales",
                color="Metric",
                barmode="group",
                title="Holiday vs Non-Holiday Average Sales",
                labels={"Average_Sales": "Average weekly sales (USD)", "Week_Type": "Week type"},
                color_discrete_map={"Actual": BLUE, "Forecast": ORANGE},
            )
            fig_h.update_yaxes(tickprefix="US$", tickformat="~s")
            st.plotly_chart(chart_style(fig_h, height=450), use_container_width=True, theme=None)
        else:
            st.info("Holiday_Week column is not available.")
    with h2:
        if "Any_MarkDown" in filtered_pred.columns:
            promo_df = (
                filtered_pred.groupby("Any_MarkDown", as_index=False)
                .agg(Actual=(ACTUAL_COL, "mean"), Forecast=("Selected_Prediction", "mean"), Markdown=("Total_MarkDown", "mean"))
            )
            promo_df["Promotion_Status"] = promo_df["Any_MarkDown"].map({0: "No markdown", 1: "Markdown active"}).fillna(promo_df["Any_MarkDown"].astype(str))
            long_p = promo_df.melt(id_vars="Promotion_Status", value_vars=["Actual", "Forecast"], var_name="Metric", value_name="Average_Sales")
            fig_p = px.bar(
                long_p,
                x="Promotion_Status",
                y="Average_Sales",
                color="Metric",
                barmode="group",
                title="Markdown Promotion Effect",
                labels={"Average_Sales": "Average weekly sales (USD)", "Promotion_Status": "Promotion status"},
                color_discrete_map={"Actual": BLUE, "Forecast": ORANGE},
            )
            fig_p.update_yaxes(tickprefix="US$", tickformat="~s")
            st.plotly_chart(chart_style(fig_p, height=450), use_container_width=True, theme=None)
        else:
            st.info("Any_MarkDown column is not available.")

    if "Holiday_Period" in filtered_sales.columns:
        period_df = (
            filtered_sales.groupby("Holiday_Period", as_index=False)[ACTUAL_COL]
            .mean()
            .sort_values(ACTUAL_COL, ascending=False)
        )
        fig_period = px.bar(
            period_df,
            x="Holiday_Period",
            y=ACTUAL_COL,
            title="Before / During / After Holiday Sales Pattern",
            labels={ACTUAL_COL: "Average actual weekly sales (USD)", "Holiday_Period": "Holiday period"},
        )
        fig_period.update_traces(marker_color=PURPLE)
        fig_period.update_yaxes(tickprefix="US$", tickformat="~s")
        st.plotly_chart(chart_style(fig_period, height=430), use_container_width=True, theme=None)

# ------------------------------
# 5. Model Explainability
# ------------------------------
with tab_model:
    st.subheader("Model Evaluation and Feature Importance")
    m1, m2 = st.columns([1.0, 1.15], gap="large")
    with m1:
        if not evaluation.empty:
            show_eval = evaluation.copy()
            numeric_cols = show_eval.select_dtypes(include=[np.number]).columns.tolist()
            st.dataframe(
                show_eval.style.format({c: "{:,.4f}" for c in numeric_cols}),
                use_container_width=True,
                height=245,
            )
            best = show_eval.sort_values("MAE").iloc[0] if "MAE" in show_eval.columns else None
            if best is not None:
                card(
                    "Best model based on MAE",
                    f"<b>{escape(str(best['Model']))}</b> has the lowest MAE in the uploaded evaluation file. This is why it is selected first in the sidebar.",
                    "Model Choice",
                    "green",
                )
        else:
            st.info("Model evaluation table is unavailable.")
    with m2:
        if not feature_importance.empty and {"Feature", "Importance"}.issubset(feature_importance.columns):
            fi = feature_importance.copy()
            fi["Readable_Feature"] = fi["Feature"].apply(clean_feature_name)
            fi = fi.sort_values("Importance", ascending=False).head(15).sort_values("Importance", ascending=True)
            fig_fi = px.bar(
                fi,
                x="Importance",
                y="Readable_Feature",
                orientation="h",
                title="Top Predictive Features",
                labels={"Readable_Feature": "Feature", "Importance": "Importance"},
            )
            fig_fi.update_traces(marker_color=GREEN)
            st.plotly_chart(chart_style(fig_fi, height=530), use_container_width=True, theme=None)
        else:
            st.info("Feature importance file is missing or does not contain Feature / Importance columns.")

# ------------------------------
# 6. Drill-down tables
# ------------------------------
with tab_tables:
    st.subheader("Detailed Data for Managers")
    table1, table2, table3 = st.tabs(["Store-Dept Summary", "Worst Error Records", "Filtered Preview"])

    with table1:
        detail = (
            filtered_pred.groupby(["Store", "Dept"], as_index=False)
            .agg(
                Actual_Sales=(ACTUAL_COL, "sum"),
                Forecasted_Sales=("Selected_Prediction", "sum"),
                Abs_Error=("Absolute_Error", "sum"),
                MAE=("Absolute_Error", "mean"),
                Risk_Records=("Business_Risk_Record", "sum"),
                High_Demand_Weeks=("High_Demand_Alert", "sum"),
                Weeks=("Date", "nunique"),
            )
        )
        detail["WAPE"] = np.where(detail["Actual_Sales"] > 0, detail["Abs_Error"] / detail["Actual_Sales"], np.nan)
        detail["Sales_Gap"] = detail["Actual_Sales"] - detail["Forecasted_Sales"]
        detail = detail.sort_values(["Risk_Records", "WAPE", "Forecasted_Sales"], ascending=False)
        st.dataframe(
            detail.style.format({
                "Actual_Sales": "US${:,.0f}",
                "Forecasted_Sales": "US${:,.0f}",
                "Abs_Error": "US${:,.0f}",
                "MAE": "US${:,.0f}",
                "WAPE": "{:.1%}",
                "Sales_Gap": "US${:,.0f}",
                "Risk_Records": "{:,.0f}",
                "High_Demand_Weeks": "{:,.0f}",
                "Weeks": "{:,.0f}",
            }),
            use_container_width=True,
            height=470,
        )

    with table2:
        if not worst_predictions.empty:
            worst = enrich_predictions(add_store_size_category(worst_predictions), sales)
            worst = apply_filters(
                worst,
                start_date,
                end_date,
                selected_stores,
                selected_depts,
                selected_types,
                selected_sizes,
                selected_seasons,
                holiday_filter,
                promo_filter,
            )
            display_cols = [c for c in [
                "Store", "Dept", "Date", ACTUAL_COL, "Type", "Store_Size_Category", "Holiday_Week", "Any_MarkDown",
                "Naive_Prediction", "Linear_Regression_Prediction", "Random_Forest_Prediction",
                "Linear_Absolute_Error", "RF_Absolute_Error", "Linear_Percentage_Error", "RF_Percentage_Error",
            ] if c in worst.columns]
            st.dataframe(worst[display_cols].head(50), use_container_width=True, height=470)
        else:
            st.info("Worst prediction file is unavailable.")

    with table3:
        preview_cols = [c for c in [
            "Store", "Dept", "Date", ACTUAL_COL, "Selected_Prediction", "Prediction_Error", "Absolute_Error",
            "Business_APE", "Raw_APE", "Forecast_Status", "Business_Risk_Record", "Type", "Store_Size_Category",
            "Holiday_Week", "Any_MarkDown", "Total_MarkDown", "Season", "Month", "WeekOfYear",
        ] if c in filtered_pred.columns]
        st.dataframe(filtered_pred[preview_cols].head(1000), use_container_width=True, height=470)

# ============================================================
# Footer
# ============================================================
st.markdown('<div class="soft-divider"></div>', unsafe_allow_html=True)
st.caption(
    "Scope note: this dashboard uses your uploaded historical cleaned data, test-set predictions, feature importance, model evaluation table, and worst-error records. "
    "It supports monitoring, comparison, promotion/holiday review, and forecast-based inventory attention. Exact next-week order quantities require a future forecast file plus inventory stock, safety stock, lead time, and capacity data."
)
