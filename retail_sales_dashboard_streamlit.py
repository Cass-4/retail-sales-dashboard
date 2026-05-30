# ================================================================
# Retail Sales Forecasting Dashboard
# Streamlit dashboard for weekly sales forecasting, inventory planning,
# performance monitoring, holiday/promotion evaluation, feature importance,
# and model prediction review.
#
# Required CSV files in the same folder as this script:
# 1. scenario3_cleaned(4).csv
# 2. model_test_predictions(2).csv
# 3. random_forest_feature_importance(3).csv
# 4. comprehensive_evaluation_table(2).csv
# 5. worst_50_predictions_random_forest(2).csv
#
# Run:
#   pip install streamlit pandas numpy plotly
#   streamlit run retail_sales_dashboard_streamlit.py
# ================================================================

from pathlib import Path
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st


# -----------------------------
# Page configuration
# -----------------------------
st.set_page_config(
    page_title="Retail Sales Forecasting Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Color-blind friendly palette based on Okabe-Ito / Plotly safe colors
COLOR_ACTUAL = "#0072B2"       # blue
COLOR_FORECAST = "#E69F00"     # orange
COLOR_GOOD = "#009E73"         # bluish green
COLOR_WARNING = "#E69F00"      # orange/yellow
COLOR_RISK = "#D55E00"         # vermillion
COLOR_NEUTRAL = "#666666"

px.defaults.template = "plotly_white"
px.defaults.color_discrete_sequence = px.colors.qualitative.Safe

st.markdown(
    """
    <style>
    html, body, [class*="css"] {
        font-family: "Inter", "Segoe UI", Arial, sans-serif;
    }
    .main-title {
        font-size: 2.1rem;
        font-weight: 800;
        margin-bottom: 0.15rem;
    }
    .subtitle {
        color: #555;
        font-size: 1rem;
        margin-bottom: 1.0rem;
    }
    .kpi-card {
        border: 1px solid #E6E6E6;
        border-radius: 16px;
        padding: 1rem 1.05rem;
        background: #FFFFFF;
        box-shadow: 0 2px 10px rgba(0,0,0,0.045);
        min-height: 120px;
    }
    .kpi-label {
        color: #555;
        font-size: 0.85rem;
        font-weight: 650;
        margin-bottom: 0.5rem;
    }
    .kpi-value {
        color: #111;
        font-size: 1.55rem;
        font-weight: 800;
        line-height: 1.1;
    }
    .kpi-help {
        color: #777;
        font-size: 0.78rem;
        margin-top: 0.45rem;
    }
    .alert-card {
        border-radius: 14px;
        padding: 0.85rem 1rem;
        margin-bottom: 0.65rem;
        border-left: 8px solid;
        background: #fff;
        box-shadow: 0 1px 7px rgba(0,0,0,0.045);
    }
    .alert-green {
        border-left-color: #009E73;
        background: #F2FBF7;
    }
    .alert-yellow {
        border-left-color: #E69F00;
        background: #FFF8E8;
    }
    .alert-red {
        border-left-color: #D55E00;
        background: #FFF2EC;
    }
    .section-note {
        color: #666;
        font-size: 0.88rem;
        margin-top: -0.5rem;
        margin-bottom: 0.6rem;
    }
    .recommendation {
        border: 1px solid #EAEAEA;
        border-radius: 14px;
        padding: 0.85rem 1rem;
        margin-bottom: 0.65rem;
        background: #FFFFFF;
    }
    .tag {
        display: inline-block;
        border-radius: 999px;
        padding: 0.1rem 0.55rem;
        font-size: 0.75rem;
        font-weight: 700;
        background: #EEF3F8;
        color: #333;
        margin-left: 0.35rem;
    }
    .small-muted {
        color: #777;
        font-size: 0.82rem;
    }
    </style>
    """,
    unsafe_allow_html=True
)


# -----------------------------
# File paths
# -----------------------------
BASE_DIR = Path(__file__).resolve().parent if "__file__" in globals() else Path.cwd()

DATA_FILES = {
    "cleaned_sales": BASE_DIR / "sampled_data.csv",
    "predictions": BASE_DIR / "model_test_predictions.csv",
    "feature_importance": BASE_DIR / "random_forest_feature_importance.csv",
    "evaluation": BASE_DIR / "comprehensive_evaluation_table.csv",
    "worst_predictions": BASE_DIR / "worst_50_predictions_random_forest.csv",
}


# -----------------------------
# Helpers
# -----------------------------
def file_status_badge(path: Path) -> str:
    return "✅ Found" if path.exists() else "❌ Missing"


@st.cache_data(show_spinner=False)
def load_csv(path_str: str, parse_date: bool = False) -> pd.DataFrame:
    path = Path(path_str)
    if not path.exists():
        return pd.DataFrame()

    if parse_date:
        df = pd.read_csv(path, parse_dates=["Date"])
    else:
        df = pd.read_csv(path)

    return df


def add_store_size_category(df: pd.DataFrame) -> pd.DataFrame:
    """Create Store_Size_Category if the detailed sales data only has Size."""
    if df.empty or "Store_Size_Category" in df.columns or "Size" not in df.columns or "Store" not in df.columns:
        return df

    output = df.copy()
    store_sizes = output[["Store", "Size"]].drop_duplicates().sort_values("Size")
    store_sizes["rank_for_qcut"] = store_sizes["Size"].rank(method="first")

    try:
        store_sizes["Store_Size_Category"] = pd.qcut(
            store_sizes["rank_for_qcut"],
            q=3,
            labels=["Small", "Medium", "Large"]
        )
    except ValueError:
        store_sizes["Store_Size_Category"] = "Unknown"

    size_map = store_sizes.set_index("Store")["Store_Size_Category"].astype(str).to_dict()
    output["Store_Size_Category"] = output["Store"].map(size_map).fillna("Unknown")
    return output


def clean_feature_name(name: str) -> str:
    """Make feature names easier for non-technical stakeholders to read."""
    name = str(name)
    for prefix in ["num__", "cat__", "remainder__"]:
        name = name.replace(prefix, "")
    name = name.replace("_", " ")
    name = name.replace("WeekOfYear", "Week of Year")
    name = name.replace("Lag 1 Week Sales", "Previous Week Sales")
    name = name.replace("Lag 4 Week Sales", "Sales 4 Weeks Ago")
    name = name.replace("Rolling 4 Week Mean Sales", "4-Week Rolling Average Sales")
    name = name.replace("Rolling 4 Week Avg Sales", "4-Week Rolling Average Sales")
    name = name.replace("Rolling 12 Week Avg Sales", "12-Week Rolling Average Sales")
    name = name.replace("Total MarkDown", "Total Markdown")
    name = name.replace("CPI", "Consumer Price Index")
    return name.title()


def money(value: float) -> str:
    if pd.isna(value):
        return "N/A"
    value = float(value)
    if abs(value) >= 1_000_000:
        return f"${value/1_000_000:,.2f}M"
    if abs(value) >= 1_000:
        return f"${value/1_000:,.1f}K"
    return f"${value:,.0f}"


def number(value: float, decimals: int = 0) -> str:
    if pd.isna(value):
        return "N/A"
    return f"{value:,.{decimals}f}"


def percent(value: float, decimals: int = 1) -> str:
    if pd.isna(value):
        return "N/A"
    return f"{value * 100:.{decimals}f}%"


def render_kpi(label: str, value: str, help_text: str = "") -> None:
    st.markdown(
        f"""
        <div class="kpi-card">
            <div class="kpi-label">{label}</div>
            <div class="kpi-value">{value}</div>
            <div class="kpi-help">{help_text}</div>
        </div>
        """,
        unsafe_allow_html=True
    )


def render_alert(title: str, message: str, status: str) -> None:
    css_class = {
        "green": "alert-green",
        "yellow": "alert-yellow",
        "red": "alert-red",
    }.get(status, "alert-yellow")

    st.markdown(
        f"""
        <div class="alert-card {css_class}">
            <b>{title}</b><br>
            <span>{message}</span>
        </div>
        """,
        unsafe_allow_html=True
    )


def safe_r2(y_true: pd.Series, y_pred: pd.Series) -> float:
    valid = pd.DataFrame({"y": y_true, "p": y_pred}).dropna()
    if len(valid) < 2:
        return np.nan
    ss_res = np.sum((valid["y"] - valid["p"]) ** 2)
    ss_tot = np.sum((valid["y"] - valid["y"].mean()) ** 2)
    if ss_tot == 0:
        return np.nan
    return 1 - ss_res / ss_tot


def filter_dataframe(
    df: pd.DataFrame,
    start_date,
    end_date,
    stores,
    depts,
    store_types,
    size_categories,
    holiday_filter,
    seasons=None,
    markdown_filter=None,
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

    if holiday_filter != "All" and "Holiday_Week" in out.columns:
        if holiday_filter == "Holiday weeks only":
            out = out[out["Holiday_Week"].astype(int) == 1]
        elif holiday_filter == "Non-holiday weeks only":
            out = out[out["Holiday_Week"].astype(int) == 0]

    if seasons and "Season" in out.columns:
        out = out[out["Season"].isin(seasons)]

    if markdown_filter and markdown_filter != "All" and "Any_MarkDown" in out.columns:
        if markdown_filter == "Markdown active":
            out = out[out["Any_MarkDown"].astype(int) == 1]
        elif markdown_filter == "No markdown":
            out = out[out["Any_MarkDown"].astype(int) == 0]

    return out


def get_model_row(evaluation_df: pd.DataFrame, model_label: str) -> pd.Series:
    if evaluation_df.empty or "Model" not in evaluation_df.columns:
        return pd.Series(dtype="float64")
    exact = evaluation_df[evaluation_df["Model"] == model_label]
    if len(exact) > 0:
        return exact.iloc[0]
    contains = evaluation_df[evaluation_df["Model"].str.contains(model_label, case=False, na=False)]
    if len(contains) > 0:
        return contains.iloc[0]
    return pd.Series(dtype="float64")


def add_prediction_fields(df: pd.DataFrame, prediction_col: str, error_threshold_pct: float) -> pd.DataFrame:
    output = df.copy()

    actual_col = "Weekly_Sales_NonNegative"
    if actual_col not in output.columns or prediction_col not in output.columns:
        return output

    output["Selected_Prediction"] = output[prediction_col]
    output["Prediction_Error"] = output["Selected_Prediction"] - output[actual_col]
    output["Absolute_Error"] = output["Prediction_Error"].abs()
    output["Percentage_Error"] = np.where(
        output[actual_col] > 0,
        output["Absolute_Error"] / output[actual_col],
        np.nan
    )

    low = error_threshold_pct / 100
    medium = min(low * 2, 0.50)

    conditions = [
        output["Percentage_Error"] <= low,
        output["Percentage_Error"] <= medium,
    ]
    choices = ["Good", "Watch"]
    output["Forecast_Status"] = np.select(conditions, choices, default="Risk")

    # Demand alert: current forecast is much higher than the selected Store-Dept average.
    group_cols = [c for c in ["Store", "Dept"] if c in output.columns]
    if group_cols:
        baseline = output.groupby(group_cols)["Selected_Prediction"].transform("mean")
        spread = output.groupby(group_cols)["Selected_Prediction"].transform("std").fillna(0)
        output["High_Demand_Alert"] = output["Selected_Prediction"] > (baseline + 0.75 * spread)
    else:
        output["High_Demand_Alert"] = output["Selected_Prediction"] > output["Selected_Prediction"].quantile(0.80)

    # Underperformance: model expected more sales than actual by a large margin.
    output["Underperformance_Alert"] = (
        (output["Prediction_Error"] > 0) &
        (output["Percentage_Error"] > low)
    )

    # Stockout / demand pressure signal:
    # actual sales higher than forecast can indicate demand exceeded model expectation.
    output["Demand_Pressure_Alert"] = (
        (output["Prediction_Error"] < 0) &
        (output["Percentage_Error"] > low)
    )

    return output


def make_recommendations(df: pd.DataFrame, sales_df: pd.DataFrame, prediction_col: str, error_threshold_pct: float):
    recs = []

    if df.empty:
        return ["No data available under the current filters. Widen the date range or reduce filters."]

    actual_col = "Weekly_Sales_NonNegative"
    total_actual = df[actual_col].sum()
    total_pred = df[prediction_col].sum()
    mae = df["Absolute_Error"].mean()
    mean_actual = df[actual_col].mean()
    error_rate = mae / mean_actual if mean_actual > 0 else np.nan
    bias = df["Prediction_Error"].mean()

    # 1. Replenishment priority
    high_demand = df[df["High_Demand_Alert"]].copy()
    if not high_demand.empty:
        group = (
            high_demand.groupby(["Store", "Dept"], as_index=False)
            .agg(Expected_Sales=("Selected_Prediction", "sum"),
                 Avg_Error=("Percentage_Error", "mean"),
                 Weeks=("Date", "nunique") if "Date" in high_demand.columns else ("Selected_Prediction", "count"))
            .sort_values("Expected_Sales", ascending=False)
            .head(3)
        )
        top_items = "; ".join(
            [f"Store {int(r.Store)} / Dept {int(r.Dept)} ({money(r.Expected_Sales)})" for _, r in group.iterrows()]
        )
        recs.append(
            f"Increase inventory attention for {top_items}. These Store-Dept combinations show high forecasted demand within the selected period. "
            f"<span class='tag'>Restocking</span>"
        )
    else:
        recs.append(
            "No strong high-demand spike is detected under the current filters. Keep normal replenishment rules and monitor weekly updates. "
            "<span class='tag'>Inventory</span>"
        )

    # 2. Forecast quality
    risky = df[df["Forecast_Status"] == "Risk"]
    if error_rate <= error_threshold_pct / 100:
        recs.append(
            f"Forecast credibility is acceptable: average forecast deviation is {percent(error_rate)} of average weekly sales. "
            "The selected model can be used for operational planning with routine monitoring. "
            "<span class='tag'>Model Monitoring</span>"
        )
    elif len(risky) > 0:
        group = (
            risky.groupby(["Store", "Dept"], as_index=False)
            .agg(Avg_Percentage_Error=("Percentage_Error", "mean"),
                 Avg_Absolute_Error=("Absolute_Error", "mean"))
            .sort_values("Avg_Percentage_Error", ascending=False)
            .head(3)
        )
        top_errors = "; ".join(
            [f"Store {int(r.Store)} / Dept {int(r.Dept)} ({percent(r.Avg_Percentage_Error)})" for _, r in group.iterrows()]
        )
        recs.append(
            f"Review forecast assumptions for {top_errors}. These areas exceed the selected error threshold and may need local business review. "
            "<span class='tag'>Forecast Risk</span>"
        )

    # 3. Bias management
    if pd.notna(bias):
        if bias > 0:
            recs.append(
                f"The model is over-forecasting on average by {money(bias)} per record. Investigate weak demand, promotion fatigue, or local economic pressure before ordering excess inventory. "
                "<span class='tag'>Bias Control</span>"
            )
        elif bias < 0:
            recs.append(
                f"The model is under-forecasting on average by {money(abs(bias))} per record. Watch for stockout risk because actual demand may exceed the forecast. "
                "<span class='tag'>Stockout Risk</span>"
            )

    # 4. Holiday demand
    if "Holiday_Week" in df.columns and df["Holiday_Week"].sum() > 0:
        holiday_df = df[df["Holiday_Week"].astype(int) == 1]
        non_holiday_df = df[df["Holiday_Week"].astype(int) == 0]
        if not holiday_df.empty and not non_holiday_df.empty:
            holiday_avg = holiday_df["Selected_Prediction"].mean()
            normal_avg = non_holiday_df["Selected_Prediction"].mean()
            uplift = (holiday_avg / normal_avg - 1) if normal_avg > 0 else np.nan
            if pd.notna(uplift) and uplift > 0.05:
                recs.append(
                    f"Holiday demand is expected to be {percent(uplift)} higher than non-holiday weeks. Prepare extra inventory and staff for holiday-sensitive departments. "
                    "<span class='tag'>Holiday</span>"
                )

    # 5. Promotion effectiveness from cleaned sales data
    if not sales_df.empty and "Any_MarkDown" in sales_df.columns and actual_col in sales_df.columns:
        markdown_sales = sales_df.groupby("Any_MarkDown")[actual_col].mean()
        if 0 in markdown_sales.index and 1 in markdown_sales.index:
            lift = markdown_sales.loc[1] / markdown_sales.loc[0] - 1 if markdown_sales.loc[0] > 0 else np.nan
            if pd.notna(lift):
                if lift > 0.05:
                    recs.append(
                        f"Markdown weeks show an average sales lift of {percent(lift)}. Marketing can prioritize similar promotions for departments with strong response. "
                        "<span class='tag'>Promotion</span>"
                    )
                else:
                    recs.append(
                        f"Markdown weeks show limited average lift ({percent(lift)}). Check whether discounts are too broad or concentrated in low-response departments. "
                        "<span class='tag'>Promotion</span>"
                    )

    return recs[:6]


# -----------------------------
# Load data
# -----------------------------
with st.spinner("Loading dashboard data..."):
    sales = load_csv(str(DATA_FILES["cleaned_sales"]), parse_date=True)
    predictions = load_csv(str(DATA_FILES["predictions"]), parse_date=True)
    feature_importance = load_csv(str(DATA_FILES["feature_importance"]), parse_date=False)
    evaluation = load_csv(str(DATA_FILES["evaluation"]), parse_date=False)
    worst_predictions = load_csv(str(DATA_FILES["worst_predictions"]), parse_date=True)

sales = add_store_size_category(sales)
predictions = add_store_size_category(predictions)

if predictions.empty:
    st.error(
        "The prediction file is missing. Please place `model_test_predictions(2).csv` in the same folder as this Streamlit script."
    )
    st.stop()


# -----------------------------
# Header
# -----------------------------
st.markdown("<div class='main-title'>Retail Weekly Sales Forecasting Dashboard</div>", unsafe_allow_html=True)
st.markdown(
    "<div class='subtitle'>For store managers, supply chain managers, marketing managers, regional supervisors, and finance teams.</div>",
    unsafe_allow_html=True
)


# -----------------------------
# Sidebar filters
# -----------------------------
st.sidebar.title("Filters & Controls")
st.sidebar.caption("Use the filters below to drill down from overall business performance to Store-Dept weekly details.")

with st.sidebar.expander("Data file status", expanded=False):
    for name, path in DATA_FILES.items():
        st.write(f"**{name}**: {file_status_badge(path)}")
        st.caption(str(path.name))

available_models = {
    "Model 1 - Linear Regression": "Linear_Regression_Prediction",
    "Model 2 - Tuned Random Forest": "Random_Forest_Prediction",
    "Naive Baseline": "Naive_Prediction",
}
available_models = {k: v for k, v in available_models.items() if v in predictions.columns}

model_label = st.sidebar.selectbox(
    "Forecasting model",
    options=list(available_models.keys()),
    index=0 if "Model 1 - Linear Regression" in available_models else 0,
    help="The selected model controls the forecast line, KPI deviation, prediction-vs-actual chart, and recommendations."
)
prediction_col = available_models[model_label]

min_date = predictions["Date"].min()
max_date = predictions["Date"].max()
date_range = st.sidebar.date_input(
    "Date range",
    value=(min_date.date(), max_date.date()),
    min_value=min_date.date(),
    max_value=max_date.date()
)

if isinstance(date_range, tuple) and len(date_range) == 2:
    start_date, end_date = date_range
else:
    start_date, end_date = min_date.date(), max_date.date()

stores_all = sorted(predictions["Store"].dropna().unique().tolist()) if "Store" in predictions.columns else []
depts_all = sorted(predictions["Dept"].dropna().unique().tolist()) if "Dept" in predictions.columns else []
types_all = sorted(predictions["Type"].dropna().unique().tolist()) if "Type" in predictions.columns else []
sizes_all = sorted(predictions["Store_Size_Category"].dropna().unique().tolist()) if "Store_Size_Category" in predictions.columns else []

selected_stores = st.sidebar.multiselect("Store", options=stores_all, default=[])
selected_depts = st.sidebar.multiselect("Department", options=depts_all, default=[])
selected_types = st.sidebar.multiselect("Store type", options=types_all, default=[])
selected_sizes = st.sidebar.multiselect("Store size category", options=sizes_all, default=[])

holiday_filter = st.sidebar.selectbox(
    "Holiday filter",
    options=["All", "Holiday weeks only", "Non-holiday weeks only"],
    index=0
)

season_options = sorted(sales["Season"].dropna().unique().tolist()) if "Season" in sales.columns else []
selected_seasons = st.sidebar.multiselect("Season", options=season_options, default=[])

markdown_filter = st.sidebar.selectbox(
    "Markdown / promotion filter",
    options=["All", "Markdown active", "No markdown"],
    index=0,
    help="Available from scenario3_cleaned(4).csv."
)

error_threshold_pct = st.sidebar.slider(
    "Acceptable forecast deviation threshold",
    min_value=5,
    max_value=50,
    value=20,
    step=5,
    help="Used for red/yellow/green forecast status and recommendation alerts."
)

top_n = st.sidebar.slider("Top N for ranking charts", min_value=5, max_value=25, value=10, step=1)


# -----------------------------
# Apply filters
# -----------------------------
filtered_pred = filter_dataframe(
    predictions,
    start_date,
    end_date,
    selected_stores,
    selected_depts,
    selected_types,
    selected_sizes,
    holiday_filter,
    seasons=None,
    markdown_filter=None,
)

filtered_sales = filter_dataframe(
    sales,
    start_date,
    end_date,
    selected_stores,
    selected_depts,
    selected_types,
    selected_sizes,
    holiday_filter,
    seasons=selected_seasons,
    markdown_filter=markdown_filter,
)

filtered_pred = add_prediction_fields(filtered_pred, prediction_col, error_threshold_pct)

if filtered_pred.empty:
    st.warning("No prediction records match the current filters. Please widen the date range or remove some filters.")
    st.stop()


# -----------------------------
# KPI cards
# -----------------------------
actual_col = "Weekly_Sales_NonNegative"

total_actual = filtered_pred[actual_col].sum()
total_forecast = filtered_pred["Selected_Prediction"].sum()
mae = filtered_pred["Absolute_Error"].mean()
rmse = np.sqrt(np.mean(filtered_pred["Prediction_Error"] ** 2))
filtered_r2 = safe_r2(filtered_pred[actual_col], filtered_pred["Selected_Prediction"])
avg_pct_error = filtered_pred["Percentage_Error"].mean()
forecast_bias = filtered_pred["Prediction_Error"].mean()
high_demand_count = int(filtered_pred["High_Demand_Alert"].sum())
risk_count = int((filtered_pred["Forecast_Status"] == "Risk").sum())

eval_row = get_model_row(evaluation, model_label)
overall_r2 = eval_row.get("R2", np.nan) if not eval_row.empty else np.nan

st.markdown("### Executive KPI Overview")
kpi_cols = st.columns(5)
with kpi_cols[0]:
    render_kpi("Actual Weekly Sales", money(total_actual), "Filtered total actual sales")
with kpi_cols[1]:
    render_kpi("Forecasted Sales", money(total_forecast), f"Selected model: {model_label}")
with kpi_cols[2]:
    render_kpi("Forecast Deviation", money(mae), "MAE: lower is better")
with kpi_cols[3]:
    render_kpi("Forecast Credibility", percent(filtered_r2, 2), "Filtered R² shown in business terms")
with kpi_cols[4]:
    render_kpi("Alert Records", number(risk_count), f">{error_threshold_pct}% deviation risk records")


# -----------------------------
# Main layout: side alerts + main visualization
# -----------------------------
left_col, main_col = st.columns([1.05, 2.25], gap="large")

with left_col:
    st.subheader("Business Alerts")
    mean_actual = filtered_pred[actual_col].mean()
    relative_mae = mae / mean_actual if mean_actual > 0 else np.nan

    if pd.isna(relative_mae) or relative_mae > 0.20:
        render_alert(
            "🔴 Forecast Risk",
            f"Average forecast deviation is {percent(relative_mae)} of average weekly sales. Review high-error stores and departments.",
            "red"
        )
    elif relative_mae > 0.10:
        render_alert(
            "🟡 Forecast Watch",
            f"Average forecast deviation is {percent(relative_mae)}. The forecast is usable, but high-error cases need attention.",
            "yellow"
        )
    else:
        render_alert(
            "🟢 Forecast Stable",
            f"Average forecast deviation is {percent(relative_mae)}. Forecast quality is suitable for routine planning.",
            "green"
        )

    if high_demand_count > 0:
        status = "red" if high_demand_count > max(10, len(filtered_pred) * 0.05) else "yellow"
        render_alert(
            "Demand / Stockout Watch",
            f"{high_demand_count:,} records show high forecasted demand compared with similar Store-Dept periods.",
            status
        )
    else:
        render_alert(
            "Demand Pressure",
            "No strong high-demand spike is detected under the current filters.",
            "green"
        )

    if pd.notna(forecast_bias):
        if forecast_bias > 0:
            render_alert(
                "Over-Forecast Bias",
                f"The selected model over-forecasts by {money(forecast_bias)} per record on average. Avoid unnecessary over-ordering.",
                "yellow"
            )
        elif forecast_bias < 0:
            render_alert(
                "Under-Forecast Bias",
                f"The selected model under-forecasts by {money(abs(forecast_bias))} per record on average. Monitor stockout risk.",
                "yellow"
            )
        else:
            render_alert(
                "Forecast Bias",
                "The selected model has almost no average forecast bias.",
                "green"
            )

    st.subheader("Secondary Chart")
    rank_metric = st.radio(
        "Ranking view",
        options=["Top stores by actual sales", "Top departments by forecast error"],
        horizontal=False
    )

    if rank_metric == "Top stores by actual sales" and "Store" in filtered_pred.columns:
        rank_df = (
            filtered_pred.groupby("Store", as_index=False)[actual_col]
            .sum()
            .sort_values(actual_col, ascending=False)
            .head(top_n)
        )
        fig_rank = px.bar(
            rank_df,
            x=actual_col,
            y="Store",
            orientation="h",
            labels={actual_col: "Actual sales", "Store": "Store"},
            title="Top Stores by Actual Sales"
        )
        fig_rank.update_layout(yaxis={"categoryorder": "total ascending"}, height=360)
        st.plotly_chart(fig_rank, use_container_width=True)
    else:
        rank_df = (
            filtered_pred.groupby("Dept", as_index=False)["Absolute_Error"]
            .mean()
            .sort_values("Absolute_Error", ascending=False)
            .head(top_n)
        )
        fig_rank = px.bar(
            rank_df,
            x="Absolute_Error",
            y="Dept",
            orientation="h",
            labels={"Absolute_Error": "Mean absolute error", "Dept": "Department"},
            title="Departments with Highest Forecast Error"
        )
        fig_rank.update_layout(yaxis={"categoryorder": "total ascending"}, height=360)
        st.plotly_chart(fig_rank, use_container_width=True)

with main_col:
    st.subheader("Main Visualization: Actual Sales vs Forecasted Sales")
    st.markdown(
        "<div class='section-note'>Interactive time-series chart with zoom, pan, hover details, and date-range slider.</div>",
        unsafe_allow_html=True
    )

    time_df = (
        filtered_pred.groupby("Date", as_index=False)
        .agg(
            Actual_Sales=(actual_col, "sum"),
            Forecasted_Sales=("Selected_Prediction", "sum"),
            Absolute_Error=("Absolute_Error", "mean")
        )
        .sort_values("Date")
    )

    fig_time = go.Figure()
    fig_time.add_trace(
        go.Scatter(
            x=time_df["Date"],
            y=time_df["Actual_Sales"],
            mode="lines+markers",
            name="Actual Sales",
            line=dict(color=COLOR_ACTUAL, width=3),
            hovertemplate="Date=%{x}<br>Actual Sales=%{y:$,.0f}<extra></extra>"
        )
    )
    fig_time.add_trace(
        go.Scatter(
            x=time_df["Date"],
            y=time_df["Forecasted_Sales"],
            mode="lines+markers",
            name="Forecasted Sales",
            line=dict(color=COLOR_FORECAST, width=3, dash="dash"),
            hovertemplate="Date=%{x}<br>Forecasted Sales=%{y:$,.0f}<extra></extra>"
        )
    )
    fig_time.update_layout(
        height=520,
        margin=dict(l=10, r=10, t=35, b=10),
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
        yaxis_title="Weekly Sales",
        xaxis_title="Date"
    )
    fig_time.update_xaxes(rangeslider_visible=True)
    st.plotly_chart(fig_time, use_container_width=True)


# -----------------------------
# Prediction vs actual + feature importance
# -----------------------------
st.markdown("---")
pred_col, fi_col = st.columns([1.65, 1.0], gap="large")

with pred_col:
    st.subheader("Model Predictions: Forecast vs Actual")
    st.markdown(
        "<div class='section-note'>Each point represents a Store-Dept weekly record. Red/orange/green labels show deviation status.</div>",
        unsafe_allow_html=True
    )

    scatter_df = filtered_pred.copy()
    if len(scatter_df) > 20000:
        scatter_df = scatter_df.sample(20000, random_state=42)

    fig_scatter = px.scatter(
        scatter_df,
        x=actual_col,
        y="Selected_Prediction",
        color="Forecast_Status",
        hover_data=[c for c in ["Store", "Dept", "Date", "Type", "Store_Size_Category", "Holiday_Week", "Percentage_Error"] if c in scatter_df.columns],
        labels={
            actual_col: "Actual weekly sales",
            "Selected_Prediction": "Forecasted weekly sales",
            "Forecast_Status": "Status",
        },
        title="Prediction Accuracy by Store-Department Week",
        color_discrete_map={
            "Good": COLOR_GOOD,
            "Watch": COLOR_WARNING,
            "Risk": COLOR_RISK,
        }
    )

    max_axis = float(max(scatter_df[actual_col].max(), scatter_df["Selected_Prediction"].max()))
    fig_scatter.add_trace(
        go.Scatter(
            x=[0, max_axis],
            y=[0, max_axis],
            mode="lines",
            name="Perfect prediction",
            line=dict(color=COLOR_NEUTRAL, dash="dot"),
            hoverinfo="skip"
        )
    )
    fig_scatter.update_layout(height=520, margin=dict(l=10, r=10, t=45, b=10))
    st.plotly_chart(fig_scatter, use_container_width=True)

with fi_col:
    st.subheader("Feature Importance")
    st.markdown(
        "<div class='section-note'>Top features explaining model behaviour. The uploaded file is Random Forest feature importance.</div>",
        unsafe_allow_html=True
    )

    if not feature_importance.empty and {"Feature", "Importance"}.issubset(feature_importance.columns):
        fi = feature_importance.copy()
        fi["Readable_Feature"] = fi["Feature"].apply(clean_feature_name)
        fi = fi.sort_values("Importance", ascending=False).head(15).sort_values("Importance", ascending=True)
        fig_fi = px.bar(
            fi,
            x="Importance",
            y="Readable_Feature",
            orientation="h",
            labels={"Importance": "Importance", "Readable_Feature": "Feature"},
            title="Top 15 Predictive Features"
        )
        fig_fi.update_layout(height=520, margin=dict(l=10, r=10, t=45, b=10))
        st.plotly_chart(fig_fi, use_container_width=True)
    else:
        st.info("Feature importance file is unavailable or does not contain `Feature` and `Importance` columns.")


# -----------------------------
# Supporting charts: holiday, promotion, store type/size
# -----------------------------
st.markdown("---")
st.subheader("Supporting Business Analysis")

support_1, support_2, support_3 = st.columns(3, gap="large")

with support_1:
    if "Holiday_Week" in filtered_pred.columns:
        holiday_summary = (
            filtered_pred.groupby("Holiday_Week", as_index=False)
            .agg(
                Actual_Sales=(actual_col, "mean"),
                Forecasted_Sales=("Selected_Prediction", "mean")
            )
        )
        holiday_summary["Holiday_Label"] = holiday_summary["Holiday_Week"].map({0: "Non-holiday", 1: "Holiday"})
        holiday_long = holiday_summary.melt(
            id_vars="Holiday_Label",
            value_vars=["Actual_Sales", "Forecasted_Sales"],
            var_name="Metric",
            value_name="Average_Sales"
        )
        fig_holiday = px.bar(
            holiday_long,
            x="Holiday_Label",
            y="Average_Sales",
            color="Metric",
            barmode="group",
            title="Holiday vs Non-Holiday Average Sales",
            labels={"Holiday_Label": "Week type", "Average_Sales": "Average weekly sales"}
        )
        fig_holiday.update_layout(height=380)
        st.plotly_chart(fig_holiday, use_container_width=True)

with support_2:
    if not filtered_sales.empty and "Any_MarkDown" in filtered_sales.columns and actual_col in filtered_sales.columns:
        promo_summary = (
            filtered_sales.groupby("Any_MarkDown", as_index=False)
            .agg(
                Average_Sales=(actual_col, "mean"),
                Records=(actual_col, "size")
            )
        )
        promo_summary["Promotion_Status"] = promo_summary["Any_MarkDown"].map({0: "No markdown", 1: "Markdown active"})
        fig_promo = px.bar(
            promo_summary,
            x="Promotion_Status",
            y="Average_Sales",
            text="Records",
            title="Markdown Promotion Effect",
            labels={"Promotion_Status": "Promotion status", "Average_Sales": "Average weekly sales"}
        )
        fig_promo.update_layout(height=380)
        st.plotly_chart(fig_promo, use_container_width=True)
    else:
        st.info("Promotion chart requires `Any_MarkDown` in scenario3_cleaned(4).csv.")

with support_3:
    group_dimension = st.selectbox(
        "Performance grouping",
        options=[c for c in ["Type", "Store_Size_Category", "Month"] if c in filtered_pred.columns],
        index=0
    )
    perf = (
        filtered_pred.groupby(group_dimension, as_index=False)
        .agg(
            Actual_Sales=(actual_col, "sum"),
            Forecasted_Sales=("Selected_Prediction", "sum"),
            MAE=("Absolute_Error", "mean")
        )
        .sort_values("Actual_Sales", ascending=False)
    )
    perf_long = perf.melt(
        id_vars=group_dimension,
        value_vars=["Actual_Sales", "Forecasted_Sales"],
        var_name="Metric",
        value_name="Sales"
    )
    fig_perf = px.bar(
        perf_long,
        x=group_dimension,
        y="Sales",
        color="Metric",
        barmode="group",
        title=f"Performance by {group_dimension}",
        labels={"Sales": "Sales"}
    )
    fig_perf.update_layout(height=380)
    st.plotly_chart(fig_perf, use_container_width=True)


# -----------------------------
# Recommendations panel
# -----------------------------
st.markdown("---")
st.subheader("Dynamic Recommendations Panel")
st.markdown(
    "<div class='section-note'>Recommendations are generated from the selected filters, forecast deviation, holiday impact, markdown activity, and demand pressure.</div>",
    unsafe_allow_html=True
)

recommendations = make_recommendations(filtered_pred, filtered_sales, prediction_col, error_threshold_pct)

for i, rec in enumerate(recommendations, start=1):
    st.markdown(
        f"""
        <div class="recommendation">
            <b>Recommendation {i}</b><br>{rec}
        </div>
        """,
        unsafe_allow_html=True
    )


# -----------------------------
# Drill-down tables
# -----------------------------
st.markdown("---")
st.subheader("Drill-Down Tables")

tab1, tab2, tab3, tab4 = st.tabs([
    "Store-Dept Detail",
    "Worst Forecast Errors",
    "Model Evaluation",
    "Data Preview"
])

with tab1:
    detail = (
        filtered_pred.groupby(["Store", "Dept"], as_index=False)
        .agg(
            Actual_Sales=(actual_col, "sum"),
            Forecasted_Sales=("Selected_Prediction", "sum"),
            MAE=("Absolute_Error", "mean"),
            Mean_Percentage_Error=("Percentage_Error", "mean"),
            High_Demand_Records=("High_Demand_Alert", "sum"),
            Risk_Records=("Forecast_Status", lambda x: (x == "Risk").sum())
        )
        .sort_values(["Risk_Records", "MAE"], ascending=False)
    )
    st.dataframe(
        detail.style.format({
            "Actual_Sales": "${:,.0f}",
            "Forecasted_Sales": "${:,.0f}",
            "MAE": "${:,.0f}",
            "Mean_Percentage_Error": "{:.1%}",
            "High_Demand_Records": "{:,.0f}",
            "Risk_Records": "{:,.0f}",
        }),
        use_container_width=True,
        height=420
    )

with tab2:
    if not worst_predictions.empty:
        worst = worst_predictions.copy()
        if "Date" in worst.columns:
            worst = filter_dataframe(
                worst,
                start_date,
                end_date,
                selected_stores,
                selected_depts,
                selected_types,
                selected_sizes,
                holiday_filter,
            )
        display_cols = [
            c for c in [
                "Store", "Dept", "Date", "Weekly_Sales_NonNegative", "Type", "Store_Size_Category",
                "Holiday_Week", "Linear_Regression_Prediction", "Random_Forest_Prediction",
                "Linear_Absolute_Error", "RF_Absolute_Error", "Linear_Percentage_Error", "RF_Percentage_Error"
            ]
            if c in worst.columns
        ]
        st.dataframe(worst[display_cols].head(50), use_container_width=True, height=420)
    else:
        st.info("Worst prediction file is unavailable.")

with tab3:
    if not evaluation.empty:
        eval_display = evaluation.copy()
        numeric_cols = [c for c in eval_display.columns if c != "Model"]
        st.dataframe(
            eval_display.style.format({c: "{:,.4f}" for c in numeric_cols}),
            use_container_width=True,
            height=220
        )

        st.caption(
            "Business interpretation: MAE/RMSE show forecast deviation; R² shows forecast credibility; "
            "forecast skill compares performance against the naive baseline."
        )
    else:
        st.info("Model evaluation table is unavailable.")

with tab4:
    st.write("Filtered prediction data sample")
    preview_cols = [
        c for c in [
            "Store", "Dept", "Date", actual_col, "Selected_Prediction", "Prediction_Error",
            "Absolute_Error", "Percentage_Error", "Forecast_Status", "Type",
            "Store_Size_Category", "Holiday_Week", "Month", "WeekOfYear"
        ]
        if c in filtered_pred.columns
    ]
    st.dataframe(filtered_pred[preview_cols].head(500), use_container_width=True, height=420)


# -----------------------------
# Footer
# -----------------------------
st.markdown("---")
st.caption(
    "Dashboard scope: this app uses the uploaded cleaned historical data, test predictions, feature importance, "
    "worst-error records, and model evaluation table. For true next-week ordering, add a future forecast file with "
    "future dates, Store, Dept, and predicted weekly sales."
)
