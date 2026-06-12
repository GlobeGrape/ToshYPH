"""
Mashinque GK — Traffic Statistics Dashboard
Run:  streamlit run dashboard.py
"""
import os
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import streamlit as st

STATS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "stats")

DOW_NAMES   = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
MONTH_NAMES = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]
SLOT30_LABELS = [f"{h//2:02d}:{'30' if h%2 else '00'}" for h in range(48)]
HOUR_LABELS   = [f"{h:02d}:00" for h in range(24)]

COLOR_IN    = "#2ecc71"
COLOR_OUT   = "#e74c3c"
PALETTE     = px.colors.qualitative.Set2 + px.colors.qualitative.Pastel


# ── Helpers ───────────────────────────────────────────────────────────────────

@st.cache_data(show_spinner="Loading aggregated data…")
def load_data() -> dict:
    def rd(f):
        path = os.path.join(STATS_DIR, f)
        if not os.path.exists(path):
            st.error(f"Missing file: {path}\nRun aggregate_cars.py first.")
            st.stop()
        return pd.read_parquet(path)

    return {
        "cp":       rd("control_points.parquet"),
        "cp_daily": rd("cp_daily.parquet"),
        "dow":      rd("by_dow.parquet"),
        "month":    rd("by_month.parquet"),
        "hour":     rd("by_hour.parquet"),
        "slot30":   rd("by_30min.parquet"),
    }


def poly_trend(x: np.ndarray, y: np.ndarray, degree: int = 3) -> np.ndarray:
    if len(y) <= degree:
        return y.copy()
    coeffs = np.polyfit(x, y, degree)
    trend  = np.polyval(coeffs, x)
    return np.clip(trend, 0, None)


def type_label(t) -> str:
    return f"Type {t}"


def add_trend_trace(fig, x_labels, y_vals, color, name_prefix, degree=3):
    x_num = np.arange(len(y_vals), dtype=float)
    trend  = poly_trend(x_num, np.array(y_vals, dtype=float), degree)
    fig.add_trace(go.Scatter(
        x=x_labels, y=trend,
        name=f"{name_prefix} trend",
        mode="lines",
        line=dict(color=color, width=3, dash="dot"),
        showlegend=True,
    ))


# ── Build car-type overview table ─────────────────────────────────────────────

def build_type_table(df_raw, x_col, x_labels, selected_types) -> pd.DataFrame:
    rows = []
    for ct in selected_types:
        sub = (df_raw[df_raw["car_type"].astype(str) == str(ct)]
               .set_index(x_col)["avg_count"]
               .reindex(range(len(x_labels)), fill_value=0.0))
        row = {"Car Type": type_label(ct)}
        for j, lbl in enumerate(x_labels):
            row[lbl] = round(float(sub.iloc[j]), 1)
        rows.append(row)
    return pd.DataFrame(rows)


# ── App ───────────────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Mashinque GK — Traffic Stats",
    page_icon="🚗",
    layout="wide",
)

st.title("Mashinque GK — Traffic Statistics")

data = load_data()

# Sidebar — global options
st.sidebar.header("Options")
view       = st.sidebar.radio("View", ["Control Points", "Car Types"], label_visibility="collapsed",
                               key="view_radio",
                               help="Choose what to explore")
st.sidebar.markdown(f"**{'Control Points' if view == 'Control Points' else 'Car Types'}**")
show_trend = st.sidebar.checkbox("Show trendline", value=True)
show_table = st.sidebar.checkbox("Show data table", value=True)
st.sidebar.divider()


# ══════════════════════════════════════════════════════════════════════════════
# VIEW: CONTROL POINTS
# ══════════════════════════════════════════════════════════════════════════════
if view == "Control Points":
    cp_df = data["cp"].copy()

    # Pivot to wide format
    cp_wide = (
        cp_df.pivot_table(index="object_name", columns="direction",
                          values="count", aggfunc="sum")
             .reset_index()
    )
    cp_wide.columns.name = None
    for col in ("kirish", "chiqish"):
        if col not in cp_wide.columns:
            cp_wide[col] = 0
    cp_wide = cp_wide.fillna(0)
    cp_wide["kirish"]  = cp_wide["kirish"].astype(int)
    cp_wide["chiqish"] = cp_wide["chiqish"].astype(int)
    cp_wide["total"]   = cp_wide["kirish"] + cp_wide["chiqish"]
    cp_wide = cp_wide.sort_values("total", ascending=False).reset_index(drop=True)

    # Sidebar controls
    st.sidebar.subheader("Control Point")
    cp_names    = cp_wide["object_name"].tolist()
    selected_cp = st.sidebar.selectbox("Select", ["— All —"] + cp_names)
    dir_filter  = st.sidebar.radio(
        "Direction", ["All", "Entrance (kirish)", "Exit (chiqish)"]
    )

    # ── All control points overview ───────────────────────────────────────────
    if selected_cp == "— All —":
        top_n   = st.sidebar.slider("Show top N", 5, len(cp_wide), min(30, len(cp_wide)))
        df_plot = cp_wide.head(top_n)

        fig = go.Figure()

        show_in  = dir_filter in ("All", "Entrance (kirish)")
        show_out = dir_filter in ("All", "Exit (chiqish)")

        if show_in:
            fig.add_trace(go.Bar(
                name="Entrance (kirish)",
                y=df_plot["object_name"], x=df_plot["kirish"],
                orientation="h", marker_color=COLOR_IN, opacity=0.85,
                hovertemplate="%{y}<br>Entrance: %{x:,}<extra></extra>",
            ))
        if show_out:
            fig.add_trace(go.Bar(
                name="Exit (chiqish)",
                y=df_plot["object_name"], x=df_plot["chiqish"],
                orientation="h", marker_color=COLOR_OUT, opacity=0.85,
                hovertemplate="%{y}<br>Exit: %{x:,}<extra></extra>",
            ))

        fig.update_layout(
            title=f"Cars by Control Point — top {top_n}",
            barmode="stack" if dir_filter == "All" else "relative",
            height=max(400, top_n * 24),
            xaxis_title="Total car passages",
            yaxis=dict(autorange="reversed"),
            legend=dict(orientation="h", y=1.02, x=0),
            hovermode="y unified",
        )
        st.plotly_chart(fig, use_container_width=True)

        if show_trend:
            st.caption(
                "ℹ Trendline is not shown for the overview (categorical axis). "
                "Select a specific control point to see its daily trend."
            )

        if show_table:
            st.subheader("Data Table")
            display_df = cp_wide[["object_name", "kirish", "chiqish", "total"]].copy()
            display_df.columns = ["Control Point", "Entrance", "Exit", "Total"]
            st.dataframe(display_df, use_container_width=True, hide_index=True)

    # ── Single control point drill-down ───────────────────────────────────────
    else:
        cp_row = cp_wide[cp_wide["object_name"] == selected_cp].iloc[0]

        m1, m2, m3 = st.columns(3)
        m1.metric("Total",            f"{int(cp_row['total']):,}")
        m2.metric("Entrance (kirish)", f"{int(cp_row['kirish']):,}")
        m3.metric("Exit (chiqish)",    f"{int(cp_row['chiqish']):,}")

        st.subheader("Daily traffic over time")

        cp_ts = (data["cp_daily"][data["cp_daily"]["object_name"] == selected_cp]
                 .copy())
        cp_ts["date_str"] = pd.to_datetime(cp_ts["date_str"])
        cp_ts = cp_ts.sort_values("date_str")

        if cp_ts.empty:
            st.info("No daily data available for this control point.")
        else:
            fig2 = go.Figure()

            pairs = []
            if dir_filter in ("All", "Entrance (kirish)"):
                pairs.append(("kirish",  COLOR_IN,  "Entrance"))
            if dir_filter in ("All", "Exit (chiqish)"):
                pairs.append(("chiqish", COLOR_OUT, "Exit"))

            for direction, color, label in pairs:
                sub = cp_ts[cp_ts["direction"] == direction].sort_values("date_str")
                if sub.empty:
                    continue

                fig2.add_trace(go.Scatter(
                    x=sub["date_str"], y=sub["count"],
                    name=label, mode="lines",
                    line=dict(color=color, width=1.5),
                    opacity=0.55,
                    hovertemplate="%{x|%Y-%m-%d}<br>%{y:,} cars<extra>" + label + "</extra>",
                ))

                if show_trend and len(sub) >= 4:
                    x_num = np.arange(len(sub), dtype=float)
                    trend  = poly_trend(x_num, sub["count"].values, degree=3)
                    fig2.add_trace(go.Scatter(
                        x=sub["date_str"], y=trend,
                        name=f"{label} trend",
                        mode="lines",
                        line=dict(color=color, width=3, dash="dash"),
                    ))

            fig2.update_layout(
                title=selected_cp,
                xaxis_title="Date", yaxis_title="Cars per day",
                height=420,
                legend=dict(orientation="h", y=1.02, x=0),
                hovermode="x unified",
            )
            st.plotly_chart(fig2, use_container_width=True)

            if show_table:
                pivot = (cp_ts.pivot_table(
                             index="date_str", columns="direction",
                             values="count", aggfunc="sum")
                             .reset_index())
                pivot.columns.name = None
                pivot["date_str"] = pivot["date_str"].dt.strftime("%Y-%m-%d")
                st.dataframe(pivot, use_container_width=True, hide_index=True)


# ══════════════════════════════════════════════════════════════════════════════
# VIEW: CAR TYPES
# ══════════════════════════════════════════════════════════════════════════════
elif view == "Car Types":
    all_types = sorted(
        data["dow"]["car_type"].unique().tolist(),
        key=lambda x: (str(x).zfill(10))
    )

    st.sidebar.subheader("Car Types")
    selected_types = st.sidebar.multiselect(
        "Select types",
        options=all_types,
        default=all_types,
        format_func=type_label,
    )

    st.sidebar.subheader("Time Period")
    period = st.sidebar.radio(
        "Period",
        ["Day of Week", "Month", "Hour", "30-Minute"],
        label_visibility="collapsed",
    )

    if not selected_types:
        st.warning("Select at least one car type from the sidebar.")
        st.stop()

    # Map period → source data + axis config
    cfg = {
        "Day of Week": dict(
            src="dow",   x_col="dayofweek", x_labels=DOW_NAMES,
            x_title="Day of Week",  y_title="Avg cars / day",
        ),
        "Month": dict(
            src="month", x_col="month",     x_labels=MONTH_NAMES,
            x_title="Month",        y_title="Avg cars / day",
        ),
        "Hour": dict(
            src="hour",  x_col="hour",      x_labels=HOUR_LABELS,
            x_title="Hour",         y_title="Avg cars / hour",
        ),
        "30-Minute": dict(
            src="slot30",x_col="slot30",    x_labels=SLOT30_LABELS,
            x_title="Time slot",    y_title="Avg cars / 30 min",
        ),
    }[period]

    df_raw = data[cfg["src"]].copy()
    df_raw["car_type"] = df_raw["car_type"].astype(str)
    x_col    = cfg["x_col"]
    x_labels = cfg["x_labels"]
    n_x      = len(x_labels)
    x_nums   = list(range(n_x))

    # Degree for polynomial trend — lower for fewer points
    trend_degree = min(3, max(1, n_x - 2))

    fig = go.Figure()

    for idx, ct in enumerate(selected_types):
        color = PALETTE[idx % len(PALETTE)]
        sub   = (df_raw[df_raw["car_type"] == str(ct)]
                 .set_index(x_col)["avg_count"]
                 .reindex(x_nums, fill_value=0.0))
        y_vals = sub.values.astype(float)

        fig.add_trace(go.Bar(
            name=type_label(ct),
            x=x_labels, y=y_vals,
            marker_color=color,
            opacity=0.75,
            hovertemplate=f"{type_label(ct)}<br>%{{x}}: %{{y:,.1f}}<extra></extra>",
        ))

        if show_trend and np.any(y_vals > 0):
            add_trend_trace(
                fig, x_labels, y_vals, color,
                name_prefix=type_label(ct),
                degree=trend_degree,
            )

    fig.update_layout(
        title=f"Cars by {period} — average",
        barmode="group" if len(selected_types) > 1 else "relative",
        height=500,
        xaxis_title=cfg["x_title"],
        yaxis_title=cfg["y_title"],
        legend=dict(orientation="h", y=1.05, x=0, xanchor="left"),
        hovermode="x unified",
        bargap=0.15,
    )
    st.plotly_chart(fig, use_container_width=True)

    # Summary metrics
    total_row = df_raw[df_raw["car_type"].isin([str(t) for t in selected_types])]
    if not total_row.empty:
        cols = st.columns(len(selected_types))
        for i, ct in enumerate(selected_types):
            sub_total = total_row[total_row["car_type"] == str(ct)]["total_count"].sum()
            cols[i].metric(type_label(ct), f"{int(sub_total):,}")

    if show_table:
        st.subheader("Data Table — avg per period")
        table_df = build_type_table(df_raw, x_col, x_labels, selected_types)
        st.dataframe(table_df, use_container_width=True, hide_index=True)
