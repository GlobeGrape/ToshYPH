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

# ── Custom basemap ─────────────────────────────────────────────────────────────
# Leave empty to use OpenStreetMap.
# Set to a tile URL with {z}/{x}/{y} placeholders to use your own tile server,
# e.g. "https://tile.yourdomain.com/map/{z}/{x}/{y}.png"
BASEMAP_TILE_URL = "https://core-renderer-tiles.maps.yandex.net/tiles?l=map&scale=2&x={x}&y={y}&z={z}&lang=ru_RU&maptype=transit&projection=web_mercator"

DOW_NAMES     = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
MONTH_NAMES   = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]
SLOT30_LABELS = [f"{h//2:02d}:{'30' if h%2 else '00'}" for h in range(48)]
HOUR_LABELS   = [f"{h:02d}:00" for h in range(24)]

COLOR_IN  = "#2ecc71"
COLOR_OUT = "#e74c3c"
PALETTE   = px.colors.qualitative.Set2 + px.colors.qualitative.Pastel


# ── Data loading ──────────────────────────────────────────────────────────────

@st.cache_data(show_spinner="Loading aggregated data…")
def load_data() -> dict:
    def rd(f, required=True):
        path = os.path.join(STATS_DIR, f)
        if not os.path.exists(path):
            if required:
                st.error(f"Missing file: {path}\nRun aggregate_cars.py first.")
                st.stop()
            return None
        return pd.read_parquet(path)

    return {
        "cp":         rd("control_points.parquet"),
        "cp_daily":   rd("cp_daily.parquet"),
        "cp_coords":  rd("cp_coords.parquet"),
        "dow":        rd("by_dow.parquet"),
        "month":      rd("by_month.parquet"),
        "hour":       rd("by_hour.parquet"),
        "slot30":     rd("by_30min.parquet"),
        "dow_cp":     rd("by_dow_cp.parquet",    required=False),
        "month_cp":   rd("by_month_cp.parquet",  required=False),
        "hour_cp":    rd("by_hour_cp.parquet",   required=False),
        "slot30_cp":  rd("by_30min_cp.parquet",  required=False),
    }


# ── Helpers ───────────────────────────────────────────────────────────────────

def poly_trend(x: np.ndarray, y: np.ndarray, degree: int = 3) -> np.ndarray:
    if len(y) <= degree:
        return y.copy()
    coeffs = np.polyfit(x, y, degree)
    return np.clip(np.polyval(coeffs, x), 0, None)


def type_label(t) -> str:
    return f"Type {t}"


def add_trend_trace(fig, x_labels, y_vals, color, name_prefix, degree=3):
    trend = poly_trend(np.arange(len(y_vals), dtype=float),
                       np.array(y_vals, dtype=float), degree)
    fig.add_trace(go.Scatter(
        x=x_labels, y=trend,
        name=f"{name_prefix} trend",
        mode="lines",
        line=dict(color=color, width=3, dash="dot"),
        showlegend=True,
    ))


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


def filter_by_cps(df_cp_aware, selected_cps, x_col, direction=None):
    """Filter CP-aware data to selected CPs (and optionally direction), sum avg_counts."""
    filtered = df_cp_aware[df_cp_aware["object_name"].isin(selected_cps)].copy()
    if direction is not None and "direction" in filtered.columns:
        filtered = filtered[filtered["direction"] == direction]
    return (
        filtered.groupby(["car_type", x_col])
                .agg(avg_count=("avg_count", "sum"),
                     total_count=("total_count", "sum"))
                .reset_index()
    )


# ── App setup ─────────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Mashinque GK — Traffic Stats",
    page_icon="🚗",
    layout="wide",
)
st.title("Mashinque GK — Traffic Statistics")

data = load_data()

st.sidebar.header("Options")
view = st.sidebar.radio(
    "View", ["Сводная статистика по ЙПХ", "Детализированная статистика с разбивкой по периодам и ЙПХ"],
    label_visibility="collapsed", key="view_radio",
)
st.sidebar.markdown(f"**{view}**")
show_trend = st.sidebar.checkbox("Show trendline", value=True)
show_table = st.sidebar.checkbox("Show data table", value=True)
min_flow   = st.sidebar.number_input(
    "Min total flow (hide below)", min_value=0, value=10_000, step=1_000,
    help="Control points with total passages below this threshold are hidden everywhere."
)
st.sidebar.divider()


# ══════════════════════════════════════════════════════════════════════════════
# VIEW: CONTROL POINTS
# ══════════════════════════════════════════════════════════════════════════════
if view == "Сводная статистика по ЙПХ":
    cp_df = data["cp"].copy()

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
    cp_wide = cp_wide[cp_wide["total"] >= min_flow].sort_values("total", ascending=False).reset_index(drop=True)

    # Sub-view toggle
    cp_subview = st.radio(
        "Sub-view", ["Charts", "Map"],
        horizontal=True, key="cp_subview", label_visibility="collapsed"
    )
    st.sidebar.subheader("Сводная статистика по ЙПХ")
    cp_names    = cp_wide["object_name"].tolist()
    selected_cp = st.sidebar.selectbox("Select", ["— All —"] + cp_names)
    dir_filter  = st.sidebar.radio(
        "Direction", ["All", "Entrance (kirish)", "Exit (chiqish)"]
    )

    # ── MAP sub-view ──────────────────────────────────────────────────────────
    if cp_subview == "Map":
        coords = data["cp_coords"]
        if coords is None:
            st.warning("cp_coords.parquet not found — re-run aggregate_cars.py.")
        else:
            coords = coords[coords["total"] >= min_flow].copy()
            coords["direction_type"] = "Both directions"
            coords.loc[coords["kirish"]  == 0, "direction_type"] = "Exit only (chiqish)"
            coords.loc[coords["chiqish"] == 0, "direction_type"] = "Entrance only (kirish)"

            color_map = {
                "Both directions":         "#3498db",
                "Entrance only (kirish)":  "#2ecc71",
                "Exit only (chiqish)":     "#e74c3c",
            }

            fig_map = go.Figure()
            for dtype, color in color_map.items():
                sub = coords[coords["direction_type"] == dtype]
                if sub.empty:
                    continue
                fig_map.add_trace(go.Scattermap(
                    lat=sub["actual_lat"],
                    lon=sub["actual_lon"],
                    mode="markers",
                    name=dtype,
                    marker=dict(
                        size=np.log1p(sub["total"]) * 3,
                        color=color, opacity=0.85, sizemin=8,
                    ),
                    text=sub.apply(
                        lambda r: (f"<b>{r['object_name']}</b><br>"
                                   f"Total: {int(r['total']):,}<br>"
                                   f"Entrance: {int(r['kirish']):,} &nbsp; Exit: {int(r['chiqish']):,}"),
                        axis=1,
                    ),
                    hovertemplate="%{text}<extra></extra>",
                ))

            if BASEMAP_TILE_URL:
                map_style  = "white-bg"
                map_layers = [{"below": "traces", "sourcetype": "raster",
                                "source": [BASEMAP_TILE_URL]}]
            else:
                map_style  = "open-street-map"
                map_layers = []

            fig_map.update_layout(
                map=dict(
                    style=map_style,
                    layers=map_layers,
                    center=dict(lat=float(coords["actual_lat"].median()),
                                lon=float(coords["actual_lon"].median())),
                    zoom=10,
                ),
                height=650,
                margin=dict(l=0, r=0, t=0, b=0),
                legend=dict(
                    bgcolor="rgba(20,20,20,0.82)",
                    bordercolor="#555", borderwidth=1,
                    font=dict(color="white", size=13),
                    itemsizing="constant",
                    x=0.01, y=0.99, yanchor="top",
                ),
            )
            st.plotly_chart(fig_map, use_container_width=True)

            if show_table:
                disp = coords[["object_name","actual_lat","actual_lon",
                                "total","kirish","chiqish","direction_type"]].copy()
                disp.columns = ["Control Point","Lat","Lon",
                                 "Total","Entrance","Exit","Type"]
                disp = disp.sort_values("Total", ascending=False)
                st.dataframe(disp, use_container_width=True, hide_index=True)

    # ── CHARTS sub-view ───────────────────────────────────────────────────────
    else:
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
                # Legend at the bottom so it doesn't overlap the top bars
                legend=dict(orientation="h", y=-0.12, yanchor="top", x=0),
                margin=dict(t=50, b=90),
                hovermode="y unified",
            )
            st.plotly_chart(fig, use_container_width=True)

            if show_table:
                st.subheader("Data Table")
                disp = cp_wide[["object_name","kirish","chiqish","total"]].copy()
                disp.columns = ["Control Point","Entrance","Exit","Total"]
                st.dataframe(disp, use_container_width=True, hide_index=True)

        else:
            cp_row = cp_wide[cp_wide["object_name"] == selected_cp].iloc[0]

            m1, m2, m3 = st.columns(3)
            m1.metric("Total",             f"{int(cp_row['total']):,}")
            m2.metric("Entrance (kirish)", f"{int(cp_row['kirish']):,}")
            m3.metric("Exit (chiqish)",    f"{int(cp_row['chiqish']):,}")

            st.subheader("Daily traffic over time")

            cp_ts = data["cp_daily"][data["cp_daily"]["object_name"] == selected_cp].copy()
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
                    main_dash = "solid" if direction == "kirish" else "dash"
                    fig2.add_trace(go.Scatter(
                        x=sub["date_str"], y=sub["count"],
                        name=label, mode="lines",
                        line=dict(color=color, width=1.5, dash=main_dash), opacity=0.55,
                        hovertemplate="%{x|%Y-%m-%d}<br>%{y:,} cars<extra>" + label + "</extra>",
                    ))
                    if show_trend and len(sub) >= 4:
                        trend = poly_trend(np.arange(len(sub), dtype=float),
                                           sub["count"].values, degree=3)
                        fig2.add_trace(go.Scatter(
                            x=sub["date_str"], y=trend,
                            name=f"{label} trend", mode="lines",
                            line=dict(color=color, width=3, dash="dot"),
                        ))

                fig2.update_layout(
                    title=selected_cp,
                    xaxis_title="Date", yaxis_title="Cars per day",
                    height=420,
                    legend=dict(orientation="h", y=1.0, yanchor="bottom", x=0),
                    margin=dict(t=80),
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
elif view == "Детализированная статистика с разбивкой по периодам и ЙПХ":
    all_types = sorted(
        data["dow"]["car_type"].unique().tolist(),
        key=lambda x: str(x).zfill(10),
    )

    st.sidebar.subheader("Детализированная статистика с разбивкой по периодам и ЙПХ")
    selected_types = st.sidebar.multiselect(
        "Select types", options=all_types, default=all_types,
        format_func=type_label,
    )

    # ── Control point filter ──────────────────────────────────────────────────
    st.sidebar.subheader("Filter by Control Points")
    has_cp_data = data["dow_cp"] is not None
    if not has_cp_data:
        st.sidebar.caption("Re-run aggregate_cars.py to enable CP filtering.")
        selected_cps = []
    else:
        if data["cp_coords"] is not None:
            cp_sorted    = (data["cp_coords"][data["cp_coords"]["total"] >= min_flow]
                            .sort_values("total", ascending=False))
            all_cp_names = cp_sorted["object_name"].tolist()
            cp_totals    = dict(zip(cp_sorted["object_name"], cp_sorted["total"]))
        else:
            all_cp_names = sorted(data["dow_cp"]["object_name"].unique().tolist())
            cp_totals    = {}

        def cp_label(name):
            total = cp_totals.get(name, 0)
            return f"{name}  ({total:,})" if total else name

        selected_cps = st.sidebar.multiselect(
            "Control points (empty = all)",
            options=all_cp_names,
            default=[],
            format_func=cp_label,
        )

    st.sidebar.subheader("Time Period")
    period = st.sidebar.radio(
        "Period", ["Day of Week", "Month", "Hour", "30-Minute"],
        label_visibility="collapsed",
    )

    st.sidebar.subheader("Direction")
    dir_option = st.sidebar.radio(
        "Direction",
        ["All", "Entrance (kirish)", "Exit (chiqish)"],
        label_visibility="collapsed",
        key="ct_direction",
    )
    dir_val = {"All": None, "Entrance (kirish)": "kirish", "Exit (chiqish)": "chiqish"}[dir_option]

    if not selected_types:
        st.warning("Select at least one car type from the sidebar.")
        st.stop()

    cfg = {
        "Day of Week": dict(src="dow",   src_cp="dow_cp",   x_col="dayofweek",
                            x_labels=DOW_NAMES,    x_title="Day of Week", y_title="Avg cars / day"),
        "Month":       dict(src="month", src_cp="month_cp", x_col="month",
                            x_labels=MONTH_NAMES,  x_title="Month",       y_title="Avg cars / day"),
        "Hour":        dict(src="hour",  src_cp="hour_cp",  x_col="hour",
                            x_labels=HOUR_LABELS,  x_title="Hour",        y_title="Avg cars / hour"),
        "30-Minute":   dict(src="slot30",src_cp="slot30_cp",x_col="slot30",
                            x_labels=SLOT30_LABELS,x_title="Time slot",   y_title="Avg cars / 30 min"),
    }[period]

    x_col    = cfg["x_col"]
    x_labels = cfg["x_labels"]
    n_x      = len(x_labels)
    x_nums   = list(range(n_x))
    trend_degree = min(3, max(1, n_x - 2))

    # Choose data source: CP-filtered or global, then apply direction
    if selected_cps and has_cp_data and data[cfg["src_cp"]] is not None:
        df_raw = filter_by_cps(data[cfg["src_cp"]], selected_cps, x_col, direction=dir_val)
        cp_label = f" — {len(selected_cps)} CP(s) selected"
    else:
        df_raw = data[cfg["src"]].copy()
        if "direction" in df_raw.columns:
            if dir_val is not None:
                df_raw = df_raw[df_raw["direction"] == dir_val]
            else:
                # "All" — collapse kirish+chiqish into a single row per (car_type, x_col)
                df_raw = (
                    df_raw.groupby(["car_type", x_col])
                          .agg(avg_count=("avg_count", "sum"),
                               total_count=("total_count", "sum"))
                          .reset_index()
                )
        cp_label = ""

    dir_label = {"All": "", "Entrance (kirish)": " — Kirish", "Exit (chiqish)": " — Chiqish"}[dir_option]

    df_raw["car_type"] = df_raw["car_type"].astype(str)

    fig = go.Figure()

    for idx, ct in enumerate(selected_types):
        color  = PALETTE[idx % len(PALETTE)]
        sub    = (df_raw[df_raw["car_type"] == str(ct)]
                  .set_index(x_col)["avg_count"]
                  .reindex(x_nums, fill_value=0.0))
        y_vals = sub.values.astype(float)

        fig.add_trace(go.Bar(
            name=type_label(ct),
            x=x_labels, y=y_vals,
            marker_color=color, opacity=0.75,
            hovertemplate=f"{type_label(ct)}<br>%{{x}}: %{{y:,.1f}}<extra></extra>",
        ))

        if show_trend and np.any(y_vals > 0):
            add_trend_trace(fig, x_labels, y_vals, color,
                            name_prefix=type_label(ct), degree=trend_degree)

    fig.update_layout(
        title=f"Cars by {period} — average{cp_label}{dir_label}",
        barmode="group" if len(selected_types) > 1 else "relative",
        height=500,
        xaxis_title=cfg["x_title"],
        yaxis_title=cfg["y_title"],
        # Legend below the chart so it doesn't crowd the bars
        legend=dict(orientation="h", y=-0.18, yanchor="top", x=0),
        margin=dict(b=110),
        hovermode="x unified",
        bargap=0.15,
    )
    st.plotly_chart(fig, use_container_width=True)

    # Summary total metrics per car type
    total_row = df_raw[df_raw["car_type"].isin([str(t) for t in selected_types])]
    if not total_row.empty:
        cols = st.columns(min(len(selected_types), 6))
        for i, ct in enumerate(selected_types):
            v = total_row[total_row["car_type"] == str(ct)]["total_count"].sum()
            cols[i % 6].metric(type_label(ct), f"{int(v):,}")

    if show_table:
        st.subheader("Data Table — avg per period")
        table_df = build_type_table(df_raw, x_col, x_labels, selected_types)
        st.dataframe(table_df, use_container_width=True, hide_index=True)
