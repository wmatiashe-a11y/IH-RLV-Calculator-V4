from __future__ import annotations

from typing import Any, Dict
import json

import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import numpy as np

# =========================================================
# HELPERS
# =========================================================

def _num(audit: Dict[str, Any], key: str, default: float = 0.0) -> float:
    v = audit.get(key, default)
    try:
        return float(v)
    except Exception:
        return default


def _money(x: float) -> str:
    try:
        n = float(x)
    except Exception:
        return f"R{str(x)}"
    s = f"{n:,.0f}".replace(",", " ")
    return f"R{s}"


def _flatten(d: Dict[str, Any], prefix: str = "") -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for k, v in (d or {}).items():
        key = f"{prefix}.{k}" if prefix else str(k)
        if isinstance(v, dict):
            out.update(_flatten(v, key))
        else:
            out[key] = v
    return out


def _diff_dicts(new: Dict[str, Any], old: Dict[str, Any]) -> pd.DataFrame:
    f_new = _flatten(new or {})
    f_old = _flatten(old or {})
    keys = sorted(set(f_new.keys()) | set(f_old.keys()))

    rows = []
    for k in keys:
        if k.startswith("inputs"):
            continue
        prev = f_old.get(k, None)
        cur = f_new.get(k, None)
        if prev != cur:
            rows.append({"Field": k, "Previous": prev, "Current": cur})
    return pd.DataFrame(rows)


def _card(label: str, value: str, hint: str = ""):
    # Avoid st.container(border=True) incompatibility by using markdown styling
    st.markdown(
        f"""
        <div style="border:1px solid rgba(255,255,255,0.15); border-radius:14px; padding:14px;">
          <div style="font-weight:700; margin-bottom:6px;">{label}</div>
          <div style="margin-bottom:6px;">{value}</div>
          <div style="opacity:0.7; font-size:0.85rem;">{hint}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

# =========================================================
# 2026 EXIT PRICE DATA (OPTIONAL)
# =========================================================

DEFAULT_EXIT_DATA = pd.DataFrame(
    [
        {
            "suburb": "Sea Point",
            "exit_price_psm": 65000,
            "recent_sales_psm": 62000,
            "bulk_efficiency_note": "Typical net sellable efficiency ranges 82–88% depending on parking/core.",
            "coastal_premium_note": "Coastal/sea-facing premium often supports stronger exit pricing.",
            "wind_cost_uplift": 0.04,
        },
        {
            "suburb": "Woodstock",
            "exit_price_psm": 42000,
            "recent_sales_psm": 39500,
            "bulk_efficiency_note": "Efficiency often constrained by cores/servicing; check parking & setbacks.",
            "coastal_premium_note": "Limited coastal premium; pricing tends to be more block-specific.",
            "wind_cost_uplift": 0.02,
        },
        {
            "suburb": "Claremont",
            "exit_price_psm": 48000,
            "recent_sales_psm": 45500,
            "bulk_efficiency_note": "Good efficiency possible for mid-rise; confirm parking ratios early.",
            "coastal_premium_note": "Inland node premium driven by amenities/transport and product fit.",
            "wind_cost_uplift": 0.01,
        },
    ]
)


@st.cache_data(show_spinner=False)
def load_exit_prices(uploaded_file) -> pd.DataFrame:
    if uploaded_file is None:
        return DEFAULT_EXIT_DATA.copy()

    df = pd.read_csv(uploaded_file)
    df.columns = [c.strip().lower() for c in df.columns]

    if "suburb" not in df.columns:
        raise ValueError("Exit prices CSV must include a 'suburb' column.")

    for col, default in [
        ("exit_price_psm", np.nan),
        ("recent_sales_psm", np.nan),
        ("bulk_efficiency_note", ""),
        ("coastal_premium_note", ""),
        ("wind_cost_uplift", 0.0),
    ]:
        if col not in df.columns:
            df[col] = default

    # Ensure numeric
    for col in ["exit_price_psm", "recent_sales_psm", "wind_cost_uplift"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df["suburb"] = df["suburb"].astype(str)
    return df


def get_suburb_row(df: pd.DataFrame, suburb: str) -> Dict[str, Any]:
    if df is None or df.empty or not suburb:
        return {}
    hit = df[df["suburb"].str.lower() == str(suburb).lower()]
    if hit.empty:
        return {}
    return hit.iloc[0].to_dict()


# =========================================================
# CORE FEASIBILITY ENGINE
# =========================================================

def compute_feasibility(
    plot_size_m2: float,
    floor_factor: float,
    efficiency: float,
    exit_price_psm: float,
    build_cost_psm: float,
    include_affordable: bool,
    affordable_share: float,
    affordable_price_psm: float,
    contingency_rate: float,
    escalation_rate: float,
    prof_fees_rate: float,
    rates_taxes_rate: float,
    marketing_rate: float,
    finance_rate: float,
    profit_rate: float,
    wind_cost_uplift: float = 0.0,
) -> Dict[str, Any]:
    gross_bulk_m2 = plot_size_m2 * floor_factor
    sellable_area_m2 = gross_bulk_m2 * efficiency

    if include_affordable and affordable_share > 0:
        affordable_area_m2 = sellable_area_m2 * affordable_share
        market_area_m2 = sellable_area_m2 - affordable_area_m2
        gdv = (market_area_m2 * exit_price_psm) + (affordable_area_m2 * affordable_price_psm)
    else:
        affordable_area_m2 = 0.0
        market_area_m2 = sellable_area_m2
        gdv = sellable_area_m2 * exit_price_psm

    build_cost_base = gross_bulk_m2 * build_cost_psm
    wind_uplift_cost = build_cost_base * max(0.0, float(wind_cost_uplift or 0.0))
    build_cost = build_cost_base + wind_uplift_cost

    contingency = build_cost * contingency_rate
    escalation = build_cost * escalation_rate
    professional_fees = build_cost * prof_fees_rate
    rates_taxes = gdv * rates_taxes_rate

    base_costs = build_cost + contingency + escalation + professional_fees + rates_taxes

    marketing = gdv * marketing_rate
    finance = (base_costs + marketing) * finance_rate
    total_costs_ex_profit = base_costs + marketing + finance

    profit = gdv * profit_rate

    residual_land_value = gdv - total_costs_ex_profit - profit

    total_project_cost = total_costs_ex_profit + residual_land_value
    profit_on_cost = (profit / total_project_cost) if total_project_cost > 0 else 0

    # IMPORTANT: don't store raw locals() directly (it can include non-serializable objects)
    inputs_for_sensitivity = dict(
        plot_size_m2=plot_size_m2,
        floor_factor=floor_factor,
        efficiency=efficiency,
        exit_price_psm=exit_price_psm,
        build_cost_psm=build_cost_psm,
        include_affordable=include_affordable,
        affordable_share=affordable_share,
        affordable_price_psm=affordable_price_psm,
        contingency_rate=contingency_rate,
        escalation_rate=escalation_rate,
        prof_fees_rate=prof_fees_rate,
        rates_taxes_rate=rates_taxes_rate,
        marketing_rate=marketing_rate,
        finance_rate=finance_rate,
        profit_rate=profit_rate,
        wind_cost_uplift=wind_cost_uplift,
    )

    return {
        "gross_bulk_m2": gross_bulk_m2,
        "sellable_area_m2": sellable_area_m2,
        "market_area_m2": market_area_m2,
        "affordable_area_m2": affordable_area_m2,
        "gdv": gdv,
        "build_cost_base": build_cost_base,
        "wind_uplift_cost": wind_uplift_cost,
        "build_cost": build_cost,
        "contingency": contingency,
        "escalation": escalation,
        "professional_fees": professional_fees,
        "rates_taxes": rates_taxes,
        "base_costs": base_costs,
        "marketing": marketing,
        "finance": finance,
        "total_costs_ex_profit": total_costs_ex_profit,
        "profit": profit,
        "residual_land_value": residual_land_value,
        "profit_on_cost": profit_on_cost,
        "inputs": inputs_for_sensitivity,
    }


# =========================================================
# VISUALIZATIONS
# =========================================================

def render_waterfall(audit: Dict[str, Any]):
    gdv = _num(audit, "gdv")
    rlv = _num(audit, "residual_land_value")
    profit = _num(audit, "profit")

    other_costs = (
        _num(audit, "marketing")
        + _num(audit, "finance")
        + _num(audit, "rates_taxes")
        + _num(audit, "contingency")
        + _num(audit, "escalation")
        + _num(audit, "wind_uplift_cost")
    )

    items = [
        ("Build Cost", _num(audit, "build_cost")),
        ("Professional Fees", _num(audit, "professional_fees")),
        ("Other Costs (Fin/Mkt/Tax/Adj.)", other_costs),
        ("Target Profit", profit),
    ]

    labels = ["GDV"] + [item[0] for item in items] + ["Residual Land Value"]
    measures = ["absolute"] + ["relative"] * len(items) + ["total"]
    values = [gdv] + [-item[1] for item in items] + [rlv]

    fig = go.Figure(
        go.Waterfall(
            orientation="v",
            measure=measures,
            x=labels,
            y=values,
            connector={"line": {"width": 1, "color": "rgb(63, 63, 63)"}},
            decreasing={"marker": {"color": "#ef5350"}},
            increasing={"marker": {"color": "#66bb6a"}},
            totals={"marker": {"color": "#42a5f5"}},
        )
    )
    fig.update_layout(title="Residual Land Value Waterfall", height=500, margin=dict(t=50, b=20, l=10, r=10))
    st.plotly_chart(fig, use_container_width=True)


def render_sensitivity(inputs: Dict[str, Any]):
    price_steps = np.linspace(inputs["exit_price_psm"] * 0.9, inputs["exit_price_psm"] * 1.1, 5)
    cost_steps = np.linspace(inputs["build_cost_psm"] * 0.9, inputs["build_cost_psm"] * 1.1, 5)

    z_data = []
    for p in price_steps:
        row = []
        for c in cost_steps:
            sim_inputs = {**inputs, "exit_price_psm": float(p), "build_cost_psm": float(c)}
            res = compute_feasibility(**sim_inputs)
            row.append(res["residual_land_value"])
        z_data.append(row)

    # Some plotly versions don't support text_auto in imshow -> handle safely
    try:
        fig = px.imshow(
            z_data,
            x=[f"Build: R{int(c):,}".replace(",", " ") for c in cost_steps],
            y=[f"Exit: R{int(p):,}".replace(",", " ") for p in price_steps],
            labels=dict(x="Construction Cost (Bulk)", y="Market Exit (Sellable)", color="RLV"),
            color_continuous_scale="RdYlGn",
            text_auto=".2s",
        )
    except TypeError:
        fig = px.imshow(
            z_data,
            x=[f"Build: R{int(c):,}".replace(",", " ") for c in cost_steps],
            y=[f"Exit: R{int(p):,}".replace(",", " ") for p in price_steps],
            labels=dict(x="Construction Cost (Bulk)", y="Market Exit (Sellable)", color="RLV"),
            color_continuous_scale="RdYlGn",
        )

    fig.update_layout(title="Sensitivity Analysis: Residual Land Value")
    st.plotly_chart(fig, use_container_width=True)


# =========================================================
# STREAMLIT UI
# =========================================================

st.set_page_config(page_title="IH RLV Calculator", layout="wide")

st.title("🏗️ IH Residual Land Value Calculator")
st.caption("Developer Feasibility Logic: GDV → Costs → Profit → Residual")

with st.sidebar:
    st.header("Data (2026 Exits)")
    uploaded_exit = st.file_uploader("Upload Exit Prices CSV", type=["csv"])
    st.caption("CSV columns: suburb, exit_price_psm, recent_sales_psm, bulk_efficiency_note, coastal_premium_note, wind_cost_uplift")

    try:
        exit_df = load_exit_prices(uploaded_exit)
    except Exception as e:
        st.error(str(e))
        exit_df = DEFAULT_EXIT_DATA.copy()

    st.divider()
    st.header("Assumptions")
    escalation_rate = st.slider("Escalation (% of build)", 0.0, 0.15, 0.03)
    rates_taxes_rate = st.slider("Rates/Taxes (% of GDV)", 0.0, 0.10, 0.02)

# Safe suburbs list handling
suburbs = sorted(exit_df["suburb"].astype(str).unique().tolist()) if (exit_df is not None and not exit_df.empty) else []

st.subheader("Property Quick-Profile")

c1, c2, c3, c4, c5 = st.columns([1.2, 1.2, 1.0, 1.0, 1.0])

with c1:
    if suburbs:
        suburb = st.selectbox("Suburb / Node", options=suburbs, index=0)
    else:
        suburb = ""
        st.warning("No suburbs available (exit dataset empty). Upload a CSV or use the defaults.")
    erf_ref = st.text_input("Erf / Address (reference)", value="", placeholder="e.g., Erf 12345, Sea Point")

with c2:
    st.markdown("**City of Cape Town Map Viewer (workflow)**")
    st.caption("Use the Map Viewer to confirm erf boundaries / area, then enter plot size below.")
    map_query = st.text_input("Map Viewer search text", value="", placeholder="Paste erf number or address keywords")
    if map_query.strip():
        st.info("Tip: open the City Map Viewer and search for the same text to confirm plot extent/area.")

row = get_suburb_row(exit_df, suburb)
default_exit = float(row.get("exit_price_psm")) if row and pd.notna(row.get("exit_price_psm")) else 45000.0
default_recent = float(row.get("recent_sales_psm")) if row and pd.notna(row.get("recent_sales_psm")) else np.nan
default_wind_uplift = float(row.get("wind_cost_uplift") or 0.0) if row else 0.0

with c3:
    plot_m2 = st.number_input("Plot size (m²)", value=1200, step=50)
    far = st.number_input("Floor Factor (FAR)", value=2.5, step=0.1)

with c4:
    eff = st.slider("Efficiency (Sellable/Bulk)", 0.50, 1.00, 0.85)
    exit_p = st.number_input("Market Price (R/m² sellable)", value=int(default_exit), step=500)

with c5:
    build_p = st.number_input("Build Cost (R/m² bulk)", value=18500, step=500)
    wind_uplift = st.slider(
        "Coastal/Wind cost uplift",
        0.0, 0.10, float(default_wind_uplift),
        help="Proxy uplift applied to build cost (e.g., glazing / exposure).",
    )

st.divider()

# ---- The Feasibility Lens ----
st.subheader("The Feasibility Lens")
card1, card2, card3 = st.columns(3)

local_comps_txt = (
    f"Recent sales in this sub-zone: <b>{_money(default_recent)}/m²</b>."
    if pd.notna(default_recent)
    else "Recent sales in this sub-zone: <i>(upload CSV to populate comps)</i>."
)

bulk_eff_note = (row.get("bulk_efficiency_note") or "").strip() if row else ""
if not bulk_eff_note:
    bulk_eff_note = "Bulk efficiency note: <i>(upload CSV to add a suburb-specific rule-of-thumb)</i>."

coastal_note = (row.get("coastal_premium_note") or "").strip() if row else ""
if not coastal_note:
    coastal_note = "Coastal premium: <i>(upload CSV to add a suburb-specific note)</i>."

with card1:
    _card("📍 Local Comps", local_comps_txt, hint="Use as a sense-check for your market exit assumption.")

with card2:
    approx_bulk = float(plot_m2) * float(far)
    _card(
        "🏗️ Bulk Efficiency",
        f"You can build ~<b>{approx_bulk:,.0f} m²</b> gross bulk at FAR <b>{float(far):.2f}</b>.<br/><br/>{bulk_eff_note}",
        hint="Bulk is a zoning proxy; confirm with scheme + overlays + parking + setbacks.",
    )

with card3:
    _card(
        "💨 Coastal Premium / Exposure",
        f"High-wind/exposure proxy: <b>+{float(wind_uplift)*100:.1f}%</b> to build cost.<br/><br/>{coastal_note}",
        hint="Proxy only — replace with QS line items when available.",
    )

st.divider()

with st.sidebar:
    st.header("1. Inclusionary Housing")
    inc_on = st.checkbox("Apply IH Policy", value=True)
    inc_share = st.slider("Affordable Share (% of Sellable)", 0.0, 0.4, 0.10)
    inc_p = st.number_input("Affordable Exit (R/m²)", value=12000, step=500)

    st.header("2. Costs & Profit")
    profit_t = st.slider("Target Profit (% of GDV)", 0.0, 0.3, 0.15)
    fin_r = st.slider("Finance Proxy (% of costs)", 0.0, 0.25, 0.10)
    mkt_r = st.slider("Marketing (% of GDV)", 0.0, 0.10, 0.03)
    fees_r = st.slider("Prof Fees (% of build)", 0.0, 0.25, 0.10)
    cont_r = st.slider("Contingency (% of build)", 0.0, 0.15, 0.05)

audit = compute_feasibility(
    plot_size_m2=float(plot_m2),
    floor_factor=float(far),
    efficiency=float(eff),
    exit_price_psm=float(exit_p),
    build_cost_psm=float(build_p),
    include_affordable=bool(inc_on),
    affordable_share=float(inc_share),
    affordable_price_psm=float(inc_p),
    contingency_rate=float(cont_r),
    escalation_rate=float(escalation_rate),
    prof_fees_rate=float(fees_r),
    rates_taxes_rate=float(rates_taxes_rate),
    marketing_rate=float(mkt_r),
    finance_rate=float(fin_r),
    profit_rate=float(profit_t),
    wind_cost_uplift=float(wind_uplift),
)

m1, m2, m3, m4 = st.columns(4)
m1.metric("GDV", _money(audit["gdv"]))
m2.metric("Residual Land Value", _money(audit["residual_land_value"]))
m3.metric("Profit Target", _money(audit["profit"]))
m4.metric("Profit on Cost (ROC)", f"{audit['profit_on_cost']*100:.1f}%")

st.divider()

tab_main, tab_sens, tab_audit = st.tabs(["📊 Main Feasibility", "📉 Sensitivity", "🧾 Audit Trail"])

with tab_main:
    render_waterfall(audit)

with tab_sens:
    st.subheader("Land Value Sensitivity")
    st.info("RLV response if Market Exit or Build Costs vary by +/- 10% (other assumptions held constant).")
    render_sensitivity(audit["inputs"])

with tab_audit:
    if "prev_audit" in st.session_state:
        diff_df = _diff_dicts(audit, st.session_state.prev_audit)
        if not diff_df.empty:
            st.markdown("#### Changes Since Last Update")
            st.dataframe(diff_df, use_container_width=True, hide_index=True)
            st.divider()

    st.markdown("#### Detailed Audit Breakdown")

    LABELS: Dict[str, str] = {
        "gross_bulk_m2": "Gross Bulk (m²)",
        "sellable_area_m2": "Total Sellable (m²)",
        "market_area_m2": "Market Area (m²)",
        "affordable_area_m2": "Affordable Area (m²)",
        "gdv": "Gross Development Value (GDV)",
        "build_cost_base": "Base Construction Cost (pre-uplift)",
        "wind_uplift_cost": "Wind/Exposure Uplift (proxy)",
        "build_cost": "Total Construction Cost",
        "contingency": "Contingency",
        "escalation": "Escalation",
        "professional_fees": "Professional Fees",
        "rates_taxes": "Rates/Taxes",
        "marketing": "Marketing & Sales",
        "finance": "Finance Costs (Proxy)",
        "profit": "Target Profit",
        "residual_land_value": "Residual Land Value",
        "profit_on_cost": "Return on Cost (ROC)",
    }

    audit_rows = []
    for k, v in audit.items():
        if k == "inputs":
            continue
        label = LABELS.get(k, k.replace("_", " ").title())

        if "area" in k or "bulk" in k:
            val_str = f"{float(v):,.0f} m²".replace(",", " ")
        elif "on_cost" in k:
            val_str = f"{float(v)*100:.2f}%"
        elif any(s in k for s in ["gdv", "cost", "value", "profit", "fees", "marketing", "finance", "taxes", "contingency", "escalation", "uplift"]):
            val_str = _money(v)
        else:
            val_str = str(v)

        audit_rows.append({"Item": label, "Value": val_str})

    st.table(pd.DataFrame(audit_rows))

    # Make JSON download robust
    safe_audit = json.loads(json.dumps(audit, default=str))
    st.download_button(
        "Download Full Audit (JSON)",
        data=json.dumps(safe_audit, indent=2),
        file_name="feasibility_audit.json",
        mime="application/json",
    )

st.session_state.prev_audit = audit
