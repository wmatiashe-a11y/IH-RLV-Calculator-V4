from __future__ import annotations

from typing import Any, Dict, Optional
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
    """Safely fetch a numeric value from audit."""
    v = audit.get(key, default)
    try:
        return float(v)
    except Exception:
        return default


def _money(x: float) -> str:
    """Format money as South African Rand with spaced thousands."""
    try:
        n = float(x)
    except Exception:
        return f"R{str(x)}"
    s = f"{n:,.0f}".replace(",", " ")
    return f"R{s}"


def _flatten(d: Dict[str, Any], prefix: str = "") -> Dict[str, Any]:
    """Flatten nested dict into 'a.b.c' keys for easy diffing."""
    out: Dict[str, Any] = {}
    for k, v in (d or {}).items():
        key = f"{prefix}.{k}" if prefix else str(k)
        if isinstance(v, dict):
            out.update(_flatten(v, key))
        else:
            out[key] = v
    return out


def _diff_dicts(new: Dict[str, Any], old: Dict[str, Any]) -> pd.DataFrame:
    """Return a table of changes between two audits."""
    f_new = _flatten(new or {})
    f_old = _flatten(old or {})
    keys = sorted(set(f_new.keys()) | set(f_old.keys()))

    rows = []
    for k in keys:
        if k.startswith("inputs"): continue # Don't diff raw inputs
        prev = f_old.get(k, None)
        cur = f_new.get(k, None)
        if prev != cur:
            rows.append({"Field": k, "Previous": prev, "Current": cur})
    return pd.DataFrame(rows)


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
) -> Dict[str, Any]:
    """
    Calculates the Residual Land Value (RLV).
    The logic: GDV - Total Costs - Target Profit = RLV.
    """
    # 1. Areas
    gross_bulk_m2 = plot_size_m2 * floor_factor
    sellable_area_m2 = gross_bulk_m2 * efficiency

    # 2. Revenue (GDV)
    if include_affordable and affordable_share > 0:
        affordable_area_m2 = sellable_area_m2 * affordable_share
        market_area_m2 = sellable_area_m2 - affordable_area_m2
        gdv = (market_area_m2 * exit_price_psm) + (affordable_area_m2 * affordable_price_psm)
    else:
        affordable_area_m2 = 0.0
        market_area_m2 = sellable_area_m2
        gdv = sellable_area_m2 * exit_price_psm

    # 3. Direct Costs
    build_cost = gross_bulk_m2 * build_cost_psm
    contingency = build_cost * contingency_rate
    escalation = build_cost * escalation_rate
    professional_fees = build_cost * prof_fees_rate
    rates_taxes = gdv * rates_taxes_rate

    base_costs = build_cost + contingency + escalation + professional_fees + rates_taxes
    
    # 4. Indirect Costs
    marketing = gdv * marketing_rate
    finance = (base_costs + marketing) * finance_rate
    total_costs_ex_profit = base_costs + marketing + finance

    # 5. Profit (Target)
    profit = gdv * profit_rate

    # 6. Residual Land Value
    residual_land_value = gdv - total_costs_ex_profit - profit
    
    # 7. Secondary Metric: Profit on Cost (ROC)
    # ROC = Profit / (Total Costs + Land)
    total_project_cost = total_costs_ex_profit + residual_land_value
    profit_on_cost = (profit / total_project_cost) if total_project_cost > 0 else 0

    return {
        "gross_bulk_m2": gross_bulk_m2,
        "sellable_area_m2": sellable_area_m2,
        "market_area_m2": market_area_m2,
        "affordable_area_m2": affordable_area_m2,
        "gdv": gdv,
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
        "inputs": locals(), # Capture inputs for sensitivity runs
    }


# =========================================================
# VISUALIZATIONS
# =========================================================

def render_waterfall(audit: Dict[str, Any]):
    gdv = _num(audit, "gdv")
    rlv = _num(audit, "residual_land_value")
    profit = _num(audit, "profit")
    
    # Group costs for a clean professional look
    items = [
        ("Build Cost", _num(audit, "build_cost")),
        ("Professional Fees", _num(audit, "professional_fees")),
        ("Other Costs (Fin/Mkt/Tax)", _num(audit, "marketing") + _num(audit, "finance") + _num(audit, "rates_taxes") + _num(audit, "contingency") + _num(audit, "escalation")),
        ("Target Profit", profit),
    ]

    labels = ["GDV"] + [item[0] for item in items] + ["Residual Land Value"]
    measures = ["absolute"] + ["relative"] * len(items) + ["total"]
    values = [gdv] + [-item[1] for item in items] + [rlv]

    fig = go.Figure(go.Waterfall(
        orientation="v",
        measure=measures,
        x=labels,
        y=values,
        connector={"line": {"width": 1, "color": "rgb(63, 63, 63)"}},
        decreasing={"marker": {"color": "#ef5350"}},
        increasing={"marker": {"color": "#66bb6a"}},
        totals={"marker": {"color": "#42a5f5"}}
    ))
    fig.update_layout(title="Residual Land Value Waterfall", height=500, margin=dict(t=50, b=20, l=10, r=10))
    st.plotly_chart(fig, use_container_width=True)


def render_sensitivity(inputs: Dict[str, Any]):
    """Generates a heatmap of RLV based on Exit Price vs Build Cost variations."""
    # Create ranges (+/- 10%)
    price_steps = np.linspace(inputs['exit_price_psm'] * 0.9, inputs['exit_price_psm'] * 1.1, 5)
    cost_steps = np.linspace(inputs['build_cost_psm'] * 0.9, inputs['build_cost_psm'] * 1.1, 5)
    
    z_data = []
    for p in price_steps:
        row = []
        for c in cost_steps:
            # Re-run engine with varied params
            sim_inputs = {**inputs, "exit_price_psm": p, "build_cost_psm": c}
            # Clean sim_inputs of the 'inputs' key to avoid recursion
            sim_inputs.pop('inputs', None) 
            res = compute_feasibility(**sim_inputs)
            row.append(res['residual_land_value'])
        z_data.append(row)

    fig = px.imshow(
        z_data,
        x=[f"Build: R{int(c):,}" for c in cost_steps],
        y=[f"Exit: R{int(p):,}" for p in price_steps],
        labels=dict(x="Construction Cost (Bulk)", y="Market Exit (Sellable)", color="RLV"),
        color_continuous_scale="RdYlGn",
        text_auto=".2s"
    )
    fig.update_layout(title="Sensitivity Analysis: Residual Land Value")
    st.plotly_chart(fig, use_container_width=True)


# =========================================================
# STREAMLIT UI
# =========================================================

st.set_page_config(page_title="IH RLV Calculator", layout="wide")
st.title("🏗️ IH Residual Land Value Calculator")
st.caption("Professional Developer Feasibility Logic: GDV → Costs → Profit → Residual")

# 1. SIDEBAR INPUTS
with st.sidebar:
    st.header("1. Bulk & Efficiency")
    plot_m2 = st.number_input("Plot size (m²)", value=1200, step=100)
    far = st.number_input("Floor Factor (FAR)", value=2.5, step=0.1)
    eff = st.slider("Efficiency (Sellable/Bulk)", 0.5, 1.0, 0.85)
    
    st.header("2. Revenue (Exits)")
    exit_p = st.number_input("Market Price (R/m² sellable)", value=45000, step=500)
    build_p = st.number_input("Build Cost (R/m² bulk)", value=18500, step=500)
    
    st.header("3. Inclusionary Housing")
    inc_on = st.checkbox("Apply IH Policy", value=True)
    inc_share = st.slider("Affordable Share (% of Area)", 0.0, 0.4, 0.1)
    inc_p = st.number_input("Affordable Exit (R/m²)", value=12000)

    st.header("4. Rates & Profit")
    profit_t = st.slider("Target Profit (% of GDV)", 0.0, 0.3, 0.15)
    fin_r = st.slider("Finance Proxy (% of costs)", 0.0, 0.25, 0.10)
    mkt_r = st.slider("Marketing (% of GDV)", 0.0, 0.1, 0.03)
    fees_r = st.slider("Prof Fees (% of build)", 0.0, 0.2, 0.10)
    cont_r = st.slider("Contingency (% of build)", 0.0, 0.1, 0.05)

# 2. COMPUTE
audit = compute_feasibility(
    plot_size_m2=plot_m2, floor_factor=far, efficiency=eff,
    exit_price_psm=exit_p, build_cost_psm=build_p,
    include_affordable=inc_on, affordable_share=inc_share, affordable_price_psm=inc_p,
    contingency_rate=cont_r, escalation_rate=0.03, prof_fees_rate=fees_r,
    rates_taxes_rate=0.02, marketing_rate=mkt_r, finance_rate=fin_r, profit_rate=profit_t
)

# 3. TOP LEVEL METRICS
m1, m2, m3, m4 = st.columns(4)
m1.metric("GDV", _money(audit['gdv']))
m2.metric("Residual Land Value", _money(audit['residual_land_value']), 
          help="The maximum price you can pay for the land to hit your profit target.")
m3.metric("Profit Target", _money(audit['profit']))
m4.metric("Profit on Cost (ROC)", f"{audit['profit_on_cost']*100:.1f}%")

st.divider()

# 4. TABS FOR DETAILED VIEWS
tab_main, tab_sens, tab_audit = st.tabs(["📊 Main Feasibility", "📉 Sensitivity", "🧾 Audit Trail"])

with tab_main:
    render_waterfall(audit)

with tab_sens:
    st.subheader("Land Value Sensitivity")
    st.info("How the Residual Land Value changes if your Market Exit Price or Build Costs vary by +/- 10%.")
    render_sensitivity(audit['inputs'])

with tab_audit:
    # Handle Session State for Diffing
    if "prev_audit" in st.session_state:
        diff_df = _diff_dicts(audit, st.session_state.prev_audit)
        if not diff_df.empty:
            st.markdown("#### Changes Since Last Update")
            st.dataframe(diff_df, use_container_width=True, hide_index=True)
            st.divider()

    st.markdown("#### Detailed Audit Breakdown")
    
    # Map raw keys to pretty labels
    LABELS: Dict[str, str] = {
        "gross_bulk_m2": "Gross Bulk (m²)",
        "sellable_area_m2": "Total Sellable (m²)",
        "market_area_m2": "Market Area (m²)",
        "affordable_area_m2": "Affordable Area (m²)",
        "gdv": "Gross Development Value (GDV)",
        "build_cost": "Raw Construction Cost",
        "contingency": "Contingency",
        "professional_fees": "Professional Fees",
        "marketing": "Marketing & Sales",
        "finance": "Finance Costs (Proxy)",
        "profit": "Target Profit",
        "residual_land_value": "Residual Land Value",
        "profit_on_cost": "Return on Cost (ROC)"
    }

    audit_rows = []
    for k, v in audit.items():
        if k == "inputs": continue
        label = LABELS.get(k, k.replace("_", " ").title())
        val_str = _money(v) if "cost" in k or "value" in k or "gdv" in k or "profit" in k or "fees" in k or "marketing" in k or "finance" in k or "base" in k or "taxes" in k or "contingency" in k else str(v)
        if "area" in k or "bulk" in k: val_str = f"{v:,.0f} m²"
        if "on_cost" in k: val_str = f"{v*100:.2%}"
        
        audit_rows.append({"Item": label, "Value": val_str})

    st.table(pd.DataFrame(audit_rows))
    
    st.download_button(
        "Download Full Audit (JSON)",
        data=json.dumps(audit, indent=2, default=str),
        file_name="feasibility_audit.json",
        mime="application/json"
    )

# Save current audit for next run diff
st.session_state.prev_audit = audit
