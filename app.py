from __future__ import annotations

from typing import Any, Dict, Optional
import json

import pandas as pd
import streamlit as st
import plotly.graph_objects as go


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
    """Format money as South African Rand with spaced thousands (e.g. R1 234 567)."""
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
        prev = f_old.get(k, None)
        cur = f_new.get(k, None)
        if prev != cur:
            rows.append({"Field": k, "Previous": prev, "Current": cur})
    return pd.DataFrame(rows)


# =========================================================
# WATERFALL (Developer-style)
# =========================================================

def render_audit_waterfall(
    audit: Dict[str, Any],
    *,
    title: str = "📉 Developer Waterfall (GDV → RLV)",
    expanded: bool = True,
    show_key_drivers: bool = True,
) -> None:
    if not audit:
        st.info("No audit data available for waterfall.")
        return

    gdv = _num(audit, "gdv", 0.0)

    build_cost = _num(audit, "build_cost", 0.0)
    contingency = _num(audit, "contingency", 0.0)
    escalation = _num(audit, "escalation", 0.0)
    professional_fees = _num(audit, "professional_fees", 0.0)
    rates_taxes = _num(audit, "rates_taxes", 0.0)

    marketing = _num(audit, "marketing", 0.0)
    finance = _num(audit, "finance", 0.0)
    finance_marketing = _num(audit, "finance_marketing", 0.0)

    profit = _num(audit, "profit", 0.0)
    rlv = _num(audit, "residual_land_value", 0.0)
    acquisition = _num(audit, "acquisition_costs", 0.0)

    # Prefer detailed finance + marketing if present; otherwise use combined legacy key
    use_combined = (marketing == 0.0 and finance == 0.0 and finance_marketing != 0.0)

    cost_items = [
        ("Build cost", build_cost),
        ("Contingency", contingency),
        ("Escalation", escalation),
        ("Professional fees", professional_fees),
        ("Rates & taxes", rates_taxes),
    ]
    if use_combined:
        cost_items.append(("Finance + marketing", finance_marketing))
    else:
        cost_items.extend([("Marketing", marketing), ("Finance", finance)])

    cost_items = [(n, v) for n, v in cost_items if abs(v) > 1e-9]

    computed_rlv = gdv - sum(v for _, v in cost_items) - abs(profit)
    rlv_to_show = rlv if abs(rlv) > 1e-9 else computed_rlv

    with st.expander(title, expanded=expanded):
        if show_key_drivers:
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("GDV", _money(gdv))
            total_costs = _num(audit, "total_costs_ex_profit", sum(v for _, v in cost_items))
            c2.metric("Total costs (ex profit)", _money(total_costs))
            c3.metric("Profit", _money(profit))
            c4.metric("Residual land value", _money(rlv_to_show))

        labels = ["GDV"]
        measures = ["absolute"]
        values = [gdv]

        for name, val in cost_items:
            labels.append(name)
            measures.append("relative")
            values.append(-abs(val))

        if abs(profit) > 1e-9:
            labels.append("Profit")
            measures.append("relative")
            values.append(-abs(profit))

        labels.append("Residual land value")
        measures.append("total")
        values.append(rlv_to_show)

        fig = go.Figure(
            go.Waterfall(
                orientation="v",
                measure=measures,
                x=labels,
                y=values,
                connector={"line": {"width": 1}},
            )
        )
        fig.update_layout(
            margin=dict(l=10, r=10, t=20, b=10),
            height=420,
            yaxis_title="ZAR",
            showlegend=False,
        )
        st.plotly_chart(fig, width="stretch")

        if abs(acquisition) > 1e-9:
            st.markdown("#### Acquisition impact (post-RLV)")
            fig2 = go.Figure(
                go.Waterfall(
                    orientation="v",
                    measure=["absolute", "relative", "total"],
                    x=["Residual land value", "Acquisition costs", "Net land (after acquisition)"],
                    y=[rlv_to_show, -abs(acquisition), rlv_to_show - abs(acquisition)],
                    connector={"line": {"width": 1}},
                )
            )
            fig2.update_layout(
                margin=dict(l=10, r=10, t=10, b=10),
                height=320,
                yaxis_title="ZAR",
                showlegend=False,
            )
            st.plotly_chart(fig2, width="stretch")


# =========================================================
# AUDIT TABLE (Clean + Diff + Download)
# =========================================================

def render_audit(
    audit: Dict[str, Any],
    *,
    title: str = "🧾 Audit Trail",
    expanded: bool = False,
    previous_audit: Optional[Dict[str, Any]] = None,
    show_raw_json: bool = True,
) -> None:
    if not audit:
        st.info("No audit data available for this run.")
        return

    money_key_set = {
        "gdv", "build_cost", "contingency", "escalation", "professional_fees", "rates_taxes",
        "base_costs", "marketing", "finance", "finance_marketing", "total_costs_ex_profit",
        "profit_base", "profit", "residual_land_value", "acquisition_costs",
    }
    area_unit_key_set = {
        "gross_bulk_m2", "sellable_area_m2", "market_area_m2", "affordable_area_m2", "units_estimate"
    }

    money_count = sum(1 for k in audit.keys() if k in money_key_set)
    area_count = sum(1 for k in audit.keys() if k in area_unit_key_set)
    other_count = max(0, len(audit) - money_count - area_count)

    with st.expander(title, expanded=expanded):
        st.markdown("#### Summary")

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Fields", len(audit))
        col2.metric("Money items", money_count)
        col3.metric("Areas / Units", area_count)
        col4.metric("Other", other_count)

        st.divider()

        try:
            audit_json = json.dumps(audit, indent=2, default=str).encode("utf-8")
            st.download_button(
                "⬇️ Download audit (JSON)",
                data=audit_json,
                file_name="audit_trail.json",
                mime="application/json",
                width="stretch",
            )
        except Exception:
            st.warning("Could not serialize audit to JSON.")

        if previous_audit:
            st.divider()
            st.markdown("#### Changes since previous run")
            changes = _diff_dicts(audit, previous_audit)
            if changes.empty:
                st.success("No changes detected.")
            else:
                st.dataframe(changes, width="stretch", hide_index=True)
                st.download_button(
                    "⬇️ Download changes (CSV)",
                    data=changes.to_csv(index=False).encode("utf-8"),
                    file_name="audit_changes.csv",
                    mime="text/csv",
                    width="stretch",
                )

        st.divider()
        st.markdown("#### Audit Table")

        LABELS: Dict[str, str] = {
            "gross_bulk_m2": "Gross bulk (m²)",
            "sellable_area_m2": "Sellable area (m²)",
            "units_estimate": "Units (estimate)",
            "market_area_m2": "Market area (m²)",
            "affordable_area_m2": "Affordable area (m²)",
            "gdv": "GDV (total revenue)",
            "build_cost_psm_effective": "Build cost (effective) (R/m² bulk)",
            "cost_uplift_factor": "Cost uplift factor",
            "affordable_cost_multiplier": "Affordable cost multiplier",
            "build_cost": "Build cost (total)",
            "contingency": "Contingency",
            "escalation": "Escalation",
            "professional_fees": "Professional fees",
            "rates_taxes": "Rates & taxes",
            "base_costs": "Base costs (pre mkt/fin/profit)",
            "marketing": "Marketing",
            "finance": "Finance (proxy)",
            "finance_marketing": "Legacy finance + marketing",
            "total_costs_ex_profit": "Total costs (ex profit)",
            "profit_basis": "Profit basis",
            "profit_base": "Profit base",
            "profit": "Profit",
            "residual_land_value": "Residual land value (pre acquisition)",
            "acquisition_costs": "Acquisition costs",
        }

        factor_keys = {"cost_uplift_factor", "affordable_cost_multiplier"}
        area_keys = {"gross_bulk_m2", "sellable_area_m2", "market_area_m2", "affordable_area_m2"}
        rands_per_sqm_keys = {"build_cost_psm_effective"}
        unit_keys = {"units_estimate"}

        def label_for(k: str) -> str:
            return LABELS.get(k, k.replace("_", " ").strip().title())

        def fmt_int(n: float) -> str:
            return f"{int(round(n, 0)):,}".replace(",", " ")

        def fmt_float(n: float, dp: int = 2) -> str:
            return f"{n:,.{dp}f}".replace(",", " ")

        def fmt_factor(x: float) -> str:
            return f"{x:.2f}×"

        def fmt_pct(p: float) -> str:
            return f"{p * 100:.1f}%"

        def fmt_value(k: str, v: Any) -> str:
            if isinstance(v, str):
                return v
            if isinstance(v, bool):
                return "Yes" if v else "No"

            try:
                n = float(v)
            except Exception:
                return str(v)

            if k in money_key_set:
                return _money(n)
            if k in rands_per_sqm_keys:
                return f"{_money(n)}/m²"
            if k in area_keys:
                return f"{fmt_int(n)} m²"
            if k in unit_keys:
                if abs(n - round(n)) < 1e-9:
                    return fmt_int(n)
                return fmt_float(n, 1)
            if k in factor_keys:
                return fmt_factor(n)
            if k.endswith("_rate") or "rate" in k:
                return fmt_pct(n)
            if abs(n) >= 1000:
                return fmt_float(n, 0)
            return fmt_float(n, 2)

        SECTIONS: Dict[str, list[str]] = {
            "Areas": ["gross_bulk_m2", "sellable_area_m2", "units_estimate", "market_area_m2", "affordable_area_m2"],
            "Revenue": ["gdv"],
            "Costs": [
                "build_cost_psm_effective", "cost_uplift_factor", "affordable_cost_multiplier",
                "build_cost", "contingency", "escalation", "professional_fees", "rates_taxes",
                "base_costs", "marketing", "finance", "finance_marketing", "total_costs_ex_profit",
            ],
            "Profit": ["profit_basis", "profit_base", "profit"],
            "Land": ["residual_land_value", "acquisition_costs"],
        }

        rows: list[tuple[str, str]] = []
        used_keys = set()

        def add_separator(section_title: str) -> None:
            rows.append((f"— {section_title} —", ""))

        for section, keys in SECTIONS.items():
            add_separator(section)
            for k in keys:
                if k in audit:
                    rows.append((label_for(k), fmt_value(k, audit[k])))
                    used_keys.add(k)

        leftovers = [k for k in audit.keys() if k not in used_keys]
        if leftovers:
            add_separator("Other")
            for k in leftovers:
                rows.append((label_for(k), fmt_value(k, audit[k])))

        df = pd.DataFrame(rows, columns=["Item", "Value"])
        df["Item"] = df["Item"].astype(str)
        df["Value"] = df["Value"].astype(str)

        st.dataframe(df, width="stretch", hide_index=True)

        if show_raw_json:
            with st.expander("🔍 Raw audit JSON", expanded=False):
                st.json(audit)


# =========================================================
# MINIMAL ENGINE (so app runs end-to-end)
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

    build_cost = gross_bulk_m2 * build_cost_psm
    contingency = build_cost * contingency_rate
    escalation = build_cost * escalation_rate
    professional_fees = build_cost * prof_fees_rate
    rates_taxes = gdv * rates_taxes_rate

    base_costs = build_cost + contingency + escalation + professional_fees + rates_taxes
    marketing = gdv * marketing_rate
    finance = (base_costs + marketing) * finance_rate

    total_costs_ex_profit = base_costs + marketing + finance

    profit_base = gdv
    profit = profit_base * profit_rate

    residual_land_value = gdv - total_costs_ex_profit - profit

    return {
        "gross_bulk_m2": gross_bulk_m2,
        "sellable_area_m2": sellable_area_m2,
        "market_area_m2": market_area_m2,
        "affordable_area_m2": affordable_area_m2,
        "gdv": gdv,
        "build_cost_psm_effective": build_cost_psm,
        "build_cost": build_cost,
        "contingency": contingency,
        "escalation": escalation,
        "professional_fees": professional_fees,
        "rates_taxes": rates_taxes,
        "base_costs": base_costs,
        "marketing": marketing,
        "finance": finance,
        "total_costs_ex_profit": total_costs_ex_profit,
        "profit_basis": "GDV",
        "profit_base": profit_base,
        "profit": profit,
        "residual_land_value": residual_land_value,
        "inputs": {
            "plot_size_m2": plot_size_m2,
            "floor_factor": floor_factor,
            "efficiency": efficiency,
            "exit_price_psm": exit_price_psm,
            "build_cost_psm": build_cost_psm,
            "include_affordable": include_affordable,
            "affordable_share": affordable_share,
            "affordable_price_psm": affordable_price_psm,
        },
    }


# =========================================================
# STREAMLIT UI
# =========================================================

st.set_page_config(page_title="IH RLV Calculator", layout="wide")
st.title("IH Residual Land Value Calculator")
st.caption("Developer-style audit trail + waterfall (GDV → Costs → Profit → RLV).")

with st.sidebar:
    st.header("Inputs")

    plot_size_m2 = st.number_input("Plot size (m²)", min_value=50.0, value=1000.0, step=10.0)
    floor_factor = st.number_input("Floor factor (FAR)", min_value=0.1, value=1.2, step=0.05)
    efficiency = st.slider("Efficiency (sellable / bulk)", 0.50, 0.95, 0.85, 0.01)

    exit_price_psm = st.number_input("Exit price (R/m² sellable)", min_value=1000.0, value=42000.0, step=500.0)
    build_cost_psm = st.number_input("Build cost (R/m² bulk)", min_value=1000.0, value=18000.0, step=500.0)

    st.subheader("Inclusionary Housing")
    include_affordable = st.checkbox("Include affordable component", value=True)
    affordable_share = st.slider("Affordable share of sellable area", 0.00, 0.40, 0.10, 0.01)
    affordable_price_psm = st.number_input("Affordable price (R/m²)", min_value=1000.0, value=12000.0, step=500.0)

    st.subheader("Rates")
    contingency_rate = st.slider("Contingency % of build", 0.00, 0.20, 0.05, 0.005)
    escalation_rate = st.slider("Escalation % of build", 0.00, 0.20, 0.04, 0.005)
    prof_fees_rate = st.slider("Professional fees % of build", 0.00, 0.25, 0.10, 0.005)
    rates_taxes_rate = st.slider("Rates & taxes % of GDV", 0.00, 0.10, 0.02, 0.0025)
    marketing_rate = st.slider("Marketing % of GDV", 0.00, 0.10, 0.03, 0.0025)
    finance_rate = st.slider("Finance % (proxy) on costs", 0.00, 0.30, 0.10, 0.005)
    profit_rate = st.slider("Profit % of GDV", 0.00, 0.40, 0.20, 0.01)

audit = compute_feasibility(
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
)

c1, c2, c3, c4 = st.columns(4)
c1.metric("GDV", _money(_num(audit, "gdv")))
c2.metric("Total costs (ex profit)", _money(_num(audit, "total_costs_ex_profit")))
c3.metric("Profit", _money(_num(audit, "profit")))
c4.metric("Residual Land Value", _money(_num(audit, "residual_land_value")))

st.divider()

if "prev_audit" not in st.session_state:
    st.session_state.prev_audit = None

render_audit_waterfall(audit, expanded=True)
render_audit(audit, previous_audit=st.session_state.prev_audit, expanded=False)

st.session_state.prev_audit = audit
