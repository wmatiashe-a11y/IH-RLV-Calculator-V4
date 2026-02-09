from __future__ import annotations

from typing import Dict, Any, Optional
import json
import pandas as pd
import streamlit as st

import plotly.graph_objects as go


def _num(audit: Dict[str, Any], key: str, default: float = 0.0) -> float:
    """Safely fetch a numeric value from audit."""
    v = audit.get(key, default)
    try:
        return float(v)
    except Exception:
        return default


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

    with st.expander(title, expanded=expanded):
        if show_key_drivers:
            computed_rlv = gdv - sum(v for _, v in cost_items) - abs(profit)
            rlv_to_show = rlv if abs(rlv) > 1e-9 else computed_rlv

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

        computed_rlv = gdv - sum(v for _, v in cost_items) - abs(profit)
        rlv_to_show = rlv if abs(rlv) > 1e-9 else computed_rlv

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
        st.plotly_chart(fig, use_container_width=True)

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
            st.plotly_chart(fig2, use_container_width=True)


def _money(x: float) -> str:
    """Format money as South African Rand with spaced thousands (e.g. R1 234 567)."""
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
        prev = f_old.get(k, None)
        cur = f_new.get(k, None)
        if prev != cur:
            rows.append({"Field": k, "Previous": prev, "Current": cur})
    return pd.DataFrame(rows)


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
                use_container_width=True,
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
                st.dataframe(changes, use_container_width=True, hide_index=True)

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

        st.dataframe(df, use_container_width=True, hide_index=True)

        if show_raw_json:
            with st.expander("🔍 Raw audit JSON", expanded=False):
                st.json(audit)

    # Core money values (expected from your audit)
    gdv = _num(audit, "gdv", 0.0)
    build_cost = _num(audit, "build_cost", 0.0)
    contingency = _num(audit, "contingency", 0.0)
    escalation = _num(audit, "escalation", 0.0)
    professional_fees = _num(audit, "professional_fees", 0.0)
    rates_taxes = _num(audit, "rates_taxes", 0.0)
    marketing = _num(audit, "marketing", 0.0)
    finance = _num(audit, "finance", 0.0)

    # Some apps use either "finance_marketing" legacy combined value
    finance_marketing = _num(audit, "finance_marketing", 0.0)

    # Profit / RLV
    profit = _num(audit, "profit", 0.0)
    rlv = _num(audit, "residual_land_value", 0.0)
    acquisition = _num(audit, "acquisition_costs", 0.0)

    # Prefer detailed finance + marketing if present; otherwise use combined
    use_combined = (marketing == 0.0 and finance == 0.0 and finance_marketing != 0.0)

    # Build a cost stack (positive values in audit → negative steps in waterfall)
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
        cost_items.extend([
            ("Marketing", marketing),
            ("Finance", finance),
        ])

    # Only keep non-zero items (keeps chart tidy)
    cost_items = [(n, v) for n, v in cost_items if abs(v) > 1e-9]

    with st.expander(title, expanded=expanded):
        # --- Key drivers row (optional)
        if show_key_drivers:
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("GDV", _money(gdv))
            c2.metric("Total costs (ex profit)", _money(_num(audit, "total_costs_ex_profit", sum(v for _, v in cost_items))))
            c3.metric("Profit", _money(profit))
            c4.metric("Residual land value", _money(rlv))

            # Optional “bulk/area” drivers if present
            gb = audit.get("gross_bulk_m2")
            sa = audit.get("sellable_area_m2")
            if gb is not None or sa is not None:
                st.caption(
                    " • ".join(
                        [f"Gross bulk: {gb:.0f} m²" if isinstance(gb, (int, float)) else f"Gross bulk: {gb}" if gb is not None else "",
                         f"Sellable area: {sa:.0f} m²" if isinstance(sa, (int, float)) else f"Sellable area: {sa}" if sa is not None else ""]
                    ).strip(" •")
                )

        # --- Waterfall steps
        labels = ["GDV"]
        measures = ["absolute"]
        values = [gdv]

        # Costs as negative deltas
        for name, val in cost_items:
            labels.append(name)
            measures.append("relative")
            values.append(-abs(val))

        # Profit as negative delta (only if present)
        if abs(profit) > 1e-9:
            labels.append("Profit")
            measures.append("relative")
            values.append(-abs(profit))

        # Show RLV as a total (preferred, uses your computed rlv if present)
        # If rlv isn't present, compute it: GDV - sum(costs) - profit
        computed_rlv = gdv - sum(v for _, v in cost_items) - abs(profit)
        rlv_to_show = rlv if abs(rlv) > 1e-9 else computed_rlv

        labels.append("Residual land value")
        measures.append("total")
        values.append(rlv_to_show)

        fig = go.Figure(
            go.Waterfall(
                name="Feasibility",
                orientation="v",
                measure=measures,
                x=labels,
                y=values,
                connector={"line": {"width": 1}},
            )
        )

        fig.update_layout(
            margin=dict(l=10, r=10, t=30, b=10),
            height=420,
            title=dict(text="", x=0.0),
            yaxis_title="ZAR",
            showlegend=False,
        )

        st.plotly_chart(fig, use_container_width=True)

        # --- Optional: Acquisition bridge (post-RLV)
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
            st.plotly_chart(fig2, use_container_width=True)

        # --- Breakdown table
        st.markdown("#### Breakdown")
        breakdown_rows = [{"Item": "GDV", "Value": _money(gdv)}]
        for n, v in cost_items:
            breakdown_rows.append({"Item": n, "Value": f"-{_money(abs(v))}"})
        if abs(profit) > 1e-9:
            breakdown_rows.append({"Item": "Profit", "Value": f"-{_money(abs(profit))}"})
        breakdown_rows.append({"Item": "Residual land value", "Value": _money(rlv_to_show)})

        st.dataframe(pd.DataFrame(breakdown_rows), use_container_width=True, hide_index=True)

def _money(x: float) -> str:
    """Format money as South African Rand with spaced thousands (e.g. R1 234 567)."""
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
        prev = f_old.get(k, None)
        cur = f_new.get(k, None)
        if prev != cur:
            rows.append({"Field": k, "Previous": prev, "Current": cur})
    return pd.DataFrame(rows)


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

        # --- Download audit JSON ---
        try:
            audit_json = json.dumps(audit, indent=2, default=str).encode("utf-8")
            st.download_button(
                "⬇️ Download audit (JSON)",
                data=audit_json,
                file_name="audit_trail.json",
                mime="application/json",
                use_container_width=True,
            )
        except Exception:
            st.warning("Could not serialize audit to JSON (non-serializable values).")

        # --- Optional diff vs previous run ---
        if previous_audit:
            st.divider()
            st.markdown("#### Changes since previous run")
            changes = _diff_dicts(audit, previous_audit)
            if changes.empty:
                st.success("No changes detected.")
            else:
                st.dataframe(changes, use_container_width=True, hide_index=True)
                st.download_button(
                    "⬇️ Download changes (CSV)",
                    data=changes.to_csv(index=False).encode("utf-8"),
                    file_name="audit_changes.csv",
                    mime="text/csv",
                    use_container_width=True,
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

        st.dataframe(df, use_container_width=True, hide_index=True)

        if show_raw_json:
            with st.expander("🔍 Raw audit JSON", expanded=False):
                st.json(audit)
