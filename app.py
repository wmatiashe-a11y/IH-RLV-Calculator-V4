from typing import Dict, Any

from typing import Dict, Any, Optional
import json
import pandas as pd
import streamlit as st


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
    # Guard
    if not audit:
        st.info("No audit data available for this run.")
        return

    with st.expander(title, expanded=expanded):
        st.markdown("#### Summary")

        # --- Quick stats (counts only; safe even if some keys missing) ---
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Fields", len(audit))
        col2.metric("Money items", sum(1 for k in audit.keys() if k in {
            "gdv","build_cost","contingency","escalation","professional_fees","rates_taxes",
            "base_costs","marketing","finance","finance_marketing","total_costs_ex_profit",
            "profit_base","profit","residual_land_value","acquisition_costs"
        }))
        col3.metric("Areas / Units", sum(1 for k in audit.keys() if k in {
            "gross_bulk_m2","sellable_area_m2","market_area_m2","affordable_area_m2","units_estimate"
        }))
        col4.metric("Other", max(0, len(audit) - col2._value - col3._value) if hasattr(col2, "_value") else 0)

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

        # --- Human-friendly labels (extend anytime) ---
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

        # --- Formatting buckets ---
        money_keys = {
            "gdv", "build_cost", "contingency", "escalation", "professional_fees",
            "rates_taxes", "base_costs", "marketing", "finance", "finance_marketing",
            "total_costs_ex_profit", "profit_base", "profit", "residual_land_value",
            "acquisition_costs",
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

            # NOTE: _money must exist elsewhere in your app
            if k in money_keys:
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

        # --- Section definitions ---
        SECTIONS: Dict[str, list[str]] = {
            "Areas": [
                "gross_bulk_m2",
                "sellable_area_m2",
                "units_estimate",
                "market_area_m2",
                "affordable_area_m2",
            ],
            "Revenue": ["gdv"],
            "Costs": [
                "build_cost_psm_effective",
                "cost_uplift_factor",
                "affordable_cost_multiplier",
                "build_cost",
                "contingency",
                "escalation",
                "professional_fees",
                "rates_taxes",
                "base_costs",
                "marketing",
                "finance",
                "finance_marketing",
                "total_costs_ex_profit",
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

        # ✅ Correct Streamlit param:
        st.dataframe(df, use_container_width=True, hide_index=True)

        if show_raw_json:
            with st.expander("🔍 Raw audit JSON", expanded=False):
                st.json(audit)
