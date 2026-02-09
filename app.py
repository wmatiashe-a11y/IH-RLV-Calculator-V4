def render_audit(audit: Dict[str, Any]) -> None:
    st.markdown("### Audit Trail")

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
        "gdv",
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
        "profit_base",
        "profit",
        "residual_land_value",
        "acquisition_costs",
    }

    factor_keys = {"cost_uplift_factor", "affordable_cost_multiplier"}
    area_keys = {"gross_bulk_m2", "sellable_area_m2", "market_area_m2", "affordable_area_m2"}
    rands_per_sqm_keys = {"build_cost_psm_effective"}
    unit_keys = {"units_estimate"}

    def label_for(k: str) -> str:
        if k in LABELS:
            return LABELS[k]
        return k.replace("_", " ").strip().title()

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

        # Heuristic: any *_rate style keys (if ever passed through audit later)
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
        "Revenue": [
            "gdv",
        ],
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
        "Profit": [
            "profit_basis",
            "profit_base",
            "profit",
        ],
        "Land": [
            "residual_land_value",
            "acquisition_costs",
        ],
    }

    # Build rows with separators (still Arrow-safe because strings)
    rows: list[tuple[str, str]] = []
    used_keys = set()

    def add_separator(title: str) -> None:
        rows.append((f"— {title} —", ""))

    for section, keys in SECTIONS.items():
        add_separator(section)
        for k in keys:
            if k in audit:
                rows.append((label_for(k), fmt_value(k, audit[k])))
                used_keys.add(k)

    # Any remaining keys fall into "Other"
    leftovers = [k for k in audit.keys() if k not in used_keys]
    if leftovers:
        add_separator("Other")
        for k in leftovers:
            rows.append((label_for(k), fmt_value(k, audit[k])))

    import pandas as pd

    df = pd.DataFrame(rows, columns=["Item", "Value"])

    # ✅ Arrow-safe: force string dtype
    df["Item"] = df["Item"].astype(str)
    df["Value"] = df["Value"].astype(str)

    st.dataframe(df, width="stretch", hide_index=True)

