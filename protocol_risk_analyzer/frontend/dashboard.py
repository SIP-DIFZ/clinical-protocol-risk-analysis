from __future__ import annotations

import html
from pathlib import Path
from typing import Any

import altair as alt
import pandas as pd
import streamlit as st

from protocol_risk_analyzer.models import AnalysisResult
from protocol_risk_analyzer.reporting import build_risk_markdown_section
from protocol_risk_analyzer.ui_helpers import (
    _format_domain_label,
    _format_domain_tab_label,
    _mitigation_status_css_class,
    _mitigation_status_label,
    _normalize_mitigation_status_for_ui,
    _sanitize_accepted_risk_ids,
    _sanitize_accepted_risk_notes,
    _safe_key,
    _score_band,
)

DOMAIN_COLORS = {
    "health": "#D95F5F",
    "financial": "#4C78A8",
    "ethical": "#8E6BBE",
    "eligibility": "#2A9D8F",
    "intervention": "#F4A261",
}


def render_analysis_results(
    analysis_state: dict[str, Any],
    saved_root: Path,
) -> None:
    result = AnalysisResult.model_validate_json(analysis_state["json_text"])

    default_view = st.session_state.get("analysis_view")
    if default_view not in {"Dashboard", "Report", "Mitigate Protocol"}:
        default_view = "Dashboard"
        st.session_state["analysis_view"] = default_view
    selected_view = st.segmented_control(
        "View",
        options=["Dashboard", "Report", "Mitigate Protocol"],
        selection_mode="single",
        key="analysis_view",
        label_visibility="collapsed",
    )
    analysis_state["analysis_view"] = selected_view
    st.session_state["analysis_state"] = analysis_state

    if selected_view == "Dashboard":
        _render_risk_dashboard(result, analysis_state)
    elif selected_view == "Report":
        st.markdown(analysis_state["md_text"])
    else:
        from protocol_risk_analyzer.frontend.mitigate_protocol import render_editor_agent_builder

        render_editor_agent_builder(
            result,
            analysis_state,
            render_editor_job_monitor_fragment=_render_editor_job_monitor_fragment,
        )

    from protocol_risk_analyzer.frontend.assistant_popup import render_ai_assistant_popup

    render_ai_assistant_popup(
        result,
        analysis_state,
        saved_root,
        _sync_saved_analysis_snapshot,
    )


def _render_risk_dashboard(result: AnalysisResult, analysis_state: dict[str, Any]) -> None:
    st.subheader("Risk Summary Dashboard")

    df = _build_risk_dataframe(result)
    if df.empty:
        st.info("No risks were extracted for this protocol.")
        return
    valid_risk_ids = set(df["risk_id"].tolist())
    accepted_risk_ids = _sanitize_accepted_risk_ids(analysis_state.get("accepted_risk_ids", []), valid_risk_ids)
    if accepted_risk_ids != analysis_state.get("accepted_risk_ids", []):
        analysis_state["accepted_risk_ids"] = accepted_risk_ids
        st.session_state["analysis_state"] = analysis_state
    accepted_risk_notes = _sanitize_accepted_risk_notes(analysis_state.get("accepted_risk_notes", {}), valid_risk_ids)
    if accepted_risk_notes != analysis_state.get("accepted_risk_notes", {}):
        analysis_state["accepted_risk_notes"] = accepted_risk_notes
        st.session_state["analysis_state"] = analysis_state
    accepted_risk_id_set = set(accepted_risk_ids)
    df["is_accepted"] = df["risk_id"].apply(lambda risk_id: risk_id in accepted_risk_id_set)
    df["is_accepted_label"] = df["is_accepted"].map(lambda is_accepted: "Yes" if is_accepted else "No")

    summary_df = df[~df["is_accepted"]].copy()
    _render_portfolio_summary(summary_df)
    with st.expander("Severity Overview Chart", expanded=True):
        if summary_df.empty:
            st.info("All risks are currently accepted, so no active risks are included in summary metrics.")
        else:
            st.altair_chart(_build_severity_overview_chart(summary_df), use_container_width=True)

    _render_accepted_risks_overview(df)

    ordered_domains = ["health", "financial", "ethical", "eligibility", "intervention"]
    active_domains = [d for d in ordered_domains if not df[df["domain"] == d].empty]
    domain_tabs = st.tabs([_format_domain_tab_label(d) for d in active_domains])
    for tab, domain in zip(domain_tabs, active_domains):
        with tab:
            _render_domain_section(df, domain, analysis_state)


def _render_portfolio_summary(df: pd.DataFrame) -> None:
    total_risks = len(df)
    unmitigated_risk_count = int((df["mitigation_status"] == "unmitigated").sum())
    partially_mitigated_risk_count = int((df["mitigation_status"] == "partially_mitigated").sum())
    adequately_mitigated_risk_count = int((df["mitigation_status"] == "adequately_mitigated").sum())
    high_risk_count = int((df["score"] >= 12).sum())
    avg_score = float(df["score"].mean()) if not df.empty else 0.0

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Risks", total_risks)
    c2.markdown(
        (
            "<div class='summary-status-metric-card'>"
            "<div class='summary-status-metric-label'>Unmitigated / Partially / Adequately</div>"
            f"<div class='summary-status-metric-value'>{unmitigated_risk_count} / "
            f"{partially_mitigated_risk_count} / {adequately_mitigated_risk_count}</div>"
            "</div>"
        ),
        unsafe_allow_html=True,
    )
    c3.metric("Average Score", f"{avg_score:.2f}")
    c4.metric("High Risks", high_risk_count)
    key_pair_col, _ = st.columns([2.9, 5.1], gap="small")
    with key_pair_col:
        key_left_col, key_right_col = st.columns([1, 1], gap="small")
        with key_left_col:
            with st.popover("Score Scale (0-25)", use_container_width=True):
                _render_score_scale_key()
        with key_right_col:
            with st.popover("Mitigation Status Key", use_container_width=True):
                _render_mitigation_status_key()


def _render_accepted_risks_overview(df: pd.DataFrame) -> None:
    accepted_df = df[df["is_accepted"]]
    with st.expander(f"Accepted Risks ({len(accepted_df)})", expanded=False):
        if accepted_df.empty:
            st.caption("No risks are currently accepted.")
            return
        overview = accepted_df[["title", "domain_label", "score", "mitigation_status_label"]].rename(
            columns={
                "title": "Risk",
                "domain_label": "Domain",
                "score": "Score",
                "mitigation_status_label": "Status",
            }
        )
        overview["Score"] = overview["Score"].map(lambda value: f"{float(value):.2f}")
        st.dataframe(overview, use_container_width=True, hide_index=True)
        st.caption("Use the Accepted column in the domain risk tables to accept or unaccept risks.")


def _render_domain_section(df: pd.DataFrame, domain: str, analysis_state: dict[str, Any]) -> None:
    domain_df = _filter_risks(df, selected_domain=domain, min_score=0.0, query="")
    domain_label = _format_domain_label(domain)

    if domain_df.empty:
        st.info(f"No {domain_label} risks were extracted.")
        return

    metrics_df = domain_df[~domain_df["is_accepted"]]
    has_metrics = not metrics_df.empty
    top_row = metrics_df.iloc[0] if has_metrics else domain_df.iloc[0]
    c1, c2, c3 = st.columns(3)
    c1.metric(f"{domain_label} Risks", len(metrics_df))
    c2.metric(f"Average {domain_label} Score", f"{metrics_df['score'].mean():.2f}" if has_metrics else "0.00")
    c3.metric(f"Highest {domain_label} Risk", f"{top_row['score']:.2f}" if has_metrics else "N/A")

    st.markdown(f"### {domain_label} Risk List")
    risk_status_filter = st.radio(
        f"{domain_label} risk status",
        options=["All", "Unmitigated", "Partially Mitigated", "Adequately Mitigated"],
        index=0,
        horizontal=True,
        key=f"{domain}_risk_status_filter",
        label_visibility="collapsed",
    )
    group_by_category = st.toggle(
        f"Group {domain_label} risks by category",
        value=False,
        key=f"{domain}_group_by_category",
    )
    show_accepted_only = st.toggle(
        f"Show accepted {domain_label} risks",
        value=False,
        key=f"{domain}_show_accepted_only",
    )
    filtered = domain_df[domain_df["is_accepted"]] if show_accepted_only else domain_df[~domain_df["is_accepted"]]
    status_value_map = {
        "Unmitigated": "unmitigated",
        "Partially Mitigated": "partially_mitigated",
        "Adequately Mitigated": "adequately_mitigated",
    }
    if risk_status_filter != "All":
        filtered = filtered[filtered["mitigation_status"] == status_value_map[risk_status_filter]]

    if filtered.empty:
        st.info(f"No {domain_label} risks match the selected status.")
        return

    valid_risk_ids = set(df["risk_id"].tolist())
    accepted_risk_ids = _sanitize_accepted_risk_ids(
        analysis_state.get("accepted_risk_ids", []),
        valid_risk_ids,
    )
    selected_risk_id = st.session_state.get(f"{domain}_selected_risk_id")
    if group_by_category:
        selected_from_table = _render_grouped_risk_tables(
            filtered=filtered,
            domain=domain,
            selected_risk_state_key=f"{domain}_selected_risk_id",
            valid_risk_ids=valid_risk_ids,
            selected_risk_id=selected_risk_id,
        )
    else:
        selected_from_table = _render_editable_risk_table(
            filtered=filtered,
            table_key=f"{domain}_risk_table",
            selected_risk_state_key=f"{domain}_selected_risk_id",
            valid_risk_ids=valid_risk_ids,
            selected_risk_id=selected_risk_id,
        )
    if selected_from_table:
        st.session_state[f"{domain}_selected_risk_id"] = selected_from_table
    selected_row = _resolve_selected_row_from_state(filtered, domain)
    st.divider()
    st.markdown(f"### {domain_label} Risk Details")
    selected_risk_id = str(selected_row.get("risk_id", "")).strip()
    accepted_risk_id_set = set(accepted_risk_ids)
    is_accepted = selected_risk_id in accepted_risk_id_set
    accepted_risk_notes = _sanitize_accepted_risk_notes(analysis_state.get("accepted_risk_notes", {}), valid_risk_ids)
    if accepted_risk_notes != analysis_state.get("accepted_risk_notes", {}):
        analysis_state["accepted_risk_notes"] = accepted_risk_notes
        st.session_state["analysis_state"] = analysis_state
    action_label = "Unaccept Risk" if is_accepted else "Accept Risk"
    guidance_text = (
        "This risk is accepted and excluded from Editor Agent patch generation."
        if is_accepted
        else "Accepting a risk means you are accepting its influence on your protocol. It will be removed from your summary statistics."
    )
    accept_pending_key = f"{domain}_accept_pending_risk_id"
    pending_risk_id = str(st.session_state.get(accept_pending_key, "")).strip()
    is_accept_pending = (not is_accepted) and pending_risk_id == selected_risk_id
    risk_note_key = f"{domain}_accepted_risk_note_{_safe_key(selected_risk_id)}"
    existing_note = str(accepted_risk_notes.get(selected_risk_id, "")).strip()
    if risk_note_key not in st.session_state:
        st.session_state[risk_note_key] = existing_note
    action_col, guidance_col = st.columns([0.65, 4.35], gap="small", vertical_alignment="center")
    with action_col:
        if is_accepted:
            if st.button(action_label, key=f"{domain}_accept_toggle_{_safe_key(selected_risk_id)}", use_container_width=True):
                current_accepted = _sanitize_accepted_risk_ids(
                    analysis_state.get("accepted_risk_ids", []),
                    valid_risk_ids,
                )
                current_notes = _sanitize_accepted_risk_notes(analysis_state.get("accepted_risk_notes", {}), valid_risk_ids)
                updated_accepted = [risk_id for risk_id in current_accepted if risk_id != selected_risk_id]
                current_notes.pop(selected_risk_id, None)
                analysis_state["accepted_risk_ids"] = _sanitize_accepted_risk_ids(updated_accepted, valid_risk_ids)
                analysis_state["accepted_risk_notes"] = _sanitize_accepted_risk_notes(current_notes, valid_risk_ids)
                st.session_state[accept_pending_key] = ""
                st.session_state["analysis_state"] = analysis_state
                st.rerun()
        elif not is_accept_pending:
            if st.button(action_label, key=f"{domain}_accept_toggle_{_safe_key(selected_risk_id)}", use_container_width=True):
                st.session_state[accept_pending_key] = selected_risk_id
                st.rerun()
    with guidance_col:
        if not is_accept_pending:
            st.caption(guidance_text)
        if is_accepted and existing_note:
            st.caption(f"Reason: {existing_note}")
    if is_accept_pending:
        st.caption(guidance_text)
        st.text_area(
            "Acceptance comment",
            key=risk_note_key,
            placeholder="Add a short reason for accepting this risk.",
            height=80,
        )
        confirm_col, cancel_col = st.columns(2, gap="small")
        with confirm_col:
            if st.button("Confirm Accept", key=f"{domain}_accept_confirm_{_safe_key(selected_risk_id)}", use_container_width=True):
                current_accepted = _sanitize_accepted_risk_ids(
                    analysis_state.get("accepted_risk_ids", []),
                    valid_risk_ids,
                )
                current_notes = _sanitize_accepted_risk_notes(analysis_state.get("accepted_risk_notes", {}), valid_risk_ids)
                acceptance_note = str(st.session_state.get(risk_note_key, "")).strip()
                if not acceptance_note:
                    st.warning("Please add an acceptance comment before accepting this risk.")
                    return
                updated_accepted = current_accepted + [selected_risk_id]
                current_notes[selected_risk_id] = acceptance_note
                analysis_state["accepted_risk_ids"] = _sanitize_accepted_risk_ids(updated_accepted, valid_risk_ids)
                analysis_state["accepted_risk_notes"] = _sanitize_accepted_risk_notes(current_notes, valid_risk_ids)
                st.session_state[accept_pending_key] = ""
                st.session_state["analysis_state"] = analysis_state
                st.rerun()
        with cancel_col:
            if st.button("Cancel", key=f"{domain}_accept_cancel_{_safe_key(selected_risk_id)}", use_container_width=True):
                st.session_state[accept_pending_key] = ""
                st.rerun()

    status_label = str(selected_row.get("mitigation_status_label", "Unmitigated")).strip() or "Unmitigated"
    status_value = str(selected_row.get("mitigation_status", "unmitigated")).strip() or "unmitigated"
    status_class = _mitigation_status_css_class(status_value)
    status_reason = " ".join(str(selected_row.get("mitigation_status_reason", "")).strip().split())
    st.markdown(
        (
            f"<div class='status-reason-chip {status_class}'>"
            f"<span class='status-dot-inline {status_class}' aria-hidden='true'></span>"
            f"{html.escape(status_label)}: {html.escape(status_reason)}"
            "</div>"
        ),
        unsafe_allow_html=True,
    )
    with st.container(border=True):
        _render_risk_detail(selected_row)


def _render_risk_detail(selected_row: pd.Series) -> None:
    st.markdown(selected_row["report_markdown"])


def _build_risk_dataframe(result: AnalysisResult) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for idx, risk in enumerate(result.risks, start=1):
        mitigation_status = _normalize_mitigation_status_for_ui(risk.mitigation_status)
        rows.append(
            {
                "risk_id": f"{risk.domain.value}:{idx}:{risk.title}",
                "title": risk.title,
                "domain": risk.domain.value,
                "domain_label": _format_domain_label(risk.domain.value),
                "category": risk.category,
                "mitigation_status": mitigation_status,
                "mitigation_status_label": _mitigation_status_label(mitigation_status),
                "mitigation_status_reason": " ".join(str(risk.mitigation_status_reason or "").strip().split()),
                "likelihood": risk.likelihood,
                "likelihood_label": f"{risk.likelihood}/5",
                "impact": risk.impact,
                "impact_label": f"{risk.impact}/5",
                "confidence": risk.confidence,
                "confidence_label": f"{risk.confidence:.2f}",
                "score": risk.risk_score,
                "severity": _score_band(risk.risk_score),
                "rationale": risk.rationale,
                "factor_triggers": risk.factor_triggers,
                "factor_count": len(risk.factor_triggers),
                "mitigations": risk.mitigations,
                "evidence": [ev.model_dump() for ev in risk.evidence],
                "report_markdown": build_risk_markdown_section(risk),
                "is_accepted": False,
                "is_accepted_label": "No",
            }
        )
    return pd.DataFrame(rows).sort_values(["score", "title"], ascending=[False, True])


def _build_risk_table_editor_df(filtered: pd.DataFrame, selected_risk_id: str | None) -> pd.DataFrame:
    del selected_risk_id
    display_df = filtered[
        [
            "title",
            "category",
            "mitigation_status_label",
            "score",
            "severity",
            "likelihood_label",
            "impact_label",
            "confidence_label",
        ]
    ].rename(
        columns={
            "title": "Risk",
            "category": "Category",
            "mitigation_status_label": "Status",
            "score": "Score",
            "severity": "Severity",
            "likelihood_label": "Likelihood",
            "impact_label": "Impact",
            "confidence_label": "Confidence",
        }
    )
    return display_df.reset_index(drop=True)


def _render_editable_risk_table(
    filtered: pd.DataFrame,
    table_key: str,
    selected_risk_state_key: str,
    valid_risk_ids: set[str],
    selected_risk_id: str | None,
) -> str | None:
    editor_df = _build_risk_table_editor_df(filtered, selected_risk_id)
    displayed_risk_ids = [str(risk_id).strip() for risk_id in filtered["risk_id"].reset_index(drop=True).tolist()]
    if not displayed_risk_ids:
        return None

    current_selected = str(selected_risk_id or "").strip()
    if current_selected not in valid_risk_ids:
        current_selected = displayed_risk_ids[0]

    table_state = st.dataframe(
        editor_df,
        use_container_width=True,
        hide_index=True,
        key=table_key,
        column_config={
            "Score": st.column_config.NumberColumn("Score", format="%.2f"),
        },
        on_select="rerun",
        selection_mode="single-row",
    )

    selected_from_table: str | None = current_selected
    selected_rows = table_state.selection.rows if table_state is not None else []
    if selected_rows:
        selected_index = int(selected_rows[0])
        if 0 <= selected_index < len(displayed_risk_ids):
            selected_from_table = displayed_risk_ids[selected_index]

    if selected_from_table and selected_from_table != current_selected:
        st.session_state[selected_risk_state_key] = selected_from_table
    return selected_from_table


def _filter_risks(
    df: pd.DataFrame,
    selected_domain: str,
    min_score: float,
    query: str,
) -> pd.DataFrame:
    filtered = df[df["score"] >= min_score]

    if selected_domain != "all":
        filtered = filtered[filtered["domain"] == selected_domain]

    q = query.strip().lower()
    if q:
        mask = (
            filtered["title"].str.lower().str.contains(q)
            | filtered["category"].str.lower().str.contains(q)
            | filtered["factor_triggers"].apply(lambda factors: q in " ".join(factors).lower())
        )
        filtered = filtered[mask]

    return filtered.sort_values(["score", "title"], ascending=[False, True])


def _build_severity_overview_chart(df: pd.DataFrame) -> alt.Chart:
    severity_order = ["Low", "Moderate", "High", "Very High"]
    domain_order = ["Health", "Financial", "Ethical", "Eligibility", "Intervention"]
    summary_df = (
        df.groupby(["severity", "domain_label"], as_index=False)
        .agg(risk_count=("risk_id", "count"), average_score=("score", "mean"))
    )
    active_domain_labels = summary_df["domain_label"].unique().tolist()
    domain_labels = [label for label in domain_order if label in active_domain_labels]
    summary_df["severity"] = pd.Categorical(summary_df["severity"], categories=severity_order, ordered=True)
    summary_df["domain_label"] = pd.Categorical(
        summary_df["domain_label"], categories=domain_labels, ordered=True
    )
    summary_df = summary_df.sort_values(["severity", "domain_label"])
    color_range = [DOMAIN_COLORS.get(label.lower(), "#6B7280") for label in domain_labels]
    return (
        alt.Chart(summary_df)
        .mark_bar(cornerRadiusTopLeft=6, cornerRadiusTopRight=6)
        .encode(
            x=alt.X("severity:N", sort=severity_order, title="Severity Band"),
            y=alt.Y("risk_count:Q", title="Number of Risks"),
            color=alt.Color(
                "domain_label:N",
                scale=alt.Scale(domain=domain_labels, range=color_range),
                legend=alt.Legend(title="Risk Type"),
            ),
            xOffset=alt.XOffset("domain_label:N", sort=domain_labels),
            tooltip=[
                alt.Tooltip("domain_label:N", title="Risk Type"),
                alt.Tooltip("severity:N", title="Severity"),
                alt.Tooltip("risk_count:Q", title="Risk Count"),
                alt.Tooltip("average_score:Q", title="Average Score", format=".2f"),
            ],
        )
        .properties(height=320)
    )


def _resolve_selected_row_from_state(filtered: pd.DataFrame, domain: str) -> pd.Series:
    selected_risk_id = st.session_state.get(f"{domain}_selected_risk_id")
    if selected_risk_id:
        matched = filtered[filtered["risk_id"] == selected_risk_id]
        if not matched.empty:
            return matched.iloc[0]
    default_row = filtered.iloc[0]
    st.session_state[f"{domain}_selected_risk_id"] = default_row["risk_id"]
    return default_row


def _render_grouped_risk_tables(
    filtered: pd.DataFrame,
    domain: str,
    selected_risk_state_key: str,
    valid_risk_ids: set[str],
    selected_risk_id: str | None,
) -> str | None:
    selected_from_table: str | None = None
    for category, category_group in filtered.groupby("category", sort=True):
        with st.expander(f"{_format_domain_label(category)} ({len(category_group)})", expanded=False):
            selected_candidate = _render_editable_risk_table(
                filtered=category_group,
                table_key=f"{domain}_grouped_{_safe_key(category)}",
                selected_risk_state_key=selected_risk_state_key,
                valid_risk_ids=valid_risk_ids,
                selected_risk_id=selected_risk_id,
            )
            if selected_candidate:
                selected_from_table = selected_candidate
    return selected_from_table


def _resolve_selected_risk_id_from_checkboxes(
    displayed_risk_ids: list[str],
    selected_flags: list[object],
    current_selected_risk_id: str | None,
) -> str | None:
    if not displayed_risk_ids:
        return None
    current_id = str(current_selected_risk_id or "").strip()
    if current_id not in displayed_risk_ids:
        current_id = displayed_risk_ids[0]
    previous_flags = [risk_id == current_id for risk_id in displayed_risk_ids]
    normalized_flags = [
        bool(selected_flags[idx]) if idx < len(selected_flags) else False
        for idx in range(len(displayed_risk_ids))
    ]
    changed_indices = [
        idx
        for idx in range(len(displayed_risk_ids))
        if normalized_flags[idx] != previous_flags[idx]
    ]
    if not changed_indices:
        return current_id
    changed_to_true_indices = [
        idx
        for idx in changed_indices
        if normalized_flags[idx]
    ]
    if changed_to_true_indices:
        return displayed_risk_ids[changed_to_true_indices[-1]]
    return current_id


def _render_editor_job_monitor_fragment() -> None:
    from protocol_risk_analyzer.frontend.mitigate_protocol import _render_editor_job_monitor_fragment as _render

    _render()


def _sync_saved_analysis_snapshot(analysis_state: dict[str, Any], saved_root: Path) -> None:
    from protocol_risk_analyzer.saved_analysis_store import _sync_saved_analysis_snapshot as _sync

    _sync(analysis_state, saved_root)


def _render_mitigation_status_key() -> None:
    st.markdown(
        """
        <div class="mitigation-key">
            <div class="mitigation-key-item">
                <span class="mitigation-key-dot mitigation-key-dot-unmitigated" aria-hidden="true"></span>
                <div><strong>Unmitigated</strong> — no adequate controls are identified for the specific risk.</div>
            </div>
            <div class="mitigation-key-item">
                <span class="mitigation-key-dot mitigation-key-dot-partial" aria-hidden="true"></span>
                <div><strong>Partially Mitigated</strong> — existing controls help, but additional actions are needed to achieve an acceptable level of control.</div>
            </div>
            <div class="mitigation-key-item">
                <span class="mitigation-key-dot mitigation-key-dot-adequate" aria-hidden="true"></span>
                <div><strong>Adequately Mitigated</strong> — existing controls appear sufficient; listed items are optional enhancements, not required for acceptable management.</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _render_score_scale_key() -> None:
    st.markdown(
        """
        <div class="score-key">
            <div class="score-key-item"><span class="score-key-dot score-low"></span><strong>Low (0.00-7.99):</strong>&nbsp;Limited residual concern; usually monitor and maintain controls.</div>
            <div class="score-key-item"><span class="score-key-dot score-moderate"></span><strong>Moderate (8.00-11.99):</strong>&nbsp;Meaningful concern; strengthen controls and track implementation.</div>
            <div class="score-key-item"><span class="score-key-dot score-high"></span><strong>High (12.00-14.99):</strong>&nbsp;Substantial concern; prioritize mitigation planning and tighter oversight.</div>
            <div class="score-key-item"><span class="score-key-dot score-very-high"></span><strong>Very High (15.00-25.00):</strong>&nbsp;Critical residual concern; urgent control actions and escalation expected.</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


__all__ = ["render_analysis_results"]
