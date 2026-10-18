"""SRE triage dashboard (Streamlit)."""

from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # allow `streamlit run` without installing

from sre_triage.ui import pages  # noqa: E402

st.set_page_config(page_title="SRE Incident Triage", page_icon=":material/emergency:", layout="wide")

pages.PAGES.update(
    incidents=st.Page(pages.incidents, title="Incidents", icon=":material/emergency:", url_path="incidents", default=True),
    case=st.Page(pages.case_page, title="Case", icon=":material/manage_search:", url_path="case"),
    memory=st.Page(pages.memory_page, title="Incident memory", icon=":material/history_edu:", url_path="memory"),
    evaluation=st.Page(pages.evaluation_page, title="Evaluation", icon=":material/analytics:", url_path="evaluation"),
    faults=st.Page(pages.fault_lab, title="Fault lab (operator)", icon=":material/science:", url_path="faults"),
)
nav = st.navigation(list(pages.PAGES.values()))
nav.run()
