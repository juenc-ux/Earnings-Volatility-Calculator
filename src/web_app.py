import os
import pandas as pd
import streamlit as st

# Ensure matplotlib does not require Tk
os.environ.setdefault("EVC_USE_TK", "0")

from calculator import OptionsAnalyzer, SessionManager, show_interactive_chart  # type: ignore


@st.cache_resource
def get_analyzer() -> OptionsAnalyzer:
    session_manager = SessionManager()
    return OptionsAnalyzer(session_manager)


st.set_page_config(page_title="Earnings Volatility Calculator", layout="wide")
st.title("Earnings Volatility Calculator (Web)")

analyzer = get_analyzer()

tab1, tab2 = st.tabs(["Single Stock", "Earnings Scan"])

with tab1:
    ticker = st.text_input("Enter Stock Symbol", value="AAPL").strip().upper()
    colA, colB = st.columns([1, 1])
    with colA:
        analyze_clicked = st.button("Analyze", use_container_width=True)
    if analyze_clicked and ticker:
        with st.spinner(f"Analyzing {ticker}..."):
            result = analyzer.analyze_stock(ticker)
        if result is None:
            st.warning("No data or analysis failed.")
        else:
            st.success("Done.")
            st.json(result)

with tab2:
    date = st.date_input("Earnings Date")
    run_scan = st.button("Scan Earnings", use_container_width=True)
    if run_scan:
        with st.spinner("Scanning earnings..."):
            rows = analyzer.scan_earnings_stocks(pd.Timestamp(date).to_pydatetime())
        if not rows:
            st.info("No results.")
        else:
            df = pd.DataFrame([analyzer.build_row_values(r) for r in rows],
                              columns=analyzer.headings)
            st.dataframe(df, use_container_width=True)


