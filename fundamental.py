import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

st.set_page_config(page_title="Mining Value + RSI Screener", layout="wide", page_icon="gold_bar")
st.title("Gold & Silver Miners – Value + RSI + Debt Safety")
st.markdown("**Pure trailing metrics • RSI • Debt-aware Fundamental Score**")

# -------------------------------
# STOCK SELECTION – NOW WITH MANUAL ENTRY
# -------------------------------
all_miners = ["AEM","NEM","GOLD","WPM","FNV","PAAS","KGC","SSRM","RGLD","HL","CDE","SAND","OR","EXK","AGI","BTG","HMY","IAG"]

# Manual ticker input box
manual_input = st.text_input(
    "Enter tickers manually (comma/space separated) → e.g. AEM PAAS WPM GOLD",
    placeholder="AEM, PAAS, WPM, GOLD"
)

# Preset selection (only shown if no manual input)
if not manual_input.strip():
    col1, col2 = st.columns([1, 3])
    with col1:
        preset = st.selectbox("Quick List:", ["Top Producers", "Royalty", "All Miners"])
        if preset == "Top Producers": default = ["AEM","NEM","GOLD","KGC","PAAS","SSRM"]
        elif preset == "Royalty": default = ["WPM","FNV","RGLD","SAND","OR"]
        else: default = all_miners[:10]
    with col2:
        selected = st.multiselect("Or pick from list:", all_miners, default=default)
    tickers = selected
else:
    # Parse manual input
    raw = manual_input.replace(",", " ").upper().split()
    tickers = [t.strip() for t in raw if t.strip()]

if not tickers:
    st.info("Enter tickers above or select from the list")
    st.stop()

st.markdown(f"**Analyzing {len(tickers)} stocks:** {', '.join(tickers)}")

# -------------------------------
# FETCH DATA
# -------------------------------
@st.cache_data(ttl=1800)
def get_data(tickers):
    rows = []
    for t in tickers:
        try:
            stock = yf.Ticker(t)
            info = stock.info
            hist = stock.history(period="60d")

            price = info.get("currentPrice") or info.get("regularMarketPrice") or np.nan

            # RSI (14-day)
            delta = hist["Close"].diff()
            gain = delta.clip(lower=0).rolling(14).mean()
            loss = -delta.clip(upper=0).rolling(14).mean()
            rs = gain / loss
            rsi = 100 - (100 / (1 + rs))
            current_rsi = round(rsi.iloc[-1], 1) if len(rsi) > 0 else np.nan

            # Real dividend
            divs = stock.dividends
            recent = divs[divs.index > divs.index.max() - pd.Timedelta(days=400)]
            last_div = recent.iloc[-1] if not recent.empty else 0
            annual_div = last_div * 4 if last_div > 0 else 0
            yield_pct = round((annual_div / price) * 100, 2) if price and annual_div > 0 else 0

            rows.append({
                "Ticker": t,
                "Name": info.get("longName", t),
                "Price": price,
                "Yield %": yield_pct,
                "Annual Div $": round(annual_div, 3),
                "Trailing P/E": info.get("trailingPE"),
                "P/B": info.get("priceToBook"),
                "Debt/Equity": info.get("debtToEquity"),
                "ROE %": round((info.get("returnOnEquity") or 0)*100, 1),
                "RSI (14)": current_rsi,
            })
        except Exception as e:
            st.warning(f"{t}: {e}")
            rows.append({"Ticker": t, "Name": "Error", "Price": np.nan, "Yield %": np.nan, "Annual Div $": np.nan,
                         "Trailing P/E": np.nan, "P/B": np.nan, "Debt/Equity": np.nan, "ROE %": np.nan, "RSI (14)": np.nan})
    return pd.DataFrame(rows).set_index("Ticker")

df = get_data(tickers)

# -------------------------------
# 1. VALUATION TABLE (NO SCORE)
# -------------------------------
st.subheader("1. Relative Fair Value (Trailing P/E + P/B)")

peer_pe = df["Trailing P/E"].median(skipna=True)
peer_pb = df["P/B"].median(skipna=True)

val_results = []
for idx, row in df.iterrows():
    rel_pe = row["Price"] * (peer_pe / row["Trailing P/E"]) if pd.notna(row["Trailing P/E"]) and row["Trailing P/E"] > 0 else np.nan
    rel_pb = row["Price"] * (peer_pb / row["P/B"]) if pd.notna(row["P/B"]) and row["P/B"] > 0 else np.nan
    fair = np.nanmean([rel_pe, rel_pb])
    upside = (fair / row["Price"] - 1) * 100 if pd.notna(fair) and pd.notna(row["Price"]) else np.nan

    val_results.append({
        "Ticker": idx,
        "Price": row["Price"],
        "P/E Fair": rel_pe,
        "P/B Fair": rel_pb,
        "Fair Value": fair,
        "Upside %": upside,
    })

val_df = pd.DataFrame(val_results).round(2).set_index("Ticker")
styled_val = val_df.style.format({
    "Price": "${:,.2f}", "P/E Fair": "${:,.2f}", "P/B Fair": "${:,.2f}",
    "Fair Value": "${:,.2f}", "Upside %": "{:+.1f}%"
}, na_rep="—").background_gradient(subset=["Upside %"], cmap="RdYlGn")
st.dataframe(styled_val, use_container_width=True)

# -------------------------------
# 2. FUNDAMENTAL + TECHNICAL TABLE WITH SCORE
# -------------------------------
st.subheader("2. RSI + Yield + Debt Safety + Fundamental Score")

score_results = []
for idx, row in df.iterrows():
    score = 0
    points = 0

    # P/E
    pe = row["Trailing P/E"]
    if pd.notna(pe) and pe > 0:
        if pe < 10: score += 40
        elif pe < 15: score += 30
        elif pe < 20: score += 15
        points += 1

    # P/B
    pb = row["P/B"]
    if pd.notna(pb) and pb > 0:
        if pb < 1.0: score += 40
        elif pb < 1.5: score += 30
        elif pb < 2.0: score += 15
        points += 1

    # Debt/Equity — REQUIRED
    de = row["Debt/Equity"]
    if pd.notna(de):
        if de < 20: score += 20
        elif de < 50: score += 10
        points += 1
    else:
        score = np.nan

    fund_score = int(round(score, 0)) if pd.notna(score) else "—"

    score_results.append({
        "Ticker": idx,
        "RSI (14)": row["RSI (14)"],
        "Yield %": row["Yield %"],
        "Debt/Equity": row["Debt/Equity"],
        "Fundamental Score": fund_score,
    })

score_df = pd.DataFrame(score_results).set_index("Ticker")

def color_fund(val):
    if val == "—": return ""
    if val >= 80: return "background-color: #d4edda; font-weight: bold"
    if val >= 60: return "background-color: #fff3cd"
    return "background-color: #f8d7da"

styled_score = score_df.style.format({
    "RSI (14)": "{:.1f}", "Yield %": "{:.2f}%", "Debt/Equity": "{:.1f}"
}, na_rep="—").applymap(color_fund, subset=["Fundamental Score"])
st.dataframe(styled_score, use_container_width=True)

# -------------------------------
# PLOT – USING FUNDAMENTAL SCORE
# -------------------------------
st.subheader("3. Best Investment Opportunities (Highest Score)")

valid = score_df["Fundamental Score"].replace("—", np.nan).dropna()
if not valid.empty:
    plot_df = score_df.loc[valid.index]
    scores = plot_df["Fundamental Score"].astype(int)
    colors = ["#27ae60" if s >= 80 else "#f39c12" if s >= 60 else "#e74c3c" for s in scores]

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh(plot_df.index[::-1], scores[::-1], color=colors[::-1], height=0.7)
    ax.set_xlabel("Fundamental Score (0–100) – Higher = Safer & Cheaper")
    ax.set_title("Green = Strong Buy • Yellow = Fair • Red = Risky")
    ax.grid(axis='x', alpha=0.3)
    for i, (t, s) in enumerate(zip(plot_df.index[::-1], scores[::-1])):
        ax.text(s + 1, i, f"{s}", va='center', fontweight='bold')
    st.pyplot(fig)
else:
    st.info("No stocks have complete Debt/Equity data → no scores available")

st.success("Manual ticker entry • RSI • Debt-aware score • Pure value investing")
st.caption("Data: Yahoo Finance • Nov 30 2025 • Not financial advice")
