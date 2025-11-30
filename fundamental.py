import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

st.set_page_config(page_title="Mining Fair Value Calculator", layout="wide", page_icon="gold_bar")
st.title("Gold & Silver Mining Fair Value Calculator")
st.markdown("### Dividend yields fixed — decimal values (0.016 = 1.6%)")

# Sidebar
st.sidebar.header("Settings")
default_tickers = ["AEM", "WPM", "FNV", "PAAS"]
tickers_input = st.sidebar.text_input("Tickers:", value=", ".join(default_tickers))
companies = [t.strip().upper() for t in tickers_input.split(",") if t.strip()]

discount_rate = st.sidebar.slider("Discount Rate", 0.05, 0.15, 0.075, 0.005)
terminal_growth = st.sidebar.slider("Terminal Growth", 0.00, 0.06, 0.03, 0.005)

# -------------------------------
# Fetch fundamentals
# -------------------------------
@st.cache_data(ttl=3600)
def get_fundamentals(tickers):
    data = {}
    for ticker in tickers:
        try:
            stock = yf.Ticker(ticker)
            info = stock.info or {}

            price = info.get("currentPrice") or np.nan

            # --- Dividend handling (decimal, no scaling, no cap) ---
            div_yield_decimal = info.get("dividendYield")
            if div_yield_decimal is None:
                div_yield_decimal = 0.0

            fwd_div = info.get("forwardDividend")
            if fwd_div is not None and not pd.isna(fwd_div):
                annual_div = fwd_div
                if not pd.isna(price) and price > 0:
                    div_yield_decimal = annual_div / price
            else:
                if not pd.isna(price):
                    annual_div = div_yield_decimal * price
                else:
                    annual_div = 0.0

            rev_growth = info.get("revenueGrowth") or 0.06
            earn_growth = info.get("earningsGrowth") or 0.04

            fcf_raw = info.get("freeCashflow")
            if fcf_raw is None or (isinstance(fcf_raw, (int, float)) and pd.isna(fcf_raw)):
                fcf_m = np.nan
            else:
                fcf_val = float(fcf_raw)
                if abs(fcf_val) >= 1e6:
                    fcf_m = fcf_val / 1e6
                else:
                    fcf_m = fcf_val

            shares_raw = info.get("sharesOutstanding")
            shares_m = float(shares_raw)/1e6 if shares_raw else np.nan

            data[ticker] = {
                "Price": price,
                "Yield": div_yield_decimal,      # decimal
                "Annual Div $": annual_div,
                "Rev Growth": rev_growth,
                "Earn Growth": earn_growth,
                "FCF TTM $M": fcf_m,
                "Shares M": shares_m,
                "Fwd EPS": info.get("forwardEps"),
                "P/E": info.get("forwardPE") or info.get("trailingPE"),
                "P/B": info.get("priceToBook"),
                "Book Value": info.get("bookValue"),
                "Name": info.get("longName", ticker),
            }
        except Exception as e:
            st.warning(f"Data issue for {ticker}: {e}")
            data[ticker] = {k: np.nan for k in ["Price","Yield","Annual Div $","Rev Growth","Earn Growth",
                                                "FCF TTM $M","Shares M","Fwd EPS","P/E","P/B","Book Value","Name"]}
    df = pd.DataFrame.from_dict(data, orient="index")
    df.index.name = "Ticker"
    return df

if not companies:
    st.stop()

df = get_fundamentals(companies)

# Display fundamentals
st.subheader("1. Verified Fundamental Data")
styled = df.style.format({
    "Price": "${:,.2f}",
    "Yield": "{:.3f}",           # decimal
    "Annual Div $": "${:,.3f}",
    "Rev Growth": "{:.1%}",
    "Earn Growth": "{:.1%}",
    "FCF TTM $M": "{:,.0f}M",
    "Shares M": "{:,.1f}M",
    "Fwd EPS": "{:,.2f}",
    "P/E": "{:.1f}",
    "P/B": "{:.2f}",
    "Book Value": "${:,.2f}",
}, na_rep="—")
st.dataframe(styled, use_container_width=True)

# -------------------------------
# DDM model
# -------------------------------
def ddm_value(annual_div, r, g):
    if pd.isna(annual_div) or annual_div <= 0 or r <= g:
        return np.nan
    return annual_div * (1 + g) / (r - g)

# -------------------------------
# Fair Value & Fundamental Score
# -------------------------------
st.subheader("2. Fair Value & Fundamental Score")

peer_pe = df["P/E"].median(skipna=True) or 15
peer_pb = df["P/B"].median(skipna=True) or 1.8

results = []
for ticker, row in df.iterrows():
    g = min(row.get("Earn Growth", 0.04), discount_rate - 0.005)
    ddm = ddm_value(row.get("Annual Div $", np.nan), discount_rate, g)

    rel_eps = row["Fwd EPS"] * peer_pe if pd.notna(row["Fwd EPS"]) else np.nan
    rel_pb = row["Book Value"] * peer_pb if pd.notna(row["Book Value"]) else np.nan
    relative = np.nanmean([rel_eps, rel_pb])

    fair_value = np.nanmean([ddm, relative])
    price = row.get("Price", np.nan)
    upside = (fair_value / price - 1) * 100 if pd.notna(fair_value) and pd.notna(price) else np.nan

    # Fundamental score
    score = 0
    items = 0
    pe = row.get("P/E")
    if pd.notna(pe):
        if pe < 12: score += 30
        elif pe < 18: score += 20
        elif pe < 25: score += 10
        items += 1
    pb = row.get("P/B")
    if pd.notna(pb):
        if pb < 1.2: score += 30
        elif pb < 1.8: score += 20
        elif pb < 2.5: score += 10
        items += 1
    yld = row.get("Yield")
    if pd.notna(yld):
        if yld > 0.04: score += 25
        elif yld > 0.025: score += 15
        elif yld > 0.01: score += 8
        items += 1

    fund_score = round(score / max(items,1) * 2.5,0) if items>0 else 0

    results.append({
        "Ticker": ticker,
        "Price": price,
        "DDM Value": ddm,
        "Relative Value": relative,
        "Fair Value": fair_value,
        "Upside %": upside,
        "Fundamental Score": fund_score,
    })

val_df = pd.DataFrame(results).set_index("Ticker").round(2)
val_df = val_df.sort_values("Fundamental Score", ascending=False)

# -------------------------------
# Styling
# -------------------------------
def color_score(val):
    if pd.isna(val):
        return ""
    if val >= 75: return "background-color: #d4edda"
    if val >= 55: return "background-color: #fff3cd"
    return "background-color: #f8d7da"

styled_val = val_df.style.format({
    "Price": "${:,.2f}",
    "DDM Value": "${:,.2f}",
    "Relative Value": "${:,.2f}",
    "Fair Value": "${:,.2f}",
    "Upside %": "{:+.1f}%",
}, na_rep="—").background_gradient(subset=["Upside %"], cmap="RdYlGn").applymap(color_score, subset=["Fundamental Score"])

st.dataframe(styled_val, use_container_width=True)

# -------------------------------
# Ranking Chart
# -------------------------------
st.subheader("3. Best Deals Ranked")
fig, ax = plt.subplots(figsize=(10,6))
scores = val_df["Fundamental Score"].fillna(0)
colors = ["#27ae60" if s>=75 else "#f39c12" if s>=55 else "#e74c3c" for s in scores]

ax.barh(val_df.index[::-1], scores[::-1], color=colors[::-1])
ax.set_xlabel("Fundamental Score (Higher = Cheaper Deal)")
ax.set_title("Mining Stock Ranking – Green = Strong Buy Zone")
ax.grid(axis='x', alpha=0.3)
for i,(idx,s) in enumerate(zip(val_df.index[::-1], scores[::-1])):
    ax.text(s+1, i, f"{int(s)}", va='center', fontweight='bold')
st.pyplot(fig)

# -------------------------------
# Guide
# -------------------------------
with st.expander("How the Fundamental Score Works", expanded=False):
    st.markdown("""
    ### Fundamental Score (0–100) – Valuation Ranking
    | Metric | Points | Logic |
    |--------|--------|-------|
    | Forward P/E | 30/20/10 | <12 = 30pts, <18 = 20pts, <25 = 10pts |
    | P/B Ratio | 30/20/10 | <1.2 = 30pts, <1.8 = 20pts, <2.5 = 10pts |
    | Dividend Yield | 25/15/8 | >4% = 25pts, >2.5% = 15pts, >1% = 8pts |
    | ROE | 15/10 | >15% = 15pts, >10% = 10pts |

    **80–100** = Deep bargain (buy aggressively)  
    **60–79** = Attractive (buy on dips)  
    **<60** = Fair/expensive (hold or sell)
    """)

st.success("Dividend yields are decimals (0.016 = 1.6%) — clean, reliable outputs.")
st.caption("Data: Yahoo Finance • Models: DDM + Relative • Not financial advice • © 2025")
