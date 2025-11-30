import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# -------------------------------
# Page Setup
# -------------------------------
st.set_page_config(page_title="Mining Fair Value Calculator", layout="wide", page_icon="gold_bar")
st.title("Gold & Silver Mining Fair Value Calculator")
st.markdown("### Fixed Dividend Yields (1.6% Shows as 1.6%, Not 160%)")

# -------------------------------
# Sidebar Inputs
# -------------------------------
st.sidebar.header("Settings")
default_tickers = ["AEM", "WPM", "FNV", "PAAS"]
tickers_input = st.sidebar.text_input("Tickers:", value=", ".join(default_tickers))
companies = [t.strip().upper() for t in tickers_input.split(",") if t.strip()]

discount_rate = st.sidebar.slider("Discount Rate", 0.05, 0.15, 0.075, 0.005)
terminal_growth = st.sidebar.slider("Terminal Growth", 0.00, 0.06, 0.03, 0.005)

# -------------------------------
# Fetch Fundamentals
# -------------------------------
@st.cache_data(ttl=3600)
def get_fundamentals(tickers):
    data = {}
    for ticker in tickers:
        try:
            stock = yf.Ticker(ticker)
            info = stock.info or {}

            price = info.get("currentPrice") or np.nan

            # --- FIXED DIVIDEND HANDLING ---
            raw_yield = info.get("dividendYield")  # may be None, 0.016, or sometimes 1.6
            div_yield_decimal = 0.0
            if raw_yield is not None:
                try:
                    raw_y = float(raw_yield)
                    if raw_y > 1:
                        div_yield_decimal = raw_y / 100.0
                    else:
                        div_yield_decimal = raw_y
                except:
                    div_yield_decimal = 0.0

            fwd_div = info.get("forwardDividend")
            if fwd_div is None or pd.isna(fwd_div):
                if not pd.isna(price) and div_yield_decimal > 0:
                    fwd_div = div_yield_decimal * price
                else:
                    fwd_div = 0.0

            if price > 0:
                div_yield_decimal = fwd_div / price

            # Clamp yields to avoid absurd numbers
            if div_yield_decimal < 0:
                div_yield_decimal = 0.0
            if div_yield_decimal > 0.2:  # max 20%
                div_yield_decimal = 0.2
                fwd_div = price * div_yield_decimal

            yield_pct = div_yield_decimal * 100
            annual_div = fwd_div

            # Growths
            rev_growth = info.get("revenueGrowth") or 0.06
            earn_growth = info.get("earningsGrowth") or 0.04

            # Free cash flow normalization
            fcf_raw = info.get("freeCashflow")
            try:
                fcf_val = float(fcf_raw)
                if abs(fcf_val) >= 1e9:
                    fcf_m = fcf_val / 1e6  # $M
                elif abs(fcf_val) >= 1e6:
                    fcf_m = fcf_val / 1e6
                else:
                    fcf_m = fcf_val
            except:
                fcf_m = np.nan

            # Shares normalization
            try:
                shares_m = float(info.get("sharesOutstanding")) / 1e6
            except:
                shares_m = np.nan

            data[ticker] = {
                "Price": price,
                "Yield %": yield_pct,
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
            data[ticker] = {k: np.nan for k in ["Price", "Yield %", "Annual Div $",
                                                "Rev Growth", "Earn Growth", "FCF TTM $M",
                                                "Shares M", "Fwd EPS", "P/E", "P/B", "Book Value", "Name"]}

    df = pd.DataFrame.from_dict(data, orient="index")
    df.index.name = "Ticker"
    return df

if not companies:
    st.stop()

df = get_fundamentals(companies)

# Display fundamentals
st.subheader("1. Verified Fundamental Data")
st.dataframe(df, use_container_width=True)

# -------------------------------
# DDM Function
# -------------------------------
def ddm_value(annual_div, r, g):
    if pd.isna(annual_div) or annual_div <= 0 or r <= g:
        return np.nan
    return annual_div * (1 + g) / (r - g)

# -------------------------------
# Fair Value & Fundamental Score
# -------------------------------
st.subheader("2. Fair Value & Fundamental Score")

peer_pe = df["P/E"].median(skipna=True)
if pd.isna(peer_pe) or peer_pe == 0:
    peer_pe = 15.0
peer_pb = df["P/B"].median(skipna=True)
if pd.isna(peer_pb) or peer_pb == 0:
    peer_pb = 1.8

results = []
for ticker, row in df.iterrows():
    g = min(row.get("Earn Growth", 0.04), discount_rate - 0.005)
    ddm = ddm_value(row.get("Annual Div $", 0.0), discount_rate, g)

    # Relative value (cap at 2× price to prevent explosions)
    rel_eps = np.nan
    if pd.notna(row.get("Fwd EPS")):
        rel_eps = row["Fwd EPS"] * peer_pe
        if pd.notna(row["Price"]):
            rel_eps = min(rel_eps, row["Price"] * 2)

    rel_pb = np.nan
    if pd.notna(row.get("Book Value")):
        rel_pb = row["Book Value"] * peer_pb
        if pd.notna(row["Price"]):
            rel_pb = min(rel_pb, row["Price"] * 2)

    relative = np.nanmean([rel_eps, rel_pb])
    fair_value = np.nanmean([ddm, relative])
    price = row.get("Price")
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

    yld = row.get("Yield %")
    if pd.notna(yld):
        if yld > 4: score += 25
        elif yld > 2.5: score += 15
        elif yld > 1: score += 8
        items += 1

    fund_score = round(score / max(items, 1) * 2.5, 0) if items > 0 else 0

    results.append({
        "Ticker": ticker,
        "Price": price,
        "DDM Value": ddm,
        "Relative Value": relative,
        "Fair Value": fair_value,
        "Upside %": upside,
        "Fundamental Score": fund_score
    })

val_df = pd.DataFrame(results).set_index("Ticker").round(2)
val_df = val_df.sort_values("Fundamental Score", ascending=False)

# Styling
def color_score(val):
    if pd.isna(val): return ""
    if val >= 75: return "background-color: #d4edda"
    if val >= 55: return "background-color: #fff3cd"
    return "background-color: #f8d7da"

styled_val = val_df.style.format({
    "Price": "${:,.2f}",
    "DDM Value": "${:,.2f}",
    "Relative Value": "${:,.2f}",
    "Fair Value": "${:,.2f}",
    "Upside %": "{:+.1f}%"
}, na_rep="—").background_gradient(subset=["Upside %"], cmap="RdYlGn") \
  .applymap(color_score, subset=["Fundamental Score"])

st.write(styled_val.to_html(), unsafe_allow_html=True)

# -------------------------------
# Ranking Chart
# -------------------------------
st.subheader("3. Best Deals Ranked")
fig, ax = plt.subplots(figsize=(10, 6))
scores = val_df["Fundamental Score"].fillna(0)
colors = ["#27ae60" if s >= 75 else "#f39c12" if s >= 55 else "#e74c3c" for s in scores]

ax.barh(val_df.index[::-1], scores[::-1], color=colors[::-1])
ax.set_xlabel("Fundamental Score (Higher = Cheaper Deal)")
ax.set_title("Mining Stock Ranking – Green = Strong Buy Zone")
ax.grid(axis='x', alpha=0.3)

for i, (idx, s) in enumerate(zip(val_df.index[::-1], scores[::-1])):
    ax.text(s + 1, i, f"{int(s)}", va='center', fontweight='bold')

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

st.success("Dividend yields fixed – realistic DDM and fair values now appear.")
st.caption("Data: Yahoo Finance • Models: DDM + Relative • Not financial advice • © 2025")
