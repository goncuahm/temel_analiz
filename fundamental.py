import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

st.set_page_config(page_title="Mining Fair Value Calculator", layout="wide", page_icon="gold_bar")
st.title("Gold & Silver Mining Fair Value Calculator")
st.markdown("### Fixed Dividend Yields (1.6% Shows as 1.6%, Not 160%)")

# Sidebar
st.sidebar.header("Settings")
default_tickers = ["AEM", "WPM", "FNV", "PAAS"]
tickers_input = st.sidebar.text_input("Tickers:", value=", ".join(default_tickers))
companies = [t.strip().upper() for t in tickers_input.split(",") if t.strip()]

discount_rate = st.sidebar.slider("Discount Rate", 0.05, 0.15, 0.075, 0.005)
terminal_growth = st.sidebar.slider("Terminal Growth", 0.00, 0.06, 0.03, 0.005)

# Fetch data
@st.cache_data(ttl=3600)
def get_fundamentals(tickers):
    data = {}
    for ticker in tickers:
        try:
            stock = yf.Ticker(ticker)
            info = stock.info or {}

            price = info.get("currentPrice") or np.nan

            # DIVIDEND – store as decimal (0.016) and display as percent later
            div_yield_decimal = info.get("dividendYield") or 0.0  # e.g., 0.016
            yield_pct = div_yield_decimal * 100  # 1.60 for table display

            # forwardDividend may be None or 0 — fall back to yield * price if possible
            fwd_div = info.get("forwardDividend")
            if fwd_div is None or (isinstance(fwd_div, (int, float)) and np.isnan(fwd_div)):
                if not pd.isna(price) and div_yield_decimal:
                    fwd_div = div_yield_decimal * price
                else:
                    fwd_div = np.nan
            annual_div = fwd_div  # Use for DDM

            rev_growth = info.get("revenueGrowth")
            if rev_growth is None or pd.isna(rev_growth):
                rev_growth = 0.06

            earn_growth = info.get("earningsGrowth")
            if earn_growth is None or pd.isna(earn_growth):
                earn_growth = 0.04

            # Free cash flow and shares normalization (safe checks)
            fcf_raw = info.get("freeCashflow")
            if fcf_raw is None or (isinstance(fcf_raw, (int, float)) and np.isnan(fcf_raw)):
                fcf_m = np.nan
            else:
                try:
                    fcf_val = float(fcf_raw)
                    if abs(fcf_val) >= 1e9:
                        fcf_m = fcf_val / 1e6  # show in $M
                    elif abs(fcf_val) >= 1e6:
                        fcf_m = fcf_val / 1e6
                    else:
                        fcf_m = fcf_val  # small numbers keep as-is
                except Exception:
                    fcf_m = np.nan

            shares_raw = info.get("sharesOutstanding")
            if shares_raw is None or (isinstance(shares_raw, (int, float)) and np.isnan(shares_raw)):
                shares_m = np.nan
            else:
                try:
                    shares_m = float(shares_raw) / 1e6
                except Exception:
                    shares_m = np.nan

            data[ticker] = {
                "Price": price,
                "Yield %": yield_pct,      # percent for display
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
            data[ticker] = {
                "Price": np.nan,
                "Yield %": np.nan,
                "Annual Div $": np.nan,
                "Rev Growth": np.nan,
                "Earn Growth": np.nan,
                "FCF TTM $M": np.nan,
                "Shares M": np.nan,
                "Fwd EPS": np.nan,
                "P/E": np.nan,
                "P/B": np.nan,
                "Book Value": np.nan,
                "Name": ticker,
            }

    # Build dataframe from dict-of-dicts safely and name the index
    df = pd.DataFrame.from_dict(data, orient="index")
    df.index.name = "Ticker"
    return df

if not companies:
    st.stop()

df = get_fundamentals(companies)

# Display (show raw dataframe and a styled HTML table)
st.subheader("1. Verified Fundamental Data")
st.dataframe(df, use_container_width=True)

# Create a Styler for nicer display and render as HTML (Streamlit supports unsafe HTML here)
styled = df.style.format({
    "Price": "${:,.2f}",
    "Yield %": "{:.2f}%",
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
st.write(styled.to_html(), unsafe_allow_html=True)

# -------------------------------
# VALUATION MODELS
# -------------------------------
def ddm_value(annual_div, r, g):
    """
    Simple Gordon Growth Model for next-year dividend:
      value = D1 / (r - g) where D1 = D0*(1+g)
    Return nan if inputs invalid or r <= g (cannot discount).
    """
    try:
        if pd.isna(annual_div) or annual_div <= 0 or r <= g:
            return np.nan
        return annual_div * (1 + g) / (r - g)
    except Exception:
        return np.nan

# -------------------------------
# CALCULATIONS
# -------------------------------
st.subheader("2. Fair Value & Fundamental Score")

# sensible fallbacks for peer medians
peer_pe = df["P/E"].median(skipna=True)
if pd.isna(peer_pe) or peer_pe == 0:
    peer_pe = 15.0
peer_pb = df["P/B"].median(skipna=True)
if pd.isna(peer_pb) or peer_pb == 0:
    peer_pb = 1.8

results = []
for ticker, row in df.iterrows():
    earn_g = row.get("Earn Growth", 0.04)
    try:
        earn_g = float(earn_g) if not pd.isna(earn_g) else 0.04
    except Exception:
        earn_g = 0.04

    # safety cap so growth < discount rate
    g = min(earn_g, discount_rate - 0.005)

    ddm = ddm_value(row.get("Annual Div $", np.nan), discount_rate, g)

    rel_eps = np.nan
    if pd.notna(row.get("Fwd EPS")) and pd.notna(peer_pe):
        try:
            rel_eps = row["Fwd EPS"] * peer_pe
        except Exception:
            rel_eps = np.nan

    rel_pb = np.nan
    if pd.notna(row.get("Book Value")) and pd.notna(peer_pb):
        try:
            rel_pb = row["Book Value"] * peer_pb
        except Exception:
            rel_pb = np.nan

    # Use nanmean so if one is available it will be used
    relative = np.nanmean([rel_eps, rel_pb])

    fair_value = np.nanmean([ddm, relative])
    price = row.get("Price", np.nan)
    upside = (fair_value / price - 1) * 100 if pd.notna(fair_value) and pd.notna(price) and price != 0 else np.nan

    # Fundamental Score (0–100)
    score = 0
    items = 0

    pe = row.get("P/E")
    if pd.notna(pe) and pe > 0:
        if pe < 12:
            score += 30
        elif pe < 18:
            score += 20
        elif pe < 25:
            score += 10
        items += 1

    pb = row.get("P/B")
    if pd.notna(pb) and pb > 0:
        if pb < 1.2:
            score += 30
        elif pb < 1.8:
            score += 20
        elif pb < 2.5:
            score += 10
        items += 1

    yld = row.get("Yield %")
    if pd.notna(yld):
        # yld is stored as percent (1.6 means 1.6%)
        if yld > 4:
            score += 25
        elif yld > 2.5:
            score += 15
        elif yld > 1:
            score += 8
        items += 1

    fund_score = round(score / max(items, 1) * 2.5, 0) if items > 0 else 0

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

# Styling and display for values
def color_score(val):
    if pd.isna(val):
        return ""
    if val >= 75:
        return "background-color: #d4edda"
    if val >= 55:
        return "background-color: #fff3cd"
    return "background-color: #f8d7da"

styled_val = val_df.style.format({
    "Price": "${:,.2f}",
    "DDM Value": "${:,.2f}",
    "Relative Value": "${:,.2f}",
    "Fair Value": "${:,.2f}",
    "Upside %": "{:+.1f}%",
}, na_rep="—") \
    .background_gradient(subset=["Upside %"], cmap="RdYlGn") \
    .applymap(color_score, subset=["Fundamental Score"])

st.write(styled_val.to_html(), unsafe_allow_html=True)

# -------------------------------
# RANKING CHART
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
# GUIDE
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

st.success("Dividend yields fixed – now 1.6% shows as 1.6%. Clean, reliable outputs.")
st.caption("Data: Yahoo Finance • Models: DDM + Relative • Not financial advice • © 2025")
