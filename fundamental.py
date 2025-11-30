import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

st.set_page_config(page_title="Mining Fair Value Calculator", layout="wide", page_icon="gold_bar")
st.title("Gold & Silver Mining Fair Value Calculator")
st.markdown("### Dividend yields now display correctly (1.6% shows as 1.6%)")

# Sidebar
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

            # --- CORRECT DIVIDEND HANDLING ---
            div_yield_decimal = info.get("dividendYield") or 0.0  # keep as decimal
            fwd_div = info.get("forwardDividend")
            if fwd_div is None or pd.isna(fwd_div):
                fwd_div = div_yield_decimal * price if not pd.isna(price) else 0.0

            # Ensure dividend yield aligns with forward dividend
            if not pd.isna(price) and price > 0:
                div_yield_decimal = fwd_div / price

            # Clamp unrealistic yields
            if div_yield_decimal < 0: div_yield_decimal = 0.0
            if div_yield_decimal > 0.2:  # max 20% dividend yield
                div_yield_decimal = 0.2
                fwd_div = div_yield_decimal * price

            annual_div = fwd_div  # for DDM
            # No *100 here — display formatting will handle %

            rev_growth = info.get("revenueGrowth") or 0.06
            earn_growth = info.get("earningsGrowth") or 0.04

            # Free cash flow and shares
            try:
                fcf_val = float(info.get("freeCashflow"))
                fcf_m = fcf_val / 1e6 if abs(fcf_val) >= 1e6 else fcf_val
            except:
                fcf_m = np.nan

            try:
                shares_m = float(info.get("sharesOutstanding")) / 1e6
            except:
                shares_m = np.nan

            data[ticker] = {
                "Price": price,
                "Yield": div_yield_decimal,  # keep as decimal
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
            data[ticker] = {k: np.nan for k in ["Price", "Yield", "Annual Div $",
                                                "Rev Growth", "Earn Growth", "FCF TTM $M",
                                                "Shares M", "Fwd EPS", "P/E", "P/B", "Book Value", "Name"]}

    df = pd.DataFrame.from_dict(data, orient="index")
    df.index.name = "Ticker"
    return df

if not companies:
    st.stop()

df = get_fundamentals(companies)

# -------------------------------
# Display
# -------------------------------
st.subheader("1. Verified Fundamental Data")
st.dataframe(df.style.format({
    "Price": "${:,.2f}",
    "Yield": "{:.2%}",  # display as percentage
    "Annual Div $": "${:,.3f}",
    "Rev Growth": "{:.1%}",
    "Earn Growth": "{:.1%}",
    "FCF TTM $M": "{:,.0f}M",
    "Shares M": "{:,.1f}M",
    "Fwd EPS": "${:,.2f}",
    "P/E": "{:.1f}",
    "P/B": "{:.2f}",
    "Book Value": "${:,.2f}",
}, na_rep="—"), use_container_width=True)

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
peer_pe = df["P/E"].median(skipna=True) or 15
peer_pb = df["P/B"].median(skipna=True) or 1.8

results = []
for ticker, row in df.iterrows():
    g = min(row.get("Earn Growth", 0.04), discount_rate - 0.005)
    ddm = ddm_value(row.get("Annual Div $", 0.0), discount_rate, g)

    rel_eps = row["Fwd EPS"] * peer_pe if pd.notna(row["Fwd EPS"]) else np.nan
    rel_pb = row["Book Value"] * peer_pb if pd.notna(row["Book Value"]) else np.nan

    # Cap relative value to 2× price to avoid huge spikes
    price = row["Price"]
    if pd.notna(price):
        if pd.notna(rel_eps): rel_eps = min(rel_eps, 2*price)
        if pd.notna(rel_pb): rel_pb = min(rel_pb, 2*price)

    relative = np.nanmean([rel_eps, rel_pb])
    fair_value = np.nanmean([ddm, relative])
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

    yld = row.get("Yield")  # now decimal
    if pd.notna(yld):
        if yld > 0.04: score += 25
        elif yld > 0.025: score += 15
        elif yld > 0.01: score += 8
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

# -------------------------------
# Display Valuation
# -------------------------------
st.subheader("3. Valuation & Scores")
st.dataframe(val_df.style.format({
    "Price": "${:,.2f}",
    "DDM Value": "${:,.2f}",
    "Relative Value": "${:,.2f}",
    "Fair Value": "${:,.2f}",
    "Upside %": "{:+.1f}%"
}, na_rep="—"), use_container_width=True)
