import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# -------------------------------
# PAGE CONFIGURATION
# -------------------------------
st.set_page_config(page_title="Mining & Dividend Stock Valuation", layout="wide", page_icon="gold_bar")
st.title("Fundamental Fair Value Calculator")
st.markdown("### For Dividend-Paying Mining & Royalty Companies (and any stock)")

# -------------------------------
# USER INPUTS
# -------------------------------
st.sidebar.header("Model Parameters")
default_tickers = ["AEM", "WPM", "FNV", "PAAS"]  # Top dividend mining/royalty stocks
tickers_input = st.sidebar.text_input(
    "Enter tickers (comma-separated):",
    value=", ".join(default_tickers),
    help="Default: AEM (Agnico Eagle), WPM (Wheaton), FNV (Franco-Nevada), PAAS (Pan American Silver)"
)
companies = [t.strip().upper() for t in tickers_input.split(",") if t.strip()]

global_discount_rate = st.sidebar.number_input(
    "Global Discount Rate (Cost of Equity)", 
    value=0.065, min_value=0.04, max_value=0.15, step=0.005, 
    help="6–8% for stable miners/royalties, 9–12% for producers"
)
global_terminal_growth = st.sidebar.number_input(
    "Terminal (Perpetual) Growth Rate", 
    value=0.03, min_value=0.0, max_value=0.05, step=0.005,
    help="Usually 2–4% (long-term inflation + world GDP)"
)
use_yf_growth = st.sidebar.checkbox("Use Yahoo Finance growth estimates by default", value=True)

st.sidebar.markdown("---")
st.sidebar.info("You can override growth per stock in the expandable sections below.")

# Per-stock growth overrides
overrides = {}
for ticker in companies:
    with st.sidebar.expander(f"{ticker} Growth Override"):
        overrides[ticker] = {
            "revenue_growth": st.number_input(f"{ticker} Revenue Growth", value=0.06, step=0.01, key=f"rg_{ticker}"),
            "earnings_growth": st.number_input(f"{ticker} Earnings Growth (for DDM)", value=0.04, step=0.01, key=f"eg_{ticker}")
        }

# -------------------------------
# FETCH DATA
# -------------------------------
@st.cache_data(ttl=3600)
def get_fundamentals(tickers):
    data = {}
    for ticker in tickers:
        try:
            stock = yf.Ticker(ticker)
            info = stock.info
            price = info.get("currentPrice") or info.get("regularMarketPrice")
            data[ticker] = {
                "Name": info.get("longName", ticker),
                "Current Price": price,
                "EPS (Forward)": info.get("forwardEps", info.get("trailingEps")),
                "Dividend Yield (%)": (info.get("dividendYield") or 0) * 100,
                "Forward Dividend": info.get("forwardDividend") or (info.get("dividendYield") or 0) * price,
                "P/E (Trailing)": info.get("trailingPE"),
                "P/B": info.get("priceToBook"),
                "Book Value": info.get("bookValue"),
                "Free Cash Flow (TTM)": info.get("freeCashflow"),
                "Shares Outstanding": info.get("sharesOutstanding"),
                "Revenue Growth": info.get("revenueGrowth", 0.06),
                "Earnings Growth": info.get("earningsGrowth", 0.04),
            }
        except:
            st.warning(f"Could not fetch {ticker}")
            data[ticker] = {k: np.nan for k in ["Name", "Current Price", "EPS (Forward)", "Dividend Yield (%)", "Forward Dividend", "P/E (Trailing)", "P/B", "Book Value", "Free Cash Flow (TTM)", "Shares Outstanding", "Revenue Growth", "Earnings Growth"]}
    return pd.DataFrame(data).T

st.subheader("1. Fundamental Data")
if not companies:
    st.stop()
fund_df = get_fundamentals(companies)
st.dataframe(fund_df.style.format("{:,.2f}"), use_container_width=True)

# -------------------------------
# VALUATION MODELS
# -------------------------------
def dcf_valuation(fcf, growth, discount_rate, terminal_growth, shares):
    if not all([fcf, shares, fcf > 0, shares > 0, discount_rate > terminal_growth]):
        return np.nan
    projected = [fcf * (1 + growth)**t for t in range(1, 6)]
    discounted = [cf / (1 + discount_rate)**t for t, cf in enumerate(projected, 1)]
    terminal = projected[-1] * (1 + terminal_growth) / (discount_rate - terminal_growth)
    tv_discounted = terminal / (1 + discount_rate)**5
    equity_value = sum(discounted) + tv_discounted
    return equity_value / shares

def gordon_growth(dividend, discount_rate, growth):
    if dividend <= 0 or discount_rate <= growth:
        return np.nan
    return dividend * (1 + growth) / (discount_rate - growth)

# -------------------------------
# CALCULATE FAIR VALUES
# -------------------------------
st.subheader("2. Fair Value Estimates")
results = {}
peer_pe = fund_df["P/E (Trailing)"].mean(skipna=True)
peer_pb = fund_df["P/B"].mean(skipna=True)

for ticker in companies:
    row = fund_df.loc[ticker]
    rev_g = overrides.get(ticker, {}).get("revenue_growth", row["Revenue Growth"]) if not use_yf_growth else row["Revenue Growth"]
    earn_g = overrides.get(ticker, {}).get("earnings_growth", row["Earnings Growth"]) if not use_yf_growth else row["Earnings Growth"]

    with st.expander(f"{ticker} – {row['Name']}"):
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Current Price", f"${row['Current Price']:.2f}")
            st.metric("Dividend Yield", f"{row['Dividend Yield (%)']:.2f}%")
        with col2:
            st.write(f"Revenue Growth: **{rev_g:.1%}**")
            st.write(f"Earnings Growth (DDM): **{earn_g:.1%}**")

        # DCF
        fair_dcf = dcf_valuation(row["Free Cash Flow (TTM)"], rev_g, global_discount_rate, global_terminal_growth, row["Shares Outstanding"])
        # DDM
        fair_ddm = gordon_growth(row["Forward Dividend"], global_discount_rate, earn_g)
        # Relative
        rel_eps = row["EPS (Forward)"] * peer_pe if pd.notna(row["EPS (Forward)"]) and pd.notna(peer_pe) else np.nan
        rel_pb = row["Book Value"] * peer_pb if pd.notna(row["Book Value"]) and pd.notna(peer_pb) else np.nan
        fair_relative = np.nanmean([rel_eps, rel_pb]) if pd.notna(rel_eps) or pd.notna(rel_pb) else np.nan

        results[ticker] = {
            "Current Price": row["Current Price"],
            "DCF Value": fair_dcf,
            "DDM Value": fair_ddm,
            "Relative Value": fair_relative,
        }

val_df = pd.DataFrame(results).T
val_df["Average Fair Value"] = val_df[["DCF Value", "DDM Value", "Relative Value"]].mean(axis=1, skipna=True)
val_df["Upside (%)"] = (val_df["Average Fair Value"] / val_df["Current Price"] - 1) * 100

st.dataframe(val_df.style.format("{:,.2f}").background_gradient(subset=["Upside (%)"], cmap="RdYlGn"), use_container_width=True)

# -------------------------------
# PLOT
# -------------------------------
st.subheader("3. Fair Value vs Current Price")
fig, ax = plt.subplots(figsize=(12, 6))
x = np.arange(len(val_df))
width = 0.18
metrics = ["Current Price", "DCF Value", "DDM Value", "Relative Value"]

for i, col in enumerate(metrics):
    values = val_df[col].fillna(0)
    ax.bar(x + i*width - 1.5*width, values, width, label=col, alpha=0.9)

ax.set_ylabel("Price ($)")
ax.set_title("Valuation Model Comparison")
ax.set_xticks(x)
ax.set_xticklabels(val_df.index)
ax.legend()
ax.grid(True, axis='y', alpha=0.3)

for i, ticker in enumerate(val_df.index):
    upside = val_df.loc[ticker, "Upside (%)"]
    color = "green" if upside > 20 else "orange" if upside > 0 else "red"
    ax.text(i, val_df.loc[ticker, "Current Price"] + 5, f"{upside:+.1f}%", ha='center', fontweight='bold', color=color)

st.pyplot(fig)

# -------------------------------
# EXPANDABLE EXPLANATION & GUIDE
# -------------------------------
with st.expander("How This Tool Works – Formulas & Parameter Guide (Click to Expand)", expanded=False):
    st.markdown("""
    ### Valuation Models Used

    | Model | Formula | Best For | Notes |
    |------|--------|---------|-------|
    | **Gordon Growth (DDM)** | `D₁ / (r − g)` | Stable dividend payers (REITs, royalty companies) | D₁ = next year dividend, r = discount rate, g = perpetual growth |
    | **Two-Stage DCF** | 5-year explicit FCF forecast + Terminal Value | Growing producers (miners, renewables) | Uses revenue growth for FCF, terminal growth 2–4% |
    | **Relative Valuation** | Average of (Forward EPS × Peer P/E) and (Book Value × Peer P/B) | Reality check in bull/bear markets | Prevents extreme over/under-valuation |

    **Final Fair Value** = Average of all valid models → smooths biases.

    ### Recommended Parameter Choices

    | Parameter | Typical Range | Mining/Royalty Recommendation |
    |---------|---------------|-------------------------------|
    | **Discount Rate** | 5–12% | **6–8%** for royalties (FNV, WPM), **8–10%** for producers (AEM, PAAS) |
    | **Terminal Growth** | 2–4% | Use **3%** (long-term global inflation + GDP) |
    | **Growth Rates** | Use Yahoo Finance values | Override only if you have strong conviction (e.g., new mine coming online) |

    ### How to Use This Tool Like a Pro
    1. Start with the **four default mining dividend stocks**.
    2. Look for **Upside > 20%** + **clean Renko uptrend** + **RSI < 45** → strong buy.
    3. If a stock shows **negative upside** → trim on RSI > 65.
    4. Re-run monthly or after earnings.

    This tool gives you **institutional-grade fair value** in seconds — for free.
    """)

st.markdown("---")
st.caption("Data: Yahoo Finance • Models: Standard academic/institutional formulas • Not financial advice • © 2025 Your Name")








# import streamlit as st
# import yfinance as yf
# import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt

# # -------------------------------
# # PAGE CONFIGURATION
# # -------------------------------
# st.set_page_config(page_title="Fundamental Valuation App", layout="wide")
# st.title("📊 Fundamental Analysis & Fair Value Estimator")

# # -------------------------------
# # USER INPUTS
# # -------------------------------
# st.sidebar.header("⚙️ Model Parameters")
# default_tickers = ["AEM", "O"]

# tickers_input = st.sidebar.text_input(
#     "Enter company tickers separated by commas (e.g. AEM, O):",
#     ", ".join(default_tickers)
# )
# companies = [t.strip().upper() for t in tickers_input.split(",") if t.strip()]

# st.sidebar.write("---")
# discount_rate_input = st.sidebar.number_input("Discount Rate (e.g. 0.05 = 5%)", value=0.05, step=0.01)
# terminal_growth_input = st.sidebar.number_input("Terminal Growth Rate", value=0.03, step=0.005)

# st.sidebar.write("---")
# st.sidebar.info("💡 You can adjust these parameters to test different assumptions.")

# # -------------------------------
# # DOWNLOAD FUNDAMENTAL DATA
# # -------------------------------
# @st.cache_data
# def get_fundamentals(tickers):
#     data = {}
#     for ticker in tickers:
#         try:
#             stock = yf.Ticker(ticker)
#             info = stock.info

#             current_price = info.get("currentPrice", np.nan)
#             eps = info.get("trailingEps", np.nan)
#             pe_ratio = info.get("trailingPE", np.nan)

#             # ✅ If P/E ratio is missing, calculate it manually
#             if (pd.isna(pe_ratio) or pe_ratio == 0) and pd.notna(current_price) and pd.notna(eps) and eps != 0:
#                 pe_ratio = current_price / eps

#             data[ticker] = {
#                 "current_price": current_price,
#                 "eps": eps,
#                 "pe_ratio": pe_ratio,
#                 "book_value": info.get("bookValue", np.nan),
#                 "pb_ratio": info.get("priceToBook", np.nan),
#                 "dividend_yield": (info.get("dividendYield", 0) or 0),
#                 "beta": info.get("beta", 1.0),
#                 "roe": info.get("returnOnEquity", np.nan),
#                 "revenue_growth": info.get("revenueGrowth", 0.05),
#                 "free_cashflow": info.get("freeCashflow", np.nan),
#                 "shares_outstanding": info.get("sharesOutstanding", np.nan),
#             }
#         except Exception as e:
#             st.warning(f"⚠️ Could not fetch data for {ticker}: {e}")
#     return pd.DataFrame(data).T

# st.subheader("1️⃣ Company Fundamental Data")
# fund_df = get_fundamentals(companies)
# st.dataframe(fund_df.style.format("{:.2f}"))

# # -------------------------------
# # VALUATION MODELS
# # -------------------------------
# def dcf_fair_value(fcf, growth_rate, discount_rate, terminal_growth, shares_outstanding):
#     """Simple 5-year DCF model."""
#     cashflows = []
#     for year in range(1, 6):
#         fcf = fcf * (1 + growth_rate)
#         cashflows.append(fcf / ((1 + discount_rate) ** year))
#     terminal_value = cashflows[-1] * (1 + terminal_growth) / (discount_rate - terminal_growth)
#     total_value = sum(cashflows) + terminal_value / ((1 + discount_rate) ** 5)
#     return total_value / shares_outstanding


# def gordon_growth_model(dividend, required_return, growth_rate):
#     """Gordon Growth Model (for dividend-paying stocks)."""
#     if dividend == 0 or required_return <= growth_rate:
#         return np.nan
#     return dividend * (1 + growth_rate) / (required_return - growth_rate)

 
# def relative_valuation(eps, peer_pe, book_value, peer_pb):
#     """Estimate value using peer P/E and P/B averages."""
#     pe_value = eps * peer_pe
#     pb_value = book_value * peer_pb
#     return (pe_value + pb_value) / 2


# # -------------------------------
# # FAIR VALUE CALCULATION
# # -------------------------------
# st.subheader("2️⃣ Valuation Parameters and Fair Value Estimates")

# valuation_results = {}

# peer_pe = fund_df["pe_ratio"].mean(skipna=True)
# peer_pb = fund_df["pb_ratio"].mean(skipna=True)

# for ticker in companies:
#     row = fund_df.loc[ticker]
#     st.markdown(f"### {ticker}")

#     # Allow user to modify inputs for each stock
#     col1, col2, col3 = st.columns(3)
#     with col1:
#         growth_rate = st.number_input(f"{ticker} Revenue Growth", value=float(row["revenue_growth"] or 0.05), key=f"gr_{ticker}")
#     with col2:
#         discount_rate = st.number_input(f"{ticker} Discount Rate", value=discount_rate_input, key=f"dr_{ticker}")
#     with col3:
#         terminal_growth = st.number_input(f"{ticker} Terminal Growth", value=terminal_growth_input, key=f"tg_{ticker}")

#     fcf = row["free_cashflow"]
#     shares = row["shares_outstanding"]

#     # --- DCF Valuation ---
#     if pd.notna(fcf) and pd.notna(shares) and fcf > 0:
#         fair_dcf = dcf_fair_value(fcf, growth_rate, discount_rate, terminal_growth, shares)
#     else:
#         fair_dcf = np.nan

#     # --- Dividend Model ---
#     annual_dividend = row["dividend_yield"] * row["current_price"] / 100
#     fair_ddm = gordon_growth_model(annual_dividend, discount_rate, terminal_growth)

#     # --- Relative Valuation ---
#     fair_relative = relative_valuation(row["eps"], peer_pe, row["book_value"], peer_pb)

#     valuation_results[ticker] = {
#         "DCF Value": fair_dcf,
#         "DDM Value": fair_ddm,
#         "Relative Value": fair_relative,
#         "Current Price": row["current_price"],
#     }

# valuation_df = pd.DataFrame(valuation_results).T
# valuation_df["Average Fair Value"] = valuation_df[["DCF Value", "DDM Value", "Relative Value"]].mean(axis=1)
# valuation_df["Upside (%)"] = (valuation_df["Average Fair Value"] / valuation_df["Current Price"] - 1) * 100

# st.dataframe(valuation_df.style.format("{:.2f}"))

# # -------------------------------
# # VISUALIZATION
# # -------------------------------
# st.subheader("3️⃣ Fair Value vs Market Price Comparison")

# # Prepare data for grouped bar chart
# plot_df = valuation_df[["Current Price", "DCF Value", "DDM Value", "Relative Value"]].copy()
# tickers = plot_df.index.tolist()
# metrics = plot_df.columns.tolist()
# num_metrics = len(metrics)

# x = np.arange(len(tickers))  # the label locations
# width = 0.2  # width of each bar

# fig, ax = plt.subplots(figsize=(10, 6))

# # Plot each valuation metric side by side
# for i, metric in enumerate(metrics):
#     ax.bar(x + i * width - width * (num_metrics - 1) / 2,
#            plot_df[metric],
#            width,
#            label=metric)

# ax.set_title("Fair Value Comparison by Model")
# ax.set_ylabel("Price (Local Currency)")
# ax.set_xticks(x)
# ax.set_xticklabels(tickers, rotation=0)
# ax.legend()
# ax.grid(True, axis='y', linestyle='--', alpha=0.7)

# st.pyplot(fig)


