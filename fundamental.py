import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# -------------------------------
# PAGE CONFIG
# -------------------------------
st.set_page_config(page_title="Mining Stock Fair Value Calculator", layout="wide", page_icon="gold_bar")
st.title("Mining & Dividend Stock Fair Value Calculator")
st.markdown("### For Gold/Silver Royalty & Producer Valuation Tool")

# -------------------------------
# SIDEBAR
# -------------------------------
st.sidebar.header("Model Settings")
default_tickers = ["AEM", "WPM", "FNV", "PAAS"]
tickers_input = st.sidebar.text_input(
    "Tickers (comma-separated):",
    value=", ".join(default_tickers),
    help="AEM, WPM, FNV, PAAS = top dividend mining stocks"
)
companies = [t.strip().upper() for t in tickers_input.split(",") if t.strip()]

global_discount_rate = st.sidebar.slider("Discount Rate (Cost of Equity)", 0.04, 0.15, 0.075, 0.005,
                                         help="6–8% for royalties, 8–10% for producers")
global_terminal_growth = st.sidebar.slider("Terminal Growth Rate", 0.00, 0.06, 0.03, 0.005,
                                           help="Long-term inflation + GDP ≈ 3%")
use_yf_growth = st.sidebar.checkbox("Use Yahoo Finance growth rates", value=True)

# Per-stock override
overrides = {}
for ticker in companies:
    with st.sidebar.expander(f"{ticker} Growth Override"):
        overrides[ticker] = {
            "rev": st.number_input(f"{ticker} Revenue Growth", 0.00, 0.30, 0.06, 0.01, key=f"r_{ticker}"),
            "earn": st.number_input(f"{ticker} Earnings Growth", 0.00, 0.20, 0.04, 0.01, key=f"e_{ticker}")
        }

# -------------------------------
# FETCH DATA
# -------------------------------
@st.cache_data(ttl=3600, show_spinner="Fetching latest data...")
def get_fundamentals(tickers):
    rows = []
    for ticker in tickers:
        try:
            stock = yf.Ticker(ticker)
            info = stock.info
            price = info.get("currentPrice") or info.get("regularMarketPrice", np.nan)

            rows.append({
                "Ticker": ticker,
                "Name": info.get("longName", ticker),
                "Price": price,
                "Fwd EPS": info.get("forwardEps"),
                "Yield %": (info.get("dividendYield") or 0) * 100,
                "Fwd Div": info.get("forwardDividend") or (info.get("dividendYield", 0) * price),
                "P/E": info.get("trailingPE"),
                "P/B": info.get("priceToBook"),
                "Book Value": info.get("bookValue"),
                "FCF TTM": info.get("freeCashflow"),
                "Shares Outstanding": info.get("sharesOutstanding"),
                "Rev Growth": info.get("revenueGrowth", 0.06),
                "Earn Growth": info.get("earningsGrowth", 0.04),
            })
        except:
            rows.append({k: np.nan for k in ["Ticker","Name","Price","Fwd EPS","Yield %","Fwd Div","P/E","P/B","Book Value","FCF TTM","Shares Outstanding","Rev Growth","Earn Growth"]})
            rows[-1]["Ticker"] = ticker
    return pd.DataFrame(rows).set_index("Ticker")

if not companies:
    st.stop()

with st.spinner("Downloading data from Yahoo Finance..."):
    fund_df = get_fundamentals(companies)

# SAFE DISPLAY — this fixes the crash
numeric_cols = fund_df.select_dtypes(include=[np.number]).columns
styled_df = fund_df.style.format("{:,.2f}", subset=numeric_cols, na_rep="—")
st.subheader("1. Latest Fundamental Data")
st.dataframe(styled_df, use_container_width=True)

# -------------------------------
# VALUATION FUNCTIONS
# -------------------------------
def dcf_value(fcf, g, r, g_term, shares):
    if not all([fcf, shares]) or fcf <= 0 or shares <= 0 or r <= g_term:
        return np.nan
    pv = sum(fcf * (1+g)**t / (1+r)**t for t in range(1, 6))
    tv = fcf * (1+g)**5 * (1+g_term / (r - g_term)
    pv_tv = tv / (1+r)**5
    return (pv + pv_tv) / shares

def ddm_value(div, r, g):
    if div <= 0 or r <= g:
        return np.nan
    return div * (1 + g) / (r - g)

# -------------------------------
# CALCULATIONS
# -------------------------------
st.subheader("2. Fair Value Estimates")
results = []

peer_pe = fund_df["P/E"].mean(skipna=True)
peer_pb = fund_df["P/B"].mean(skipna=True)

for ticker in fund_df.itertuples():
    ticker = ticker.Index
    rev_g = overrides.get(ticker, {}).get("rev", fund_df.loc[ticker, "Rev Growth"]) if not use_yf_growth else fund_df.loc[ticker, "Rev Growth"]
    earn_g = overrides.get(ticker, {}).get("earn", fund_df.loc[ticker, "Earn Growth"]) if not use_yf_growth else fund_df.loc[ticker, "Earn Growth"]

    dcf = dcf_value(fund_df.loc[ticker, "FCF TTM"], rev_g, global_discount_rate, global_terminal_growth, fund_df.loc[ticker, "Shares Outstanding"])
    ddm = ddm_value(fund_df.loc[ticker, "Fwd Div"], global_discount_rate, earn_g)
    rel = np.nanmean([
        fund_df.loc[ticker, "Fwd EPS"] * peer_pe,
        fund_df.loc[ticker, "Book Value"] * peer_pb
    ]) if pd.notna(fund_df.loc[ticker, "Fwd EPS"]) else np.nan

    avg_fair = np.nanmean([dcf, ddm, rel])
    upside = (avg_fair / fund_df.loc[ticker, "Price"] - 1) * 100 if pd.notna(avg_fair) else np.nan

    results.append({
        "Ticker": ticker,
        "Current Price": fund_df.loc[ticker, "Price"],
        "DCF Value": dcf,
        "DDM Value": ddm,
        "Relative Value": rel,
        "Average Fair Value": avg_fair,
        "Upside %": upside
    })

val_df = pd.DataFrame(results).set_index("Ticker")
val_df = val_df.round(2)

st.dataframe(
    val_df.style.format("{:,.2f}", na_rep="—")
          .background_gradient(subset=["Upside %"], cmap="RdYlGn"),
    use_container_width=True
)

# -------------------------------
# CHART
# -------------------------------
st.subheader("3. Visual Comparison")
fig, ax = plt.subplots(figsize=(12, 6))
x = np.arange(len(val_df))
width = 0.18
for i, col in enumerate(["Current Price", "DCF Value", "DDM Value", "Relative Value"]):
    vals = val_df[col].fillna(0)
    ax.bar(x + i*width - 1.5*width, vals, width, label=col)

ax.set_ylabel("Price ($)")
ax.set_title("Fair Value vs Current Price")
ax.set_xticks(x)
ax.set_xticklabels(val_df.index)
ax.legend()
ax.grid(True, axis='y', alpha=0.3)

for i, tk in enumerate(val_df.index):
    up = val_df.loc[tk, "Upside %"]
    col = "green" if up > 20 else "orange" if up > 0 else "red"
    ax.text(i, val_df.loc[tk, "Current Price"], f"{up:+.1f}%", ha='center', va='bottom', fontweight='bold', color=col)

st.pyplot(fig)

# -------------------------------
# EXPANDABLE GUIDE
# -------------------------------
with st.expander("How This Calculator Works – Formulas & Best Practices", expanded=False):
    st.markdown("""
    ### Valuation Models Explained

    | Model                  | Formula                                      | When to Trust It Most                     |
    |------------------------|----------------------------------------------|--------------------------------------------|
    | **Gordon Growth (DDM)**| `Next Dividend × (1+g) / (r − g)`             | Royalty/streaming companies (FNV, WPM)     |
    | **Two-Stage DCF**      | 5-yr FCF forecast + Terminal Value           | Producers with growth projects (AEM, PAAS)|
    | **Relative (P/E + P/B)**| Avg of (EPS×Peer P/E) and (BV×Peer P/B)      | Quick sanity check in any market           |

    **Final Fair Value** = Average of all three valid models

    ### Recommended Settings (Mining Sector)
    - **Discount Rate**: 6.5–8% for royalties, 8–10% for producers
    - **Terminal Growth**: 3% (long-term inflation)
    - **Growth Rates**: Accept Yahoo Finance values unless you have strong new info

    **Buy Rule**: Upside > 25% + Renko green + RSI < 45 → strong buy
    """)

st.markdown("---")
st.caption("Data: Yahoo Finance • Not financial advice • Updated Nov 2025")








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


