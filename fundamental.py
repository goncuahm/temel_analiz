import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time  # For spinners

# -------------------------------
# PAGE CONFIGURATION
# -------------------------------
st.set_page_config(page_title="Fundamental Valuation App", layout="wide", page_icon="📈")
st.title("📊 Fundamental Analysis & Fair Value Estimator")
st.markdown("---")

# -------------------------------
# USER INPUTS
# -------------------------------
st.sidebar.header("⚙️ Model Parameters")
default_tickers = ["AEM", "O"]
tickers_input = st.sidebar.text_input(
    "Enter company tickers (comma-separated, e.g., AEM, O):",
    value=", ".join(default_tickers)
)
companies = [t.strip().upper() for t in tickers_input.split(",") if t.strip()]

# Global parameters with overrides
global_discount_rate = st.sidebar.number_input("Global Discount Rate (e.g., 0.05 = 5%)", value=0.05, step=0.01, help="Cost of capital; higher for riskier stocks")
global_terminal_growth = st.sidebar.number_input("Global Terminal Growth Rate", value=0.03, step=0.005, help="Long-term growth; typically 2-4%")
use_yf_growth = st.sidebar.checkbox("Use Yahoo Finance growth estimates as default", value=True, help="Auto-pull revenue/earnings growth; override below if needed")
st.sidebar.markdown("---")
st.sidebar.info("💡 Adjust per-stock growth below for custom scenarios. Cache refreshes every 1 hour.")

# Per-stock overrides (expandable)
st.sidebar.subheader("Per-Stock Overrides")
overrides = {}
for ticker in companies:
    with st.sidebar.expander(f"{ticker} Growth Rates"):
        overrides[ticker] = {
            "revenue_growth": st.number_input(f"{ticker} Revenue Growth", value=0.05, step=0.01, key=f"rg_{ticker}"),
            "earnings_growth": st.number_input(f"{ticker} Earnings Growth", value=0.03, step=0.01, key=f"eg_{ticker}")
        }

# -------------------------------
# DOWNLOAD FUNDAMENTAL DATA (with caching & error handling)
# -------------------------------
@st.cache_data(ttl=3600)  # Cache for 1 hour
def get_fundamentals(tickers):
    data = {}
    for ticker in tickers:
        try:
            with st.spinner(f"Fetching data for {ticker}..."):
                stock = yf.Ticker(ticker)
                info = stock.info
                # Core fields with fallbacks
                current_price = info.get("currentPrice", np.nan)
                eps_trailing = info.get("trailingEps", np.nan)
                pe_trailing = info.get("trailingPE", np.nan)
                # Calculate P/E if missing
                if pd.isna(pe_trailing) or pe_trailing == 0:
                    if pd.notna(current_price) and pd.notna(eps_trailing) and eps_trailing != 0:
                        pe_trailing = current_price / eps_trailing
                eps_forward = info.get("forwardEps", eps_trailing)  # Prefer forward for forecasts
                book_value = info.get("bookValue", np.nan)
                pb_ratio = info.get("priceToBook", np.nan)
                dividend_yield = info.get("dividendYield", 0) or 0
                revenue_growth = info.get("revenueGrowth", 0.05)  # Default 5%
                earnings_growth = info.get("earningsGrowth", 0.03)  # Default 3%
                free_cashflow = info.get("freeCashflow", np.nan)
                shares_outstanding = info.get("sharesOutstanding", np.nan)
                data[ticker] = {
                    "current_price": current_price,
                    "eps_trailing": eps_trailing,
                    "eps_forward": eps_forward,
                    "pe_trailing": pe_trailing,
                    "book_value": book_value,
                    "pb_ratio": pb_ratio,
                    "dividend_yield": dividend_yield,
                    "revenue_growth": revenue_growth,
                    "earnings_growth": earnings_growth,
                    "free_cashflow": free_cashflow,
                    "shares_outstanding": shares_outstanding,
                }
        except Exception as e:
            st.warning(f"⚠️ Failed to fetch {ticker}: {e}. Using defaults.")
            data[ticker] = {k: np.nan for k in ["current_price", "eps_trailing", "eps_forward", "pe_trailing", "book_value", "pb_ratio", "dividend_yield", "revenue_growth", "earnings_growth", "free_cashflow", "shares_outstanding"]}
    return pd.DataFrame(data).T

st.subheader("1️⃣ Company Fundamental Data")
if not companies:
    st.warning("Enter at least one ticker to proceed.")
else:
    fund_df = get_fundamentals(companies)
    if fund_df.empty:
        st.error("No data fetched. Check ticker symbols.")
    else:
        st.dataframe(fund_df.style.format("{:.2f}").highlight_null("yellow"))

# -------------------------------
# VALUATION MODELS (Improved with Forward Estimates & Validation)
# -------------------------------
def dcf_fair_value(fcf, growth_rate, discount_rate, terminal_growth, shares_outstanding):
    """Simple 5-year DCF with validation."""
    if pd.isna(fcf) or pd.isna(shares_outstanding) or fcf <= 0 or discount_rate <= terminal_growth:
        return np.nan
    cashflows = []
    for year in range(1, 6):
        fcf_year = fcf * (1 + growth_rate) ** year
        cashflows.append(fcf_year / ((1 + discount_rate) ** year))
    terminal_value = cashflows[-1] * (1 + terminal_growth) / (discount_rate - terminal_growth)
    total_value = sum(cashflows) + terminal_value / ((1 + discount_rate) ** 5)
    return total_value / shares_outstanding

def gordon_growth_model(dividend, required_return, growth_rate):
    """Gordon Growth Model (DDM) with validation."""
    if dividend <= 0 or required_return <= growth_rate:
        return np.nan
    return dividend * (1 + growth_rate) / (required_return - growth_rate)

def relative_valuation(eps, peer_pe, book_value, peer_pb):
    """Peer-based relative valuation."""
    pe_value = eps * peer_pe if pd.notna(eps) and pd.notna(peer_pe) else np.nan
    pb_value = book_value * peer_pb if pd.notna(book_value) and pd.notna(peer_pb) else np.nan
    return (pe_value + pb_value) / 2 if pd.notna(pe_value) and pd.notna(pb_value) else np.nan

# -------------------------------
# FAIR VALUE CALCULATION (Per-Stock Interactive)
# -------------------------------
st.subheader("2️⃣ Fair Value Estimates")
if fund_df.empty:
    st.stop()

peer_pe = fund_df["pe_trailing"].mean(skipna=True)
peer_pb = fund_df["pb_ratio"].mean(skipna=True)
st.info(f"📈 Peer Averages: P/E = {peer_pe:.2f}, P/B = {peer_pb:.2f}")

valuation_results = {}
for ticker in companies:
    with st.expander(f"🔧 Customize {ticker} Assumptions"):
        row = fund_df.loc[ticker]
        # Use yf growth if checked, else override
        revenue_growth = overrides.get(ticker, {}).get("revenue_growth", row["revenue_growth"]) if not use_yf_growth else row["revenue_growth"]
        earnings_growth = overrides.get(ticker, {}).get("earnings_growth", row["earnings_growth"]) if not use_yf_growth else row["earnings_growth"]
        st.write(f"Revenue Growth: {revenue_growth:.2%} | Earnings Growth: {earnings_growth:.2%}")
    
    # Calculations
    fcf = row["free_cashflow"]
    shares = row["shares_outstanding"]
    fair_dcf = dcf_fair_value(fcf, revenue_growth, global_discount_rate, global_terminal_growth, shares)
    
    annual_dividend = row["dividend_yield"] * row["current_price"]
    fair_ddm = gordon_growth_model(annual_dividend, global_discount_rate, earnings_growth)
    
    fair_relative = relative_valuation(row["eps_forward"], peer_pe, row["book_value"], peer_pb)
    
    valuation_results[ticker] = {
        "DCF Value": fair_dcf,
        "DDM Value": fair_ddm,
        "Relative Value": fair_relative,
        "Current Price": row["current_price"],
    }

valuation_df = pd.DataFrame(valuation_results).T
valuation_df["Average Fair Value"] = valuation_df[["DCF Value", "DDM Value", "Relative Value"]].mean(axis=1, skipna=True)
valuation_df["Upside (%)"] = ((valuation_df["Average Fair Value"] / valuation_df["Current Price"]) - 1) * 100

st.dataframe(valuation_df.style.format("{:.2f}").highlight_max(axis=1, subset=["Upside (%)"], color="lightgreen"))

# -------------------------------
# VISUALIZATION (Improved Plot)
# -------------------------------
st.subheader("3️⃣ Fair Value vs. Current Price Comparison")
if not valuation_df.empty:
    fig, ax = plt.subplots(figsize=(12, 6))
    tickers = valuation_df.index
    x = np.arange(len(tickers))
    width = 0.2
    metrics = ["Current Price", "DCF Value", "DDM Value", "Relative Value"]
    num_metrics = len(metrics)
    
    for i, metric in enumerate(metrics):
        ax.bar(x + i * width - width * (num_metrics - 1) / 2, valuation_df[metric], width, label=metric, alpha=0.8)
    
    ax.axhline(y=0, color='k', linestyle='-', alpha=0.3)  # Zero line for reference
    ax.set_title("Valuation Models Comparison")
    ax.set_ylabel("Price ($)")
    ax.set_xticks(x)
    ax.set_xticklabels(tickers, rotation=45)
    ax.legend()
    ax.grid(True, axis='y', linestyle='--', alpha=0.7)
    
    # Add upside labels
    for i, ticker in enumerate(tickers):
        upside = valuation_df.loc[ticker, "Upside (%)"]
        ax.text(i, valuation_df.loc[ticker, "Current Price"] + 5, f"{upside:.1f}%", ha='center', va='bottom', fontweight='bold')
    
    st.pyplot(fig)
else:
    st.warning("No data to plot. Enter valid tickers.")

# -------------------------------
# INSIGHTS & SUMMARY
# -------------------------------
st.subheader("4️⃣ Quick Insights")
if not valuation_df.empty:
    avg_upside = valuation_df["Upside (%)"].mean(skipna=True)
    undervalued = valuation_df[valuation_df["Upside (%)"] > 20].index.tolist()
    overvalued = valuation_df[valuation_df["Upside (%)"] < -10].index.tolist()
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Average Upside Potential", f"{avg_upside:.1f}%")
    with col2:
        st.metric("Undervalued Stocks", f"{len(undervalued)}", delta=len(undervalued))
    with col3:
        st.metric("Overvalued Stocks", f"{len(overvalued)}", delta=len(overvalued))
    
    if undervalued:
        st.success(f"🚀 **Buy Candidates:** {', '.join(undervalued)}")
    if overvalued:
        st.warning(f"⚠️ **Sell/Trim:** {', '.join(overvalued)}")
    
    st.markdown("---")
    st.info("💡 **How to Use:** Run multiple scenarios by changing growth rates. If Upside >20%, consider buying on RSI dips. Always verify with latest earnings.")

st.markdown("---")
st.caption("Data from Yahoo Finance | Models are estimates – not financial advice. © 2025 Dr. Ahmet Göncü")




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


