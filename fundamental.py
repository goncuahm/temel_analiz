import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# -------------------------------
# PAGE CONFIGURATION
# -------------------------------
st.set_page_config(page_title="Fundamental Valuation App", layout="wide")
st.title("📊 Fundamental Analysis & Fair Value Estimator")

# -------------------------------
# USER INPUTS
# -------------------------------
st.sidebar.header("⚙️ Model Parameters")
default_tickers = ["AYEN.IS", "BASGZ.IS"]

tickers_input = st.sidebar.text_input(
    "Enter company tickers separated by commas (e.g. AYEN.IS, BASGZ.IS):",
    ", ".join(default_tickers)
)
companies = [t.strip().upper() for t in tickers_input.split(",") if t.strip()]

st.sidebar.write("---")
discount_rate_input = st.sidebar.number_input("Discount Rate (e.g. 0.20 = 20%)", value=0.20, step=0.01)
terminal_growth_input = st.sidebar.number_input("Terminal Growth Rate", value=0.03, step=0.005)

st.sidebar.write("---")
st.sidebar.info("💡 You can adjust these parameters to test different assumptions.")

# -------------------------------
# DOWNLOAD FUNDAMENTAL DATA
# -------------------------------
@st.cache_data
def get_fundamentals(tickers):
    data = {}
    for ticker in tickers:
        try:
            stock = yf.Ticker(ticker)
            info = stock.info

            current_price = info.get("currentPrice", np.nan)
            eps = info.get("trailingEps", np.nan)
            pe_ratio = info.get("trailingPE", np.nan)

            # ✅ If P/E ratio is missing, calculate it manually
            if (pd.isna(pe_ratio) or pe_ratio == 0) and pd.notna(current_price) and pd.notna(eps) and eps != 0:
                pe_ratio = current_price / eps

            data[ticker] = {
                "current_price": current_price,
                "eps": eps,
                "pe_ratio": pe_ratio,
                "book_value": info.get("bookValue", np.nan),
                "pb_ratio": info.get("priceToBook", np.nan),
                "dividend_yield": (info.get("dividendYield", 0) or 0),
                "beta": info.get("beta", 1.0),
                "roe": info.get("returnOnEquity", np.nan),
                "revenue_growth": info.get("revenueGrowth", 0.05),
                "free_cashflow": info.get("freeCashflow", np.nan),
                "shares_outstanding": info.get("sharesOutstanding", np.nan),
            }
        except Exception as e:
            st.warning(f"⚠️ Could not fetch data for {ticker}: {e}")
    return pd.DataFrame(data).T

st.subheader("1️⃣ Company Fundamental Data")
fund_df = get_fundamentals(companies)
st.dataframe(fund_df.style.format("{:.2f}"))

# -------------------------------
# VALUATION MODELS
# -------------------------------
def dcf_fair_value(fcf, growth_rate, discount_rate, terminal_growth, shares_outstanding):
    """Simple 5-year DCF model."""
    cashflows = []
    for year in range(1, 6):
        fcf = fcf * (1 + growth_rate)
        cashflows.append(fcf / ((1 + discount_rate) ** year))
    terminal_value = cashflows[-1] * (1 + terminal_growth) / (discount_rate - terminal_growth)
    total_value = sum(cashflows) + terminal_value / ((1 + discount_rate) ** 5)
    return total_value / shares_outstanding


def gordon_growth_model(dividend, required_return, growth_rate):
    """Gordon Growth Model (for dividend-paying stocks)."""
    if dividend == 0 or required_return <= growth_rate:
        return np.nan
    return dividend * (1 + growth_rate) / (required_return - growth_rate)


def relative_valuation(eps, peer_pe, book_value, peer_pb):
    """Estimate value using peer P/E and P/B averages."""
    pe_value = eps * peer_pe
    pb_value = book_value * peer_pb
    return (pe_value + pb_value) / 2


# -------------------------------
# FAIR VALUE CALCULATION
# -------------------------------
st.subheader("2️⃣ Valuation Parameters and Fair Value Estimates")

valuation_results = {}

peer_pe = fund_df["pe_ratio"].mean(skipna=True)
peer_pb = fund_df["pb_ratio"].mean(skipna=True)

for ticker in companies:
    row = fund_df.loc[ticker]
    st.markdown(f"### {ticker}")

    # Allow user to modify inputs for each stock
    col1, col2, col3 = st.columns(3)
    with col1:
        growth_rate = st.number_input(f"{ticker} Revenue Growth", value=float(row["revenue_growth"] or 0.05), key=f"gr_{ticker}")
    with col2:
        discount_rate = st.number_input(f"{ticker} Discount Rate", value=discount_rate_input, key=f"dr_{ticker}")
    with col3:
        terminal_growth = st.number_input(f"{ticker} Terminal Growth", value=terminal_growth_input, key=f"tg_{ticker}")

    fcf = row["free_cashflow"]
    shares = row["shares_outstanding"]

    # --- DCF Valuation ---
    if pd.notna(fcf) and pd.notna(shares) and fcf > 0:
        fair_dcf = dcf_fair_value(fcf, growth_rate, discount_rate, terminal_growth, shares)
    else:
        fair_dcf = np.nan

    # --- Dividend Model ---
    annual_dividend = row["dividend_yield"] * row["current_price"] / 100
    fair_ddm = gordon_growth_model(annual_dividend, discount_rate, terminal_growth)

    # --- Relative Valuation ---
    fair_relative = relative_valuation(row["eps"], peer_pe, row["book_value"], peer_pb)

    valuation_results[ticker] = {
        "DCF Value": fair_dcf,
        "DDM Value": fair_ddm,
        "Relative Value": fair_relative,
        "Current Price": row["current_price"],
    }

valuation_df = pd.DataFrame(valuation_results).T
valuation_df["Average Fair Value"] = valuation_df[["DCF Value", "DDM Value", "Relative Value"]].mean(axis=1)
valuation_df["Upside (%)"] = (valuation_df["Average Fair Value"] / valuation_df["Current Price"] - 1) * 100

st.dataframe(valuation_df.style.format("{:.2f}"))

# -------------------------------
# VISUALIZATION
# -------------------------------
st.subheader("3️⃣ Fair Value vs Market Price Comparison")

# Prepare data for grouped bar chart
plot_df = valuation_df[["Current Price", "DCF Value", "DDM Value", "Relative Value"]].copy()
tickers = plot_df.index.tolist()
metrics = plot_df.columns.tolist()
num_metrics = len(metrics)

x = np.arange(len(tickers))  # the label locations
width = 0.2  # width of each bar

fig, ax = plt.subplots(figsize=(10, 6))

# Plot each valuation metric side by side
for i, metric in enumerate(metrics):
    ax.bar(x + i * width - width * (num_metrics - 1) / 2,
           plot_df[metric],
           width,
           label=metric)

ax.set_title("Fair Value Comparison by Model")
ax.set_ylabel("Price (Local Currency)")
ax.set_xticks(x)
ax.set_xticklabels(tickers, rotation=0)
ax.legend()
ax.grid(True, axis='y', linestyle='--', alpha=0.7)

st.pyplot(fig)


