import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.express as px

st.set_page_config(page_title="Undervalued Stock Screener", layout="wide")

st.title("Undervalued Stock Screener")
st.write("This dashboard screens stocks using valuation, quality, and momentum metrics.")

default_tickers = "AAPL, MSFT, GOOGL, AMZN, META, NVDA, AMD, JPM, BAC, XOM, CVX, JNJ, PFE, KO, WMT, COST"

tickers_input = st.text_area(
    "Enter stock tickers separated by commas:",
    value=default_tickers
)

tickers = [ticker.strip().upper() for ticker in tickers_input.split(",") if ticker.strip()]

st.sidebar.header("Scoring Weights")

value_weight = st.sidebar.slider("Value Weight", 0.0, 1.0, 0.4, 0.05)
quality_weight = st.sidebar.slider("Quality Weight", 0.0, 1.0, 0.4, 0.05)
momentum_weight = st.sidebar.slider("Momentum Weight", 0.0, 1.0, 0.2, 0.05)

total_weight = value_weight + quality_weight + momentum_weight

if total_weight == 0:
    st.error("Total weight cannot be zero.")
    st.stop()

value_weight = value_weight / total_weight
quality_weight = quality_weight / total_weight
momentum_weight = momentum_weight / total_weight

@st.cache_data
def get_stock_data(tickers):
    rows = []

    for ticker in tickers:
        try:
            stock = yf.Ticker(ticker)
            info = stock.info

            hist = stock.history(period="1y")

            if hist.empty:
                one_year_return = np.nan
            else:
                first_price = hist["Close"].iloc[0]
                last_price = hist["Close"].iloc[-1]
                one_year_return = (last_price - first_price) / first_price

            rows.append({
                "Ticker": ticker,
                "Company": info.get("shortName", np.nan),
                "Sector": info.get("sector", np.nan),
                "Market Cap": info.get("marketCap", np.nan),
                "Price": info.get("currentPrice", np.nan),
                "P/E": info.get("trailingPE", np.nan),
                "Forward P/E": info.get("forwardPE", np.nan),
                "P/S": info.get("priceToSalesTrailing12Months", np.nan),
                "P/B": info.get("priceToBook", np.nan),
                "Profit Margin": info.get("profitMargins", np.nan),
                "ROE": info.get("returnOnEquity", np.nan),
                "Debt to Equity": info.get("debtToEquity", np.nan),
                "One Year Return": one_year_return
            })

        except Exception:
            rows.append({
                "Ticker": ticker,
                "Company": np.nan,
                "Sector": np.nan,
                "Market Cap": np.nan,
                "Price": np.nan,
                "P/E": np.nan,
                "Forward P/E": np.nan,
                "P/S": np.nan,
                "P/B": np.nan,
                "Profit Margin": np.nan,
                "ROE": np.nan,
                "Debt to Equity": np.nan,
                "One Year Return": np.nan
            })

    return pd.DataFrame(rows)

def percentile_score(series, higher_is_better=True):
    clean_series = series.replace([np.inf, -np.inf], np.nan)

    if higher_is_better:
        return clean_series.rank(pct=True) * 100
    else:
        return (1 - clean_series.rank(pct=True)) * 100

def calculate_scores(df):
    scored = df.copy()

    scored["P/E Score"] = percentile_score(scored["P/E"], higher_is_better=False)
    scored["P/S Score"] = percentile_score(scored["P/S"], higher_is_better=False)
    scored["P/B Score"] = percentile_score(scored["P/B"], higher_is_better=False)

    scored["Value Score"] = scored[["P/E Score", "P/S Score", "P/B Score"]].mean(axis=1)

    scored["Profit Margin Score"] = percentile_score(scored["Profit Margin"], higher_is_better=True)
    scored["ROE Score"] = percentile_score(scored["ROE"], higher_is_better=True)
    scored["Debt Score"] = percentile_score(scored["Debt to Equity"], higher_is_better=False)

    scored["Quality Score"] = scored[["Profit Margin Score", "ROE Score", "Debt Score"]].mean(axis=1)

    scored["Momentum Score"] = percentile_score(scored["One Year Return"], higher_is_better=True)

    scored["Final Score"] = (
        value_weight * scored["Value Score"] +
        quality_weight * scored["Quality Score"] +
        momentum_weight * scored["Momentum Score"]
    )

    return scored.sort_values("Final Score", ascending=False)

if st.button("Run Stock Screener"):
    if len(tickers) == 0:
        st.error("Please enter at least one ticker.")
    else:
        with st.spinner("Pulling data and calculating scores..."):
            df = get_stock_data(tickers)
            scored_df = calculate_scores(df)

        st.subheader("Ranked Stock Screener Results")

        display_cols = [
            "Ticker", "Company", "Sector", "Market Cap", "Price",
            "P/E", "P/S", "P/B", "Profit Margin", "ROE",
            "Debt to Equity", "One Year Return",
            "Value Score", "Quality Score", "Momentum Score", "Final Score"
        ]

        st.dataframe(
            scored_df[display_cols],
            use_container_width=True
        )

        top_stock = scored_df.iloc[0]

        st.subheader("Top Ranked Stock")
        st.write(f"**{top_stock['Ticker']}** ranks highest based on the selected scoring weights.")
        st.write(
            f"It received a final score of **{top_stock['Final Score']:.2f}**, "
            f"with a value score of **{top_stock['Value Score']:.2f}**, "
            f"quality score of **{top_stock['Quality Score']:.2f}**, "
            f"and momentum score of **{top_stock['Momentum Score']:.2f}**."
        )

        st.subheader("Final Score by Stock")

        fig_score = px.bar(
            scored_df,
            x="Ticker",
            y="Final Score",
            color="Sector",
            title="Stock Attractiveness Score"
        )

        st.plotly_chart(fig_score, use_container_width=True)

        st.subheader("Value vs Quality")

        fig_scatter = px.scatter(
            scored_df,
            x="Value Score",
            y="Quality Score",
            size="Market Cap",
            color="Sector",
            hover_name="Ticker",
            title="Value Score vs Quality Score"
        )

        st.plotly_chart(fig_scatter, use_container_width=True)

        st.subheader("Valuation Multiples")

        valuation_df = scored_df[["Ticker", "P/E", "P/S", "P/B"]].set_index("Ticker")

        fig_val = px.bar(
            valuation_df,
            barmode="group",
            title="Comparison of Valuation Multiples"
        )

        st.plotly_chart(fig_val, use_container_width=True)

        csv = scored_df.to_csv(index=False).encode("utf-8")

        st.download_button(
            label="Download Results as CSV",
            data=csv,
            file_name="stock_screener_results.csv",
            mime="text/csv"
        )

else:
    st.info("Enter tickers and click 'Run Stock Screener' to begin.")