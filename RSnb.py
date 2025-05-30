import streamlit as st
import pandas as pd
import numpy as np
from mlxtend.frequent_patterns import apriori, association_rules
from sklearn.decomposition import TruncatedSVD
from prophet import Prophet
import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt

# ------------------------- Load & Preprocess Data ---------------------------
def load_data():
    url = "https://raw.githubusercontent.com/dgorgo/sales_app/main/sales_df.csv"
    try:
        df = pd.read_csv(url)
        df.columns = df.columns.str.lower().str.replace(" ", "_")
        df['delivered_date'] = pd.to_datetime(df['delivered_date'], errors='coerce')
        df.dropna(inplace=True)
    except Exception as e:
        st.error(f"Error loading data: {e}")
        st.stop()
    return df

# -------------------- Market Basket Analysis --------------------
def market_basket_analysis(df, container):
    with container:
        st.subheader("🧺 Market Basket Analysis")
        required = {'order_id', 'sku_code', 'delivered_qty'}
        if not required.issubset(df.columns):
            st.error(f"Missing required columns: {required}")
            return
        basket = (
            df.groupby(['order_id', 'sku_code'])['delivered_qty']
            .sum().unstack(fill_value=0)
            .applymap(lambda x: 1 if x > 0 else 0)
        )
        frequent_items = apriori(basket, min_support=0.02, use_colnames=True)
        rules = association_rules(frequent_items, metric="lift", min_threshold=1.0)

        st.write("📌 Frequent Itemsets")
        st.dataframe(frequent_items.sort_values('support', ascending=False))

# -------------------- Recommendation System --------------------
def recommendation_system(df, container):
    with container:
        st.subheader("🎯 SKU Recommendation System")
        required = {'salesman_code', 'sku_code', 'delivered_qty'}
        if not required.issubset(df.columns):
            st.error(f"Missing columns: {required}")
            return
        df['salesman_code'] = pd.to_numeric(df['salesman_code'], errors='coerce')
        df.dropna(subset=['salesman_code'], inplace=True)
        df['salesman_code'] = df['salesman_code'].astype(int)

        pivot = df.pivot_table(index='salesman_code', columns='sku_code',
                               values='delivered_qty', aggfunc='sum', fill_value=0)

        svd = TruncatedSVD(n_components=10, random_state=42)
        user_factors = svd.fit_transform(pivot)
        item_factors = svd.components_
        preds = np.dot(user_factors, item_factors)

        pred_df = pd.DataFrame(preds, index=pivot.index, columns=pivot.columns)
        selected_user = st.selectbox("Select Salesman Code", pred_df.index)
        if st.button("Recommend SKUs"):
            actuals = pivot.loc[selected_user]
            scores = pred_df.loc[selected_user].copy()
            scores[actuals > 0] = -np.inf
            top10 = scores.nlargest(10).reset_index()
            top10.columns = ['SKU_Code', 'Recommendation Score']
            st.write("🔍 Top Recommended SKUs:")
            st.dataframe(top10)

# -------------------- Forecasting --------------------
def forecast_sales(df, container):
    with container:
        st.subheader("📈 Forecast SKU Sales (Prophet)")
        if 'sku_code' not in df.columns or 'delivered_date' not in df.columns:
            st.warning("Missing required columns for forecasting.")
            return
        sku = st.selectbox("Select SKU Code", df['sku_code'].unique())
        sku_df = df[df['sku_code'] == sku].groupby('delivered_date')['delivered_qty'].sum().reset_index()
        sku_df = sku_df.rename(columns={'delivered_date': 'ds', 'delivered_qty': 'y'})

        model = Prophet()
        model.fit(sku_df)

        future = model.make_future_dataframe(periods=30)
        forecast = model.predict(future)

        fig = px.line(forecast, x='ds', y='yhat', title=f"📦 Forecast for SKU: {sku}")
        st.plotly_chart(fig)

# -------------------- Brand and SKU Performance --------------------
def brand_sku_performance(df, container):
    with container:
        st.subheader("📊 Brand & SKU Performance")
        if 'brand' in df.columns and 'redistribution_value' in df.columns:
            brand_perf = df.groupby('brand')['redistribution_value'].sum().reset_index()
            sku_perf = df.groupby('sku_code')['redistribution_value'].sum().reset_index()
            st.plotly_chart(px.bar(brand_perf, x='brand', y='redistribution_value', title="Brand Performance"))
            st.plotly_chart(px.bar(sku_perf, x='sku_code', y='redistribution_value', title="SKU Performance"))
        else:
            st.warning("Missing 'brand' or 'redistribution_value' columns.")

# -------------------- Salesman Performance (ADDED FUNCTION) --------------------
def Salesman_performance(df, container):
    with container:
        st.subheader("💼 Salesman Performance Overview")
        if 'salesman_code' in df.columns and 'delivered_qty' in df.columns:
            performance = df.groupby('salesman_code')['delivered_qty'].sum().reset_index()
            top_salesmen = performance.sort_values(by='delivered_qty', ascending=False)
            st.write("Top Performing Salesmen:")
            st.dataframe(top_salesmen)
            st.plotly_chart(px.bar(top_salesmen, x='salesman_code', y='delivered_qty',
                                   title='Salesman Delivered Quantity'))
        else:
            st.warning("Missing 'salesman_code' or 'delivered_qty' columns.")

# -------------------- Main App Layout --------------------
def main():
    st.set_page_config(page_title="Sales Intelligence Dashboard", layout="wide")
    st.title("🧠 Market Cart Dashboard")

    df = load_data()

    tabs = st.tabs([
        "📦 Market Basket", "🎯 Recommendations", "📈 Forecasting",
        "📊 Brand Performance", "💼 Salesman Performance"
    ])

    market_basket_analysis(df, tabs[0])
    recommendation_system(df, tabs[1])
    forecast_sales(df, tabs[2])
    brand_sku_performance(df, tabs[3])
    Salesman_performance(df, tabs[4])

if __name__ == "__main__":
    main()
