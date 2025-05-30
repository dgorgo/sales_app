import streamlit as st
import pandas as pd
import numpy as np
from mlxtend.frequent_patterns import apriori
from sklearn.decomposition import TruncatedSVD
import matplotlib.pyplot as plt
import seaborn as sns

# ------------------------- Load & Preprocess Data ---------------------------

def load_data():
    github_url = "https://raw.githubusercontent.com/dgorgo/sales_app/main/sales_df.csv"
    try:
        df = pd.read_csv(github_url)
    except Exception as e:
        st.error(f"Error loading data: {e}")
        st.stop()

    df.columns = df.columns.str.lower().str.replace(" ", "_")
    df['delivered_date'] = pd.to_datetime(df['delivered_date'], dayfirst=True, errors='coerce')
    df.dropna(inplace=True)
    return df

# ------------------------- Feature Modules ----------------------------------

def market_basket_analysis(df, container):
    with container:
        st.subheader("Market Basket Analysis")

        if {'order_id', 'sku_code', 'delivered_qty'}.issubset(df.columns):
            basket = (
                df.groupby(['order_id', 'sku_code'])['delivered_qty']
                .sum()
                .unstack(fill_value=0)
                .applymap(lambda x: 1 if x > 0 else 0)
            )
            frequent_items = apriori(basket, min_support=0.02, use_colnames=True)
            st.write("Top Frequent Itemsets")
            st.dataframe(frequent_items.sort_values('support', ascending=False))
        else:
            st.warning("Missing required columns for MBA")

def sales_forecasting(df, container):
    with container:
        st.subheader("Sales Forecasting")
        st.info("📌 Placeholder: Forecast SKU sales using Prophet/ARIMA")

def market_mix_modeling(df, container):
    with container:
        st.subheader("Market Mix Modeling")
        st.info("📌 Placeholder: Requires marketing spend, pricing, campaign data")

def consumer_analysis(df, container):
    with container:
        st.subheader("Consumer Behavior Analysis")
        st.info("📌 Placeholder: Analyze repeat customers, RFM, segmentation")

def growth_calculations(df, container):
    with container:
        st.subheader("MTD, QTD, YTD Growth")
        st.info("📌 Placeholder: Show growth per salesman/brand over time")

def sku_sales_prediction(df, container):
    with container:
        st.subheader("SKU Sales Prediction")
        st.info("📌 Placeholder: Predict SKU performance using regression/classification")

def top_bottom_salesmen(df, container):
    with container:
        st.subheader("Consistent Top/Bottom Salesmen")
        st.info("📌 Placeholder: Salesman ranking per month using delivered quantity/value")

def brand_sku_performance(df, container):
    with container:
        st.subheader("Brand & SKU Performance")
        st.info("📌 Placeholder: Aggregated metrics by brand & SKU (charts/tables)")

def design_brand_correlation(df, container):
    with container:
        st.subheader("Designation vs. Brand Correlation")
        st.info("📌 Placeholder: Heatmap of how designations interact with brands")

def comparative_analysis(df, container):
    with container:
        st.subheader("Comparative Analysis")
        st.info("📌 Placeholder: Comparison by branch, designation, period, etc.")

def recommendation_system(df, container):
    with container:
        st.subheader("Recommendation System")

        required_cols = {'salesman_code', 'sku_code', 'delivered_qty'}
        if not required_cols.issubset(df.columns):
            st.error(f"Missing columns: {required_cols}")
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

# ------------------------- Main App Layout ----------------------------------

def main():
    st.set_page_config(page_title="SalesMan Intelligence Dashboard", layout="wide")
    st.title("🧠 SalesMan Market Intelligence Dashboard")

    df = load_data()

    tabs = st.tabs([
        "Market Basket", "Forecasting", "MMM", "Consumer", "Growth",
        "Prediction", "Top/Bottom Salesmen", "Brand/SKU", "Designation ↔ Brand",
        "Comparative", "Recommendations"
    ])

    market_basket_analysis(df, tabs[0])
    sales_forecasting(df, tabs[1])
    market_mix_modeling(df, tabs[2])
    consumer_analysis(df, tabs[3])
    growth_calculations(df, tabs[4])
    sku_sales_prediction(df, tabs[5])
    top_bottom_salesmen(df, tabs[6])
    brand_sku_performance(df, tabs[7])
    design_brand_correlation(df, tabs[8])
    comparative_analysis(df, tabs[9])
    recommendation_system(df, tabs[10])

if __name__ == "__main__":
    main()





# import streamlit as st
# import pandas as pd
# import numpy as np
# from mlxtend.frequent_patterns import apriori, association_rules
# from sklearn.decomposition import TruncatedSVD

# st.title("SalesMan Market")

# # Load data from GitHub
# github_csv_url = "https://raw.githubusercontent.com/dgorgo/sales_app/main/sales_df.csv"

# try:
#     sales_data = pd.read_csv(github_csv_url)
# except Exception as e:
#     st.error(f"❌ Error reading the CSV file from GitHub: {e}")
#     st.stop()

# # Clean column names
# sales_data.columns = sales_data.columns.str.strip().str.lower()

# # Parse date column if it exists
# if 'delivered_date' in sales_data.columns:
#     sales_data['delivered_date'] = pd.to_datetime(sales_data['delivered_date'], errors='coerce')

# # Drop missing values
# sales_data.dropna(inplace=True)

# # ─── Market Basket Analysis ─────────────────────────────
# st.subheader("Market Basket Analysis")

# mba_cols = {'order_id', 'sku_code', 'delivered qty'}

# if mba_cols.issubset(sales_data.columns):
#     basket = (
#         sales_data
#         .groupby(['order_id', 'sku_code'])['delivered qty']
#         .sum()
#         .unstack(fill_value=0)
#         .applymap(lambda x: 1 if x > 0 else 0)
#     )

#     frequent_itemsets = apriori(basket, min_support=0.02, use_colnames=True)

#     st.write("**Frequent Itemsets:**")
#     st.dataframe(frequent_itemsets.sort_values('support', ascending=False))

# else:
#     st.error(f"Missing columns for Market Basket Analysis; need {mba_cols}")

# # ─── Recommendation System ──────────────────────────────
# st.subheader("Recommendation System")

# rs_cols = {'salesman_code', 'sku_code', 'delivered qty'}

# if rs_cols.issubset(sales_data.columns):
#     sales_data['salesman_code'] = pd.to_numeric(sales_data['salesman_code'], errors='coerce')
#     sales_data.dropna(subset=['salesman_code'], inplace=True)
#     sales_data['salesman_code'] = sales_data['salesman_code'].astype(int)

#     user_item = sales_data.pivot_table(
#         index='salesman_code',
#         columns='sku_code',
#         values='delivered qty',
#         fill_value=0
#     )

#     svd = TruncatedSVD(n_components=20, random_state=42)
#     user_factors = svd.fit_transform(user_item.values)
#     item_factors = svd.components_

#     preds = np.dot(user_factors, item_factors)
#     pred_df = pd.DataFrame(preds, index=user_item.index, columns=user_item.columns)

#     salesmen = user_item.index.tolist()
#     salesman = st.selectbox("Select Salesman Code:", options=salesmen)

#     if st.button("Get Recommendations"):
#         actual = user_item.loc[salesman]
#         scores = pred_df.loc[salesman].copy()
#         scores[actual > 0] = -np.inf

#         top10 = scores.nlargest(10).reset_index()
#         top10.columns = ['SKU_Code', 'Predicted Score']

#         st.write("**Top 10 Recommended Products:**")
#         st.dataframe(top10)

# else:
#     st.error(f"Missing columns for Recommendation System; need {rs_cols}")
