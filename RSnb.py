import streamlit as st
import pandas as pd
import numpy as np
from mlxtend.frequent_patterns import apriori, association_rules
from sklearn.decomposition import TruncatedSVD

st.title("SalesMan Market")

# Load data from GitHub
github_csv_url = "https://raw.githubusercontent.com/dgorgo/sales_app/main/sales_df.csv"

try:
    sales_data = pd.read_csv(github_csv_url)
except Exception as e:
    st.error(f"❌ Error reading the CSV file from GitHub: {e}")
    st.stop()

# Clean column names
sales_data.columns = sales_data.columns.str.strip().str.lower()

# Parse date column if it exists
if 'delivered_date' in sales_data.columns:
    sales_data['delivered_date'] = pd.to_datetime(sales_data['delivered_date'], errors='coerce')

# Drop missing values
sales_data.dropna(inplace=True)

# ─── Market Basket Analysis ─────────────────────────────
st.subheader("Market Basket Analysis")

mba_cols = {'order_id', 'sku_code', 'delivered qty'}

if mba_cols.issubset(sales_data.columns):
    basket = (
        sales_data
        .groupby(['order_id', 'sku_code'])['delivered qty']
        .sum()
        .unstack(fill_value=0)
        .applymap(lambda x: 1 if x > 0 else 0)
    )

    frequent_itemsets = apriori(basket, min_support=0.02, use_colnames=True)

    st.write("**Frequent Itemsets:**")
    st.dataframe(frequent_itemsets.sort_values('support', ascending=False))

else:
    st.error(f"Missing columns for Market Basket Analysis; need {mba_cols}")

# ─── Recommendation System ──────────────────────────────
st.subheader("Recommendation System")

rs_cols = {'salesman_code', 'sku_code', 'delivered qty'}

if rs_cols.issubset(sales_data.columns):
    sales_data['salesman_code'] = pd.to_numeric(sales_data['salesman_code'], errors='coerce')
    sales_data.dropna(subset=['salesman_code'], inplace=True)
    sales_data['salesman_code'] = sales_data['salesman_code'].astype(int)

    user_item = sales_data.pivot_table(
        index='salesman_code',
        columns='sku_code',
        values='delivered qty',
        fill_value=0
    )

    svd = TruncatedSVD(n_components=20, random_state=42)
    user_factors = svd.fit_transform(user_item.values)
    item_factors = svd.components_

    preds = np.dot(user_factors, item_factors)
    pred_df = pd.DataFrame(preds, index=user_item.index, columns=user_item.columns)

    salesmen = user_item.index.tolist()
    salesman = st.selectbox("Select Salesman Code:", options=salesmen)

    if st.button("Get Recommendations"):
        actual = user_item.loc[salesman]
        scores = pred_df.loc[salesman].copy()
        scores[actual > 0] = -np.inf

        top10 = scores.nlargest(10).reset_index()
        top10.columns = ['SKU_Code', 'Predicted Score']

        st.write("**Top 10 Recommended Products:**")
        st.dataframe(top10)

else:
    st.error(f"Missing columns for Recommendation System; need {rs_cols}")
