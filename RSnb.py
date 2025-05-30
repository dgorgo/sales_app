import streamlit as st
import pandas as pd
import numpy as np
from mlxtend.frequent_patterns import apriori, association_rules
from sklearn.decomposition import TruncatedSVD
from prophet import Prophet
import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt



# Load and preprocess
# --------------------------------------

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


# def load_data():
#     uploaded_file = st.file_uploader("📂 Upload your sales_data.csv", type=["csv"])
#     if uploaded_file is None:
#         st.info("Please upload a CSV file to get started.")
#         st.stop()
#     try:
#         df = pd.read_csv(uploaded_file)
#         df.columns = df.columns.str.lower().str.replace(" ", "_")
#         if 'delivered_date' in df.columns:
#             df['delivered_date'] = pd.to_datetime(df['delivered_date'], errors='coerce')
#         df.dropna(inplace=True)
#         return df
#     except Exception as e:
#         st.error(f"Error reading the CSV file: {e}")
#         st.stop()

# --------------------------------------
# Market Basket Analysis
# --------------------------------------
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

        # st.write("📌 Association Rules")
        # st.dataframe(rules.sort_values('lift', ascending=False))

        # st.subheader("📊 Support vs Confidence")
        # plt.figure(figsize=(8, 5))
        # sns.scatterplot(data=rules, x='support', y='confidence', hue='lift', size='lift', palette='viridis')
        # st.pyplot(plt)

# --------------------------------------
# Recommendation System
# --------------------------------------
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

# --------------------------------------
# Forecasting Module
# --------------------------------------
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

# --------------------------------------
# Placeholder Modules
# --------------------------------------
def consumer_analysis(df, container):
    with container:
        st.subheader("👥 Consumer Behavior Analysis")
        st.info("📌 Placeholder: Analyze repeat customers, RFM, segmentation")

def growth_calculations(df, container):
    with container:
        st.subheader("📊 MTD, QTD, YTD Growth")
        st.info("📌 Placeholder: Show growth per salesman/brand over time")

def sku_sales_prediction(df, container):
    with container:
        st.subheader("🔮 SKU Sales Prediction")
        st.info("📌 Placeholder: Predict SKU performance using regression/classification")

def top_bottom_salesmen(df, container):
    with container:
        st.subheader("🏅 Consistent Top/Bottom Salesmen")
        st.info("📌 Placeholder: Salesman ranking per month using delivered quantity/value")

def brand_sku_performance(df, container):
    with container:
        st.subheader("📊 Brand & SKU Performance")
        brand_perf = df.groupby('brand')['redistribution_value'].sum().reset_index()
        sku_perf = df.groupby('sku_code')['redistribution_value'].sum().reset_index()

        st.plotly_chart(px.bar(brand_perf, x='brand', y='redistribution_value', title="Brand Performance"))
        st.plotly_chart(px.bar(sku_perf, x='sku_code', y='redistribution_value', title="SKU Performance"))

def design_brand_correlation(df, container):
    with container:
        st.subheader("🧭 Designation vs. Brand Correlation")
        st.info("📌 Placeholder: Heatmap of how designations interact with brands")

def comparative_analysis(df, container):
    with container:
        st.subheader("⚖️ Comparative Analysis")
        st.info("📌 Placeholder: Comparison by branch, designation, period, etc.")

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

# --------------------------------------
# Main App Layout
# --------------------------------------
def main():
    st.set_page_config(page_title="Sales Intelligence Dashboard", layout="wide")
    st.title("🧠 Market Cart Dashboard")

    df = load_data()

    tabs = st.tabs([
        "📦 Market Basket", "🎯 Recommendations", "📈 Forecasting",
        "📊 Brand Performance", "Salesman Performance"
    ])

    market_basket_analysis(df, tabs[0])
    recommendation_system(df, tabs[1])
    forecast_sales(df, tabs[2])
    brand_sku_performance(df, tabs[3])
    Salesman_performance(df, tabs[4])

if __name__ == "__main__":
    main()





# import streamlit as st
# import pandas as pd
# import numpy as np
# from mlxtend.frequent_patterns import apriori
# from sklearn.decomposition import TruncatedSVD
# from prophet import Prophet
# import plotly.express as px
# from datetime import datetime

# # ------------------------- Load & Preprocess Data ---------------------------
# def load_data():
#     url = "https://raw.githubusercontent.com/dgorgo/sales_app/main/sales_df.csv"
#     try:
#         df = pd.read_csv(url)
#         df.columns = df.columns.str.lower().str.replace(" ", "_")
#         df['delivered_date'] = pd.to_datetime(df['delivered_date'], errors='coerce')
#         df.dropna(inplace=True)
#     except Exception as e:
#         st.error(f"Error loading data: {e}")
#         st.stop()
#     return df

# # ------------------------- Feature Modules ----------------------------------

# def market_basket_analysis(df, container):
#     with container:
#         st.subheader("🧺 Market Basket Analysis")
#         if {'order_id', 'sku_code', 'delivered_qty'}.issubset(df.columns):
#             basket = (
#                 df.groupby(['order_id', 'sku_code'])['delivered_qty']
#                 .sum().unstack(fill_value=0)
#                 .applymap(lambda x: 1 if x > 0 else 0)
#             )
#             frequent_items = apriori(basket, min_support=0.02, use_colnames=True)
#             st.write("📌 Top Frequent Itemsets")
#             st.dataframe(frequent_items.sort_values('support', ascending=False))
#         else:
#             st.warning("Missing required columns for MBA: order_id, sku_code, delivered_qty")

# def recommendation_system(df, container):
#     with container:
#         st.subheader("🎯 SKU Recommendation System (SVD)")
#         required_cols = {'salesman_code', 'sku_code', 'delivered_qty'}
#         if not required_cols.issubset(df.columns):
#             st.error(f"Missing columns: {required_cols}")
#             return

#         df['salesman_code'] = pd.to_numeric(df['salesman_code'], errors='coerce')
#         df.dropna(subset=['salesman_code'], inplace=True)
#         df['salesman_code'] = df['salesman_code'].astype(int)

#         pivot = df.pivot_table(index='salesman_code', columns='sku_code',
#                                values='delivered_qty', aggfunc='sum', fill_value=0)

#         svd = TruncatedSVD(n_components=10, random_state=42)
#         user_factors = svd.fit_transform(pivot)
#         item_factors = svd.components_
#         preds = np.dot(user_factors, item_factors)

#         pred_df = pd.DataFrame(preds, index=pivot.index, columns=pivot.columns)

#         selected_user = st.selectbox("Select Salesman Code", pred_df.index)
#         if st.button("Recommend SKUs"):
#             actuals = pivot.loc[selected_user]
#             scores = pred_df.loc[selected_user].copy()
#             scores[actuals > 0] = -np.inf  # Avoid recommending already bought
#             top10 = scores.nlargest(10).reset_index()
#             top10.columns = ['SKU_Code', 'Recommendation Score']
#             st.write("🔍 Top Recommended SKUs:")
#             st.dataframe(top10)

# def forecast_sales(df, container):
#     with container:
#         st.subheader("📈 Prophet Forecasting")
#         sku = st.selectbox("Select SKU Code for Forecasting", df['sku_code'].unique())
#         sku_df = df[df['sku_code'] == sku].groupby('delivered_date')['delivered_qty'].sum().reset_index()
#         sku_df = sku_df.rename(columns={'delivered_date': 'ds', 'delivered_qty': 'y'})

#         model = Prophet()
#         model.fit(sku_df)

#         future = model.make_future_dataframe(periods=30)
#         forecast = model.predict(future)
#         fig = px.line(forecast, x='ds', y='yhat', title=f"Forecast for SKU {sku}")
#         st.plotly_chart(fig)

# def market_mix_modeling(df, container):
#     with container:
#         st.subheader("📦 Marketing Mix Modeling (MMM) - Dummy Data")
#         dummy = pd.DataFrame({
#             'TV_Spend': [20, 25, 30, 35, 40],
#             'Social_Spend': [5, 10, 15, 20, 25],
#             'Search_Spend': [3, 4, 6, 8, 10],
#             'Sales': [200, 240, 300, 370, 410]
#         })
#         st.dataframe(dummy)
#         st.caption("📌 Add regression modeling in next iteration.")

# def consumer_analysis(df, container):
#     with container:
#         st.subheader("👥 Consumer Behavior Analysis")
#         st.info("📌 Placeholder: Analyze repeat customers, RFM, segmentation")

# def growth_calculations(df, container):
#     with container:
#         st.subheader("📊 MTD, QTD, YTD Growth")
#         st.info("📌 Placeholder: Show growth per salesman/brand over time")

# def sku_sales_prediction(df, container):
#     with container:
#         st.subheader("🔮 SKU Sales Prediction")
#         st.info("📌 Placeholder: Predict SKU performance using regression/classification")

# def top_bottom_salesmen(df, container):
#     with container:
#         st.subheader("🏅 Consistent Top/Bottom Salesmen")
#         st.info("📌 Placeholder: Salesman ranking per month using delivered quantity/value")

# def brand_sku_performance(df, container):
#     with container:
#         st.subheader("📊 Brand & SKU Performance")
#         brand_perf = df.groupby('brand')['redistribution_value'].sum().reset_index()
#         sku_perf = df.groupby('sku_code')['redistribution_value'].sum().reset_index()

#         st.plotly_chart(px.bar(brand_perf, x='brand', y='redistribution_value', title="Brand Performance"))
#         st.plotly_chart(px.bar(sku_perf, x='sku_code', y='redistribution_value', title="SKU Performance"))

# def design_brand_correlation(df, container):
#     with container:
#         st.subheader("🧭 Designation vs. Brand Correlation")
#         st.info("📌 Placeholder: Heatmap of how designations interact with brands")

# def comparative_analysis(df, container):
#     with container:
#         st.subheader("⚖️ Comparative Analysis")
#         st.info("📌 Placeholder: Comparison by branch, designation, period, etc.")

# # ------------------------- Main App Layout ----------------------------------
# def main():
#     st.set_page_config(page_title="SalesMan Intelligence Dashboard", layout="wide")
#     st.title("🧠 SalesMan Market Intelligence Dashboard")

#     df = load_data()

#     tabs = st.tabs([
#         "Market Basket", "Forecasting", "MMM", "Consumer", "Growth",
#         "Prediction", "Top/Bottom Salesmen", "Brand/SKU", "Designation ↔ Brand",
#         "Comparative", "Recommendations"
#     ])

#     market_basket_analysis(df, tabs[0])
#     forecast_sales(df, tabs[1])
#     market_mix_modeling(df, tabs[2])
#     consumer_analysis(df, tabs[3])
#     growth_calculations(df, tabs[4])
#     sku_sales_prediction(df, tabs[5])
#     top_bottom_salesmen(df, tabs[6])
#     brand_sku_performance(df, tabs[7])
#     design_brand_correlation(df, tabs[8])
#     comparative_analysis(df, tabs[9])
#     recommendation_system(df, tabs[10])

# if __name__ == "__main__":
#     main()





# # import streamlit as st
# # import pandas as pd
# # import numpy as np
# # from mlxtend.frequent_patterns import apriori
# # from sklearn.decomposition import TruncatedSVD
# # import matplotlib.pyplot as plt
# # import seaborn as sns


# # from prophet import Prophet
# # import plotly.express as px
# # from datetime import datetime


# # # ------------------------- Load & Preprocess Data ---------------------------

# # def load_data():
# #     github_url = "https://raw.githubusercontent.com/dgorgo/sales_app/main/sales_df.csv"
# #     try:
# #         df = pd.read_csv(github_url)
# #     except Exception as e:
# #         st.error(f"Error loading data: {e}")
# #         st.stop()

# #     df.columns = df.columns.str.lower().str.replace(" ", "_")
# #     df['delivered_date'] = pd.to_datetime(df['delivered_date'], dayfirst=True, errors='coerce')
# #     df.dropna(inplace=True)
# #     return df

# # # ------------------------- Feature Modules ----------------------------------

# # def market_basket_analysis(df, container):
# #     with container:
# #         st.subheader("Market Basket Analysis")

# #         if {'order_id', 'sku_code', 'delivered_qty'}.issubset(df.columns):
# #             basket = (
# #                 df.groupby(['order_id', 'sku_code'])['delivered_qty']
# #                 .sum()
# #                 .unstack(fill_value=0)
# #                 .applymap(lambda x: 1 if x > 0 else 0)
# #             )
# #             frequent_items = apriori(basket, min_support=0.02, use_colnames=True)
# #             st.write("Top Frequent Itemsets")
# #             st.dataframe(frequent_items.sort_values('support', ascending=False))
# #         else:
# #             st.warning("Missing required columns for MBA")

# # def sales_forecasting(df, container):
# #     with container:
# #         st.subheader("Sales Forecasting")
# #         st.info("📌 Placeholder: Forecast SKU sales using Prophet/ARIMA")

# # def market_mix_modeling(df, container):
# #     with container:
# #         st.subheader("Market Mix Modeling")
# #         st.info("📌 Placeholder: Requires marketing spend, pricing, campaign data")

# # def consumer_analysis(df, container):
# #     with container:
# #         st.subheader("Consumer Behavior Analysis")
# #         st.info("📌 Placeholder: Analyze repeat customers, RFM, segmentation")

# # def growth_calculations(df, container):
# #     with container:
# #         st.subheader("MTD, QTD, YTD Growth")
# #         st.info("📌 Placeholder: Show growth per salesman/brand over time")

# # def sku_sales_prediction(df, container):
# #     with container:
# #         st.subheader("SKU Sales Prediction")
# #         st.info("📌 Placeholder: Predict SKU performance using regression/classification")

# # def top_bottom_salesmen(df, container):
# #     with container:
# #         st.subheader("Consistent Top/Bottom Salesmen")
# #         st.info("📌 Placeholder: Salesman ranking per month using delivered quantity/value")

# # def brand_sku_performance(df, container):
# #     with container:
# #         st.subheader("Brand & SKU Performance")
# #         st.info("📌 Placeholder: Aggregated metrics by brand & SKU (charts/tables)")

# # def design_brand_correlation(df, container):
# #     with container:
# #         st.subheader("Designation vs. Brand Correlation")
# #         st.info("📌 Placeholder: Heatmap of how designations interact with brands")

# # def comparative_analysis(df, container):
# #     with container:
# #         st.subheader("Comparative Analysis")
# #         st.info("📌 Placeholder: Comparison by branch, designation, period, etc.")

# # def recommendation_system(df, container):
# #     with container:
# #         st.subheader("Recommendation System")

# #         required_cols = {'salesman_code', 'sku_code', 'delivered_qty'}
# #         if not required_cols.issubset(df.columns):
# #             st.error(f"Missing columns: {required_cols}")
# #             return

# #         df['salesman_code'] = pd.to_numeric(df['salesman_code'], errors='coerce')
# #         df.dropna(subset=['salesman_code'], inplace=True)
# #         df['salesman_code'] = df['salesman_code'].astype(int)

# #         pivot = df.pivot_table(index='salesman_code', columns='sku_code',
# #                                values='delivered_qty', aggfunc='sum', fill_value=0)

# #         svd = TruncatedSVD(n_components=10, random_state=42)
# #         user_factors = svd.fit_transform(pivot)
# #         item_factors = svd.components_
# #         preds = np.dot(user_factors, item_factors)

# #         pred_df = pd.DataFrame(preds, index=pivot.index, columns=pivot.columns)

# #         selected_user = st.selectbox("Select Salesman Code", pred_df.index)
# #         if st.button("Recommend SKUs"):
# #             actuals = pivot.loc[selected_user]
# #             scores = pred_df.loc[selected_user].copy()
# #             scores[actuals > 0] = -np.inf
# #             top10 = scores.nlargest(10).reset_index()
# #             top10.columns = ['SKU_Code', 'Recommendation Score']
# #             st.write("🔍 Top Recommended SKUs:")
# #             st.dataframe(top10)



# # # ─── Load and Preprocess Data ─────────────────────────────
# # def load_data():
# #     df = pd.read_csv("https://raw.githubusercontent.com/dgorgo/sales_app/main/sales_df.csv")
# #     df.columns = df.columns.str.strip().str.lower().str.replace(' ', '_')
# #     df['delivered_date'] = pd.to_datetime(df['delivered_date'], errors='coerce')
# #     df.dropna(subset=['delivered_date'], inplace=True)
# #     return df

# # # ─── Forecasting ──────────────────────────────────────────
# # def forecast_sales(df):
# #     st.subheader("📈 Prophet Forecasting")
# #     sku = st.selectbox("Select SKU Code for Forecasting", df['sku_code'].unique())
# #     sku_df = df[df['sku_code'] == sku].groupby('delivered_date')['delivered_qty'].sum().reset_index()
# #     sku_df = sku_df.rename(columns={'delivered_date': 'ds', 'delivered_qty': 'y'})

# #     model = Prophet()
# #     model.fit(sku_df)

# #     future = model.make_future_dataframe(periods=30)
# #     forecast = model.predict(future)
# #     fig = px.line(forecast, x='ds', y='yhat', title=f"Forecast for SKU {sku}")
# #     st.plotly_chart(fig)

# # # ─── Brand and SKU Performance ────────────────────────────
# # def performance_charts(df):
# #     st.subheader("📊 Brand and SKU Performance")
# #     brand_perf = df.groupby('brand')['redistribution_value'].sum().reset_index()
# #     sku_perf = df.groupby('sku_code')['redistribution_value'].sum().reset_index()

# #     st.plotly_chart(px.bar(brand_perf, x='brand', y='redistribution_value', title="Brand Performance"))
# #     st.plotly_chart(px.bar(sku_perf, x='sku_code', y='redistribution_value', title="SKU Performance"))

# # # ─── Salesperson Specific Views ───────────────────────────
# # def salesperson_views(df):
# #     st.subheader("🧍‍♂️ Salesperson Performance")
# #     person = st.selectbox("Select Salesperson", df['salesman_name'].unique())
# #     person_df = df[df['salesman_name'] == person]

# #     monthly_perf = person_df.groupby(person_df['delivered_date'].dt.to_period('M'))['redistribution_value'].sum().reset_index()
# #     monthly_perf['delivered_date'] = monthly_perf['delivered_date'].astype(str)

# #     st.plotly_chart(px.line(monthly_perf, x='delivered_date', y='redistribution_value', title=f"Monthly Performance for {person}"))
# #     st.write("Recent Sales Transactions")
# #     st.dataframe(person_df[['delivered_date', 'sku_code', 'delivered_qty', 'redistribution_value']].sort_values(by='delivered_date', ascending=False))

# # # ─── Dummy Marketing Mix Modeling ─────────────────────────
# # def mmm_section():
# #     st.subheader("📦 Marketing Mix Modeling (MMM) - Dummy Data")
# #     dummy = pd.DataFrame({
# #         'TV_Spend': [20, 25, 30, 35, 40],
# #         'Social_Spend': [5, 10, 15, 20, 25],
# #         'Search_Spend': [3, 4, 6, 8, 10],
# #         'Sales': [200, 240, 300, 370, 410]
# #     })
# #     st.dataframe(dummy)
# #     st.caption("Add regression modeling in next iteration.")

# # # ─── Main App ─────────────────────────────────────────────
# # def main():
# #     st.title("📊 Sales Analytics Dashboard")
# #     df = load_data()

# #     tab1, tab2, tab3, tab4 = st.tabs(["Forecasting", "Brand/SKU", "Salesperson", "MMM"])

# #     with tab1:
# #         forecast_sales(df)
# #     with tab2:
# #         performance_charts(df)
# #     with tab3:
# #         salesperson_views(df)
# #     with tab4:
# #         mmm_section()

# # if __name__ == '__main__':
# #     main()





# # # ------------------------- Main App Layout ----------------------------------

# # def main():
# #     st.set_page_config(page_title="SalesMan Intelligence Dashboard", layout="wide")
# #     st.title("🧠 SalesMan Market Intelligence Dashboard")

# #     df = load_data()

# #     tabs = st.tabs([
# #         "Market Basket", "Forecasting", "MMM", "Consumer", "Growth",
# #         "Prediction", "Top/Bottom Salesmen", "Brand/SKU", "Designation ↔ Brand",
# #         "Comparative", "Recommendations"
# #     ])

# #     market_basket_analysis(df, tabs[0])
# #     sales_forecasting(df, tabs[1])
# #     market_mix_modeling(df, tabs[2])
# #     consumer_analysis(df, tabs[3])
# #     growth_calculations(df, tabs[4])
# #     sku_sales_prediction(df, tabs[5])
# #     top_bottom_salesmen(df, tabs[6])
# #     brand_sku_performance(df, tabs[7])
# #     design_brand_correlation(df, tabs[8])
# #     comparative_analysis(df, tabs[9])
# #     recommendation_system(df, tabs[10])

# # if __name__ == "__main__":
# #     main()






# import streamlit as st
# import pandas as pd
# import numpy as np
# from mlxtend.frequent_patterns import apriori, association_rules
# from sklearn.decomposition import TruncatedSVD
# from prophet import Prophet
# import plotly.express as px
# import seaborn as sns
# import matplotlib.pyplot as plt



# # Load and preprocess
# # --------------------------------------
# def load_data():
#     uploaded_file = st.file_uploader("📂 Upload your sales_data.csv", type=["csv"])
#     if uploaded_file is None:
#         st.info("Please upload a CSV file to get started.")
#         st.stop()
#     try:
#         df = pd.read_csv(uploaded_file)
#         df.columns = df.columns.str.lower().str.replace(" ", "_")
#         if 'delivered_date' in df.columns:
#             df['delivered_date'] = pd.to_datetime(df['delivered_date'], errors='coerce')
#         df.dropna(inplace=True)
#         return df
#     except Exception as e:
#         st.error(f"Error reading the CSV file: {e}")
#         st.stop()

# # --------------------------------------
# # Market Basket Analysis
# # --------------------------------------
# def market_basket_analysis(df, container):
#     with container:
#         st.subheader("🧺 Market Basket Analysis")
#         required = {'order_id', 'sku_code', 'delivered_qty'}
#         if not required.issubset(df.columns):
#             st.error(f"Missing required columns: {required}")
#             return
#         basket = (
#             df.groupby(['order_id', 'sku_code'])['delivered_qty']
#             .sum().unstack(fill_value=0)
#             .applymap(lambda x: 1 if x > 0 else 0)
#         )
#         frequent_items = apriori(basket, min_support=0.02, use_colnames=True)
#         rules = association_rules(frequent_items, metric="lift", min_threshold=1.0)

#         st.write("📌 Frequent Itemsets")
#         st.dataframe(frequent_items.sort_values('support', ascending=False))

#         st.write("📌 Association Rules")
#         st.dataframe(rules.sort_values('lift', ascending=False))

#         st.subheader("📊 Support vs Confidence")
#         plt.figure(figsize=(8, 5))
#         sns.scatterplot(data=rules, x='support', y='confidence', hue='lift', size='lift', palette='viridis')
#         st.pyplot(plt)

# # --------------------------------------
# # Recommendation System
# # --------------------------------------
# def recommendation_system(df, container):
#     with container:
#         st.subheader("🎯 SKU Recommendation System (SVD)")
#         required = {'salesman_code', 'sku_code', 'delivered_qty'}
#         if not required.issubset(df.columns):
#             st.error(f"Missing columns: {required}")
#             return
#         df['salesman_code'] = pd.to_numeric(df['salesman_code'], errors='coerce')
#         df.dropna(subset=['salesman_code'], inplace=True)
#         df['salesman_code'] = df['salesman_code'].astype(int)

#         pivot = df.pivot_table(index='salesman_code', columns='sku_code',
#                                values='delivered_qty', aggfunc='sum', fill_value=0)

#         svd = TruncatedSVD(n_components=10, random_state=42)
#         user_factors = svd.fit_transform(pivot)
#         item_factors = svd.components_
#         preds = np.dot(user_factors, item_factors)

#         pred_df = pd.DataFrame(preds, index=pivot.index, columns=pivot.columns)
#         selected_user = st.selectbox("Select Salesman Code", pred_df.index)
#         if st.button("Recommend SKUs"):
#             actuals = pivot.loc[selected_user]
#             scores = pred_df.loc[selected_user].copy()
#             scores[actuals > 0] = -np.inf
#             top10 = scores.nlargest(10).reset_index()
#             top10.columns = ['SKU_Code', 'Recommendation Score']
#             st.write("🔍 Top Recommended SKUs:")
#             st.dataframe(top10)

# # --------------------------------------
# # Forecasting Module
# # --------------------------------------
# def forecast_sales(df, container):
#     with container:
#         st.subheader("📈 Forecast SKU Sales (Prophet)")
#         if 'sku_code' not in df.columns or 'delivered_date' not in df.columns:
#             st.warning("Missing required columns for forecasting.")
#             return
#         sku = st.selectbox("Select SKU Code", df['sku_code'].unique())
#         sku_df = df[df['sku_code'] == sku].groupby('delivered_date')['delivered_qty'].sum().reset_index()
#         sku_df = sku_df.rename(columns={'delivered_date': 'ds', 'delivered_qty': 'y'})

#         model = Prophet()
#         model.fit(sku_df)

#         future = model.make_future_dataframe(periods=30)
#         forecast = model.predict(future)

#         fig = px.line(forecast, x='ds', y='yhat', title=f"📦 Forecast for SKU: {sku}")
#         st.plotly_chart(fig)

# # --------------------------------------
# # Placeholder Modules
# # --------------------------------------
# def market_mix_modeling(df, container):
#     with container:
#         st.subheader("📦 Marketing Mix Modeling")
#         st.write("📌 Dummy data for modeling")
#         dummy = pd.DataFrame({
#             'TV_Spend': [20, 25, 30, 35, 40],
#             'Social_Spend': [5, 10, 15, 20, 25],
#             'Search_Spend': [3, 4, 6, 8, 10],
#             'Sales': [200, 240, 300, 370, 410]
#         })
#         st.dataframe(dummy)

# def brand_sku_performance(df, container):
#     with container:
#         st.subheader("📊 Brand & SKU Performance")
#         if 'brand' in df.columns and 'redistribution_value' in df.columns:
#             brand_perf = df.groupby('brand')['redistribution_value'].sum().reset_index()
#             sku_perf = df.groupby('sku_code')['redistribution_value'].sum().reset_index()
#             st.plotly_chart(px.bar(brand_perf, x='brand', y='redistribution_value', title="Brand Performance"))
#             st.plotly_chart(px.bar(sku_perf, x='sku_code', y='redistribution_value', title="SKU Performance"))
#         else:
#             st.warning("Missing 'brand' or 'redistribution_value' columns.")

# # --------------------------------------
# # Main App Layout
# # --------------------------------------
# def main():
#     st.set_page_config(page_title="Sales Intelligence Dashboard", layout="wide")
#     st.title("🧠 Unified Sales Intelligence Dashboard")

#     df = load_data()

#     tabs = st.tabs([
#         "📦 Market Basket", "🎯 Recommendations", "📈 Forecasting",
#         "📊 Brand Performance", "📺 MMM (Dummy)"
#     ])

#     market_basket_analysis(df, tabs[0])
#     recommendation_system(df, tabs[1])
#     forecast_sales(df, tabs[2])
#     brand_sku_performance(df, tabs[3])
#     market_mix_modeling(df, tabs[4])

# if __name__ == "__main__":
#     main()





# # import streamlit as st
# # import pandas as pd
# # import numpy as np
# # from mlxtend.frequent_patterns import apriori
# # from sklearn.decomposition import TruncatedSVD
# # from prophet import Prophet
# # import plotly.express as px
# # from datetime import datetime

# # # ------------------------- Load & Preprocess Data ---------------------------
# # def load_data():
# #     url = "https://raw.githubusercontent.com/dgorgo/sales_app/main/sales_df.csv"
# #     try:
# #         df = pd.read_csv(url)
# #         df.columns = df.columns.str.lower().str.replace(" ", "_")
# #         df['delivered_date'] = pd.to_datetime(df['delivered_date'], errors='coerce')
# #         df.dropna(inplace=True)
# #     except Exception as e:
# #         st.error(f"Error loading data: {e}")
# #         st.stop()
# #     return df

# # # ------------------------- Feature Modules ----------------------------------

# # def market_basket_analysis(df, container):
# #     with container:
# #         st.subheader("🧺 Market Basket Analysis")
# #         if {'order_id', 'sku_code', 'delivered_qty'}.issubset(df.columns):
# #             basket = (
# #                 df.groupby(['order_id', 'sku_code'])['delivered_qty']
# #                 .sum().unstack(fill_value=0)
# #                 .applymap(lambda x: 1 if x > 0 else 0)
# #             )
# #             frequent_items = apriori(basket, min_support=0.02, use_colnames=True)
# #             st.write("📌 Top Frequent Itemsets")
# #             st.dataframe(frequent_items.sort_values('support', ascending=False))
# #         else:
# #             st.warning("Missing required columns for MBA: order_id, sku_code, delivered_qty")

# # def recommendation_system(df, container):
# #     with container:
# #         st.subheader("🎯 SKU Recommendation System (SVD)")
# #         required_cols = {'salesman_code', 'sku_code', 'delivered_qty'}
# #         if not required_cols.issubset(df.columns):
# #             st.error(f"Missing columns: {required_cols}")
# #             return

# #         df['salesman_code'] = pd.to_numeric(df['salesman_code'], errors='coerce')
# #         df.dropna(subset=['salesman_code'], inplace=True)
# #         df['salesman_code'] = df['salesman_code'].astype(int)

# #         pivot = df.pivot_table(index='salesman_code', columns='sku_code',
# #                                values='delivered_qty', aggfunc='sum', fill_value=0)

# #         svd = TruncatedSVD(n_components=10, random_state=42)
# #         user_factors = svd.fit_transform(pivot)
# #         item_factors = svd.components_
# #         preds = np.dot(user_factors, item_factors)

# #         pred_df = pd.DataFrame(preds, index=pivot.index, columns=pivot.columns)

# #         selected_user = st.selectbox("Select Salesman Code", pred_df.index)
# #         if st.button("Recommend SKUs"):
# #             actuals = pivot.loc[selected_user]
# #             scores = pred_df.loc[selected_user].copy()
# #             scores[actuals > 0] = -np.inf  # Avoid recommending already bought
# #             top10 = scores.nlargest(10).reset_index()
# #             top10.columns = ['SKU_Code', 'Recommendation Score']
# #             st.write("🔍 Top Recommended SKUs:")
# #             st.dataframe(top10)

# # def forecast_sales(df, container):
# #     with container:
# #         st.subheader("📈 Prophet Forecasting")
# #         sku = st.selectbox("Select SKU Code for Forecasting", df['sku_code'].unique())
# #         sku_df = df[df['sku_code'] == sku].groupby('delivered_date')['delivered_qty'].sum().reset_index()
# #         sku_df = sku_df.rename(columns={'delivered_date': 'ds', 'delivered_qty': 'y'})

# #         model = Prophet()
# #         model.fit(sku_df)

# #         future = model.make_future_dataframe(periods=30)
# #         forecast = model.predict(future)
# #         fig = px.line(forecast, x='ds', y='yhat', title=f"Forecast for SKU {sku}")
# #         st.plotly_chart(fig)

# # def market_mix_modeling(df, container):
# #     with container:
# #         st.subheader("📦 Marketing Mix Modeling (MMM) - Dummy Data")
# #         dummy = pd.DataFrame({
# #             'TV_Spend': [20, 25, 30, 35, 40],
# #             'Social_Spend': [5, 10, 15, 20, 25],
# #             'Search_Spend': [3, 4, 6, 8, 10],
# #             'Sales': [200, 240, 300, 370, 410]
# #         })
# #         st.dataframe(dummy)
# #         st.caption("📌 Add regression modeling in next iteration.")

# # def consumer_analysis(df, container):
# #     with container:
# #         st.subheader("👥 Consumer Behavior Analysis")
# #         st.info("📌 Placeholder: Analyze repeat customers, RFM, segmentation")

# # def growth_calculations(df, container):
# #     with container:
# #         st.subheader("📊 MTD, QTD, YTD Growth")
# #         st.info("📌 Placeholder: Show growth per salesman/brand over time")

# # def sku_sales_prediction(df, container):
# #     with container:
# #         st.subheader("🔮 SKU Sales Prediction")
# #         st.info("📌 Placeholder: Predict SKU performance using regression/classification")

# # def top_bottom_salesmen(df, container):
# #     with container:
# #         st.subheader("🏅 Consistent Top/Bottom Salesmen")
# #         st.info("📌 Placeholder: Salesman ranking per month using delivered quantity/value")

# # def brand_sku_performance(df, container):
# #     with container:
# #         st.subheader("📊 Brand & SKU Performance")
# #         brand_perf = df.groupby('brand')['redistribution_value'].sum().reset_index()
# #         sku_perf = df.groupby('sku_code')['redistribution_value'].sum().reset_index()

# #         st.plotly_chart(px.bar(brand_perf, x='brand', y='redistribution_value', title="Brand Performance"))
# #         st.plotly_chart(px.bar(sku_perf, x='sku_code', y='redistribution_value', title="SKU Performance"))

# # def design_brand_correlation(df, container):
# #     with container:
# #         st.subheader("🧭 Designation vs. Brand Correlation")
# #         st.info("📌 Placeholder: Heatmap of how designations interact with brands")

# # def comparative_analysis(df, container):
# #     with container:
# #         st.subheader("⚖️ Comparative Analysis")
# #         st.info("📌 Placeholder: Comparison by branch, designation, period, etc.")

# # # ------------------------- Main App Layout ----------------------------------
# # def main():
# #     st.set_page_config(page_title="SalesMan Intelligence Dashboard", layout="wide")
# #     st.title("🧠 SalesMan Market Intelligence Dashboard")

# #     df = load_data()

# #     tabs = st.tabs([
# #         "Market Basket", "Forecasting", "MMM", "Consumer", "Growth",
# #         "Prediction", "Top/Bottom Salesmen", "Brand/SKU", "Designation ↔ Brand",
# #         "Comparative", "Recommendations"
# #     ])

# #     market_basket_analysis(df, tabs[0])
# #     forecast_sales(df, tabs[1])
# #     market_mix_modeling(df, tabs[2])
# #     consumer_analysis(df, tabs[3])
# #     growth_calculations(df, tabs[4])
# #     sku_sales_prediction(df, tabs[5])
# #     top_bottom_salesmen(df, tabs[6])
# #     brand_sku_performance(df, tabs[7])
# #     design_brand_correlation(df, tabs[8])
# #     comparative_analysis(df, tabs[9])
# #     recommendation_system(df, tabs[10])

# # if __name__ == "__main__":
# #     main()





# # # import streamlit as st
# # # import pandas as pd
# # # import numpy as np
# # # from mlxtend.frequent_patterns import apriori
# # # from sklearn.decomposition import TruncatedSVD
# # # import matplotlib.pyplot as plt
# # # import seaborn as sns


# # # from prophet import Prophet
# # # import plotly.express as px
# # # from datetime import datetime


# # # # ------------------------- Load & Preprocess Data ---------------------------

# # # def load_data():
# # #     github_url = "https://raw.githubusercontent.com/dgorgo/sales_app/main/sales_df.csv"
# # #     try:
# # #         df = pd.read_csv(github_url)
# # #     except Exception as e:
# # #         st.error(f"Error loading data: {e}")
# # #         st.stop()

# # #     df.columns = df.columns.str.lower().str.replace(" ", "_")
# # #     df['delivered_date'] = pd.to_datetime(df['delivered_date'], dayfirst=True, errors='coerce')
# # #     df.dropna(inplace=True)
# # #     return df

# # # # ------------------------- Feature Modules ----------------------------------

# # # def market_basket_analysis(df, container):
# # #     with container:
# # #         st.subheader("Market Basket Analysis")

# # #         if {'order_id', 'sku_code', 'delivered_qty'}.issubset(df.columns):
# # #             basket = (
# # #                 df.groupby(['order_id', 'sku_code'])['delivered_qty']
# # #                 .sum()
# # #                 .unstack(fill_value=0)
# # #                 .applymap(lambda x: 1 if x > 0 else 0)
# # #             )
# # #             frequent_items = apriori(basket, min_support=0.02, use_colnames=True)
# # #             st.write("Top Frequent Itemsets")
# # #             st.dataframe(frequent_items.sort_values('support', ascending=False))
# # #         else:
# # #             st.warning("Missing required columns for MBA")

# # # def sales_forecasting(df, container):
# # #     with container:
# # #         st.subheader("Sales Forecasting")
# # #         st.info("📌 Placeholder: Forecast SKU sales using Prophet/ARIMA")

# # # def market_mix_modeling(df, container):
# # #     with container:
# # #         st.subheader("Market Mix Modeling")
# # #         st.info("📌 Placeholder: Requires marketing spend, pricing, campaign data")

# # # def consumer_analysis(df, container):
# # #     with container:
# # #         st.subheader("Consumer Behavior Analysis")
# # #         st.info("📌 Placeholder: Analyze repeat customers, RFM, segmentation")

# # # def growth_calculations(df, container):
# # #     with container:
# # #         st.subheader("MTD, QTD, YTD Growth")
# # #         st.info("📌 Placeholder: Show growth per salesman/brand over time")

# # # def sku_sales_prediction(df, container):
# # #     with container:
# # #         st.subheader("SKU Sales Prediction")
# # #         st.info("📌 Placeholder: Predict SKU performance using regression/classification")

# # # def top_bottom_salesmen(df, container):
# # #     with container:
# # #         st.subheader("Consistent Top/Bottom Salesmen")
# # #         st.info("📌 Placeholder: Salesman ranking per month using delivered quantity/value")

# # # def brand_sku_performance(df, container):
# # #     with container:
# # #         st.subheader("Brand & SKU Performance")
# # #         st.info("📌 Placeholder: Aggregated metrics by brand & SKU (charts/tables)")

# # # def design_brand_correlation(df, container):
# # #     with container:
# # #         st.subheader("Designation vs. Brand Correlation")
# # #         st.info("📌 Placeholder: Heatmap of how designations interact with brands")

# # # def comparative_analysis(df, container):
# # #     with container:
# # #         st.subheader("Comparative Analysis")
# # #         st.info("📌 Placeholder: Comparison by branch, designation, period, etc.")

# # # def recommendation_system(df, container):
# # #     with container:
# # #         st.subheader("Recommendation System")

# # #         required_cols = {'salesman_code', 'sku_code', 'delivered_qty'}
# # #         if not required_cols.issubset(df.columns):
# # #             st.error(f"Missing columns: {required_cols}")
# # #             return

# # #         df['salesman_code'] = pd.to_numeric(df['salesman_code'], errors='coerce')
# # #         df.dropna(subset=['salesman_code'], inplace=True)
# # #         df['salesman_code'] = df['salesman_code'].astype(int)

# # #         pivot = df.pivot_table(index='salesman_code', columns='sku_code',
# # #                                values='delivered_qty', aggfunc='sum', fill_value=0)

# # #         svd = TruncatedSVD(n_components=10, random_state=42)
# # #         user_factors = svd.fit_transform(pivot)
# # #         item_factors = svd.components_
# # #         preds = np.dot(user_factors, item_factors)

# # #         pred_df = pd.DataFrame(preds, index=pivot.index, columns=pivot.columns)

# # #         selected_user = st.selectbox("Select Salesman Code", pred_df.index)
# # #         if st.button("Recommend SKUs"):
# # #             actuals = pivot.loc[selected_user]
# # #             scores = pred_df.loc[selected_user].copy()
# # #             scores[actuals > 0] = -np.inf
# # #             top10 = scores.nlargest(10).reset_index()
# # #             top10.columns = ['SKU_Code', 'Recommendation Score']
# # #             st.write("🔍 Top Recommended SKUs:")
# # #             st.dataframe(top10)



# # # # ─── Load and Preprocess Data ─────────────────────────────
# # # def load_data():
# # #     df = pd.read_csv("https://raw.githubusercontent.com/dgorgo/sales_app/main/sales_df.csv")
# # #     df.columns = df.columns.str.strip().str.lower().str.replace(' ', '_')
# # #     df['delivered_date'] = pd.to_datetime(df['delivered_date'], errors='coerce')
# # #     df.dropna(subset=['delivered_date'], inplace=True)
# # #     return df

# # # # ─── Forecasting ──────────────────────────────────────────
# # # def forecast_sales(df):
# # #     st.subheader("📈 Prophet Forecasting")
# # #     sku = st.selectbox("Select SKU Code for Forecasting", df['sku_code'].unique())
# # #     sku_df = df[df['sku_code'] == sku].groupby('delivered_date')['delivered_qty'].sum().reset_index()
# # #     sku_df = sku_df.rename(columns={'delivered_date': 'ds', 'delivered_qty': 'y'})

# # #     model = Prophet()
# # #     model.fit(sku_df)

# # #     future = model.make_future_dataframe(periods=30)
# # #     forecast = model.predict(future)
# # #     fig = px.line(forecast, x='ds', y='yhat', title=f"Forecast for SKU {sku}")
# # #     st.plotly_chart(fig)

# # # # ─── Brand and SKU Performance ────────────────────────────
# # # def performance_charts(df):
# # #     st.subheader("📊 Brand and SKU Performance")
# # #     brand_perf = df.groupby('brand')['redistribution_value'].sum().reset_index()
# # #     sku_perf = df.groupby('sku_code')['redistribution_value'].sum().reset_index()

# # #     st.plotly_chart(px.bar(brand_perf, x='brand', y='redistribution_value', title="Brand Performance"))
# # #     st.plotly_chart(px.bar(sku_perf, x='sku_code', y='redistribution_value', title="SKU Performance"))

# # # # ─── Salesperson Specific Views ───────────────────────────
# # # def salesperson_views(df):
# # #     st.subheader("🧍‍♂️ Salesperson Performance")
# # #     person = st.selectbox("Select Salesperson", df['salesman_name'].unique())
# # #     person_df = df[df['salesman_name'] == person]

# # #     monthly_perf = person_df.groupby(person_df['delivered_date'].dt.to_period('M'))['redistribution_value'].sum().reset_index()
# # #     monthly_perf['delivered_date'] = monthly_perf['delivered_date'].astype(str)

# # #     st.plotly_chart(px.line(monthly_perf, x='delivered_date', y='redistribution_value', title=f"Monthly Performance for {person}"))
# # #     st.write("Recent Sales Transactions")
# # #     st.dataframe(person_df[['delivered_date', 'sku_code', 'delivered_qty', 'redistribution_value']].sort_values(by='delivered_date', ascending=False))

# # # # ─── Dummy Marketing Mix Modeling ─────────────────────────
# # # def mmm_section():
# # #     st.subheader("📦 Marketing Mix Modeling (MMM) - Dummy Data")
# # #     dummy = pd.DataFrame({
# # #         'TV_Spend': [20, 25, 30, 35, 40],
# # #         'Social_Spend': [5, 10, 15, 20, 25],
# # #         'Search_Spend': [3, 4, 6, 8, 10],
# # #         'Sales': [200, 240, 300, 370, 410]
# # #     })
# # #     st.dataframe(dummy)
# # #     st.caption("Add regression modeling in next iteration.")

# # # # ─── Main App ─────────────────────────────────────────────
# # # def main():
# # #     st.title("📊 Sales Analytics Dashboard")
# # #     df = load_data()

# # #     tab1, tab2, tab3, tab4 = st.tabs(["Forecasting", "Brand/SKU", "Salesperson", "MMM"])

# # #     with tab1:
# # #         forecast_sales(df)
# # #     with tab2:
# # #         performance_charts(df)
# # #     with tab3:
# # #         salesperson_views(df)
# # #     with tab4:
# # #         mmm_section()

# # # if __name__ == '__main__':
# # #     main()





# # # # ------------------------- Main App Layout ----------------------------------

# # # def main():
# # #     st.set_page_config(page_title="SalesMan Intelligence Dashboard", layout="wide")
# # #     st.title("🧠 SalesMan Market Intelligence Dashboard")

# # #     df = load_data()

# # #     tabs = st.tabs([
# # #         "Market Basket", "Forecasting", "MMM", "Consumer", "Growth",
# # #         "Prediction", "Top/Bottom Salesmen", "Brand/SKU", "Designation ↔ Brand",
# # #         "Comparative", "Recommendations"
# # #     ])

# # #     market_basket_analysis(df, tabs[0])
# # #     sales_forecasting(df, tabs[1])
# # #     market_mix_modeling(df, tabs[2])
# # #     consumer_analysis(df, tabs[3])
# # #     growth_calculations(df, tabs[4])
# # #     sku_sales_prediction(df, tabs[5])
# # #     top_bottom_salesmen(df, tabs[6])
# # #     brand_sku_performance(df, tabs[7])
# # #     design_brand_correlation(df, tabs[8])
# # #     comparative_analysis(df, tabs[9])
# # #     recommendation_system(df, tabs[10])

# # # if __name__ == "__main__":
# # #     main()
