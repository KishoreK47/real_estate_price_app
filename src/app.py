# -------------------------------
# Real Estate Price Predictor App
# Final Production Version
# -------------------------------

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import shap
import matplotlib.pyplot as plt

st.set_page_config(
    layout="wide",
    page_title="🏠 Property Price Predictor",
    initial_sidebar_state="expanded"
)

# ----------------------------------------
# CACHED LOADING
# ----------------------------------------
@st.cache_resource
def load_pipeline(path="src/xgb_pipeline.joblib"):
    """Load any sklearn Pipeline (RF or XGB)."""
    return joblib.load(path)

@st.cache_data
def load_sample_df(path= r"C:\Users\kisho\Downloads\Real Estate Investment Advisor Predicting Property Profitability & Future Value- Project\Data\Raw\india_housing_prices.csv", nrows=2000):
    """Load sample for quantiles + SHAP reference."""
    try:
        return pd.read_csv(path, nrows=nrows)
    except:
        return None


# ----------------------------------------
# LOAD MODEL + SAMPLE DATA
# ----------------------------------------
st.sidebar.title("⚙️ Model Controls")
pipeline = load_pipeline("src/xgb_pipeline.joblib")      # change to rf_pipeline.joblib if needed
sample_df = load_sample_df(r"C:\Users\kisho\Downloads\Real Estate Investment Advisor Predicting Property Profitability & Future Value- Project\Data\Raw\india_housing_prices.csv")

st.title("🏠 Real Estate Price Predictor")
st.markdown("Predict property prices (in lakhs) and view SHAP explainability.")

mode = st.sidebar.radio("Choose Mode", ["Single Prediction", "Batch CSV Prediction"])


# ----------------------------------------
# PRICE CATEGORY HELPER
# ----------------------------------------
def price_to_category(price, q1, q2):
    if price <= q1:
        return "budget"
    elif price <= q2:
        return "mid"
    return "premium"


# =============================================================================
#                       BATCH CSV PREDICTION
# =============================================================================
if mode == "Batch CSV Prediction":

    st.header("📂 Batch Prediction")
    st.write("Upload a CSV file containing the same feature columns used during training.")

    uploaded = st.file_uploader("Upload CSV", type=["csv"])

    if uploaded:
        df = pd.read_csv(uploaded)
        st.write("### Preview:", df.head())

        if st.button("Run Batch Prediction"):
            preds = pipeline.predict(df)

            # Thresholds
            if sample_df is not None:
                q1 = np.percentile(sample_df["Price_in_Lakhs"], 33)
                q2 = np.percentile(sample_df["Price_in_Lakhs"], 66)
            else:
                q1, q2 = np.percentile(preds, [33, 66])

            cat = [price_to_category(p, q1, q2) for p in preds]

            out = df.copy()
            out["Predicted_Price_Lakhs"] = preds
            out["Predicted_Category"] = cat

            st.success("Prediction completed 🎉")
            st.write(out.head())

            st.download_button(
                label="Download Predictions",
                data=out.to_csv(index=False).encode("utf-8"),
                file_name="predictions.csv",
                mime="text/csv"
            )


# =============================================================================
#                       SINGLE PREDICTION MODE
# =============================================================================
else:
    st.header("🔍 Single Prediction")

    with st.form("predict_form"):
        col1, col2, col3 = st.columns(3)

        # ------------------ COLUMN 1 ------------------
        with col1:
            Size_in_SqFt = st.number_input("Size (SqFt)", 100, 10000, 1200)
            BHK = st.selectbox("BHK", [1,2,3,4,5], index=2)
            Property_Type = st.selectbox("Property Type", ["apartment", "independent house", "villa"])
            State = st.text_input("State", sample_df["State"].iloc[0] if sample_df is not None else "tamil nadu")
            City = st.text_input("City", sample_df["City"].iloc[0] if sample_df is not None else "chennai")
            Locality = st.text_input("Locality", sample_df["Locality"].iloc[0] if sample_df is not None else "locality_84")

        # ------------------ COLUMN 2 ------------------
        with col2:
            Price_per_SqFt = st.number_input("Price per SqFt", 100.0, 100000.0, 8000.0)
            Year_Built = st.number_input("Year Built", 1900, 2025, 2005)
            Furnished_Status = st.selectbox("Furnished Status", ["furnished", "unfurnished", "semi-furnished"])
            Floor_No = st.number_input("Floor No", 0, 50, 2)
            Total_Floors = st.number_input("Total Floors", 1, 50, 5)

        # ------------------ COLUMN 3 ------------------
        with col3:
            Nearby_Schools = st.number_input("Nearby Schools", 0, 50, 5)
            Nearby_Hospitals = st.number_input("Nearby Hospitals", 0, 50, 3)
            Parking_Space = st.selectbox("Parking Space", ["yes", "no"])
            Security = st.selectbox("Security", ["yes", "no"])
            Availability_Status = st.selectbox("Availability Status",
                ["ready_to_move", "under_construction"]
            )

        amenities_str = st.text_input("Amenities (comma-separated)", "gym,pool,garden")

        submit = st.form_submit_button("Predict Price")


    if submit:
        # Prepare input dict
        amen_list = [a.strip().lower().replace(" ","") for a in amenities_str.split(",") if a.strip()]

        input_dict = {
            "State": State,
            "City": City,
            "Locality": Locality,
            "Property_Type": Property_Type,
            "BHK": BHK,
            "Size_in_SqFt": Size_in_SqFt,
            "Price_per_SqFt": Price_per_SqFt,
            "Year_Built": Year_Built,
            "Furnished_Status": Furnished_Status,
            "Floor_No": Floor_No,
            "Total_Floors": Total_Floors,
            "Age_of_Property": 2025 - Year_Built,
            "Nearby_Schools": Nearby_Schools,
            "Nearby_Hospitals": Nearby_Hospitals,
            "Public_Transport_Accessibility": "high",
            "Parking_Space": Parking_Space,
            "Security": Security,
            "Amenities": ",".join(amen_list),
            "Facing": "north",
            "Owner_Type": "owner",
            "Availability_Status": Availability_Status
        }

        X_in = pd.DataFrame([input_dict])
        st.write("### Input Used:", X_in)

        # ----------------------
        # PRICE PREDICTION
        # ----------------------
        pred_price = pipeline.predict(X_in)[0]

        if sample_df is not None:
            q1 = np.percentile(sample_df["Price_in_Lakhs"], 33)
            q2 = np.percentile(sample_df["Price_in_Lakhs"], 66)
        else:
            q1, q2 = 132.46, 254.145

        pred_cat = price_to_category(pred_price, q1, q2)

        st.success("### 🎉 Prediction Complete!")
        st.metric("Predicted Price (Lakhs)", f"{pred_price:.2f}")
        st.metric("Price Category", pred_cat)

        # =============================================================================
        #                              SHAP EXPLAINABILITY
        # =============================================================================
        st.markdown("---")
        st.markdown("## 📊 SHAP Explainability")

        try:
            preprocessor = pipeline.named_steps["prep"]

            # Auto-detect model step
            if "xgb" in pipeline.named_steps:
                model = pipeline.named_steps["xgb"]
            elif "rf" in pipeline.named_steps:
                model = pipeline.named_steps["rf"]
            else:
                st.error("Model step not found!")
                model = None

            if model is None:
                raise ValueError("No model found in pipeline.")

            # Transform sample training data
            if sample_df is not None:
                X_train_transform = preprocessor.transform(sample_df.drop(columns=["Price_in_Lakhs"]))
            else:
                X_train_transform = None

            explainer = shap.TreeExplainer(model)

            # ------------------ GLOBAL SHAP ------------------
            if X_train_transform is not None:
                st.write("### 🌍 Global SHAP Summary")

                sample_rows = min(2000, X_train_transform.shape[0])
                idx = np.random.choice(X_train_transform.shape[0], sample_rows, replace=False)
                shap_vals_global = explainer.shap_values(X_train_transform[idx])

                fig1 = plt.figure(figsize=(10,4))
                shap.summary_plot(shap_vals_global, X_train_transform[idx], show=False)
                st.pyplot(fig1)

            # ------------------ LOCAL SHAP ------------------
            st.write("### 🎯 Local SHAP Explanation (This Prediction)")

            X_in_trans = preprocessor.transform(X_in)
            shap_local = explainer.shap_values(X_in_trans)

            fig2 = plt.figure(figsize=(10,4))
            shap.plots._waterfall.waterfall_legacy(
                explainer.expected_value,
                shap_local[0]
            )
            st.pyplot(fig2)

        except Exception as e:
            st.error(f"SHAP failed: {e}")


# Sidebar footer
st.sidebar.markdown("---")
st.sidebar.write("Model loaded from: `xgb_pipeline.joblib`")
