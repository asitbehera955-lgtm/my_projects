import pickle

import pandas as pd
import streamlit as st

MODEL_PATH = "sales_prediction_model.pkl"
FEATURES = ["youtube", "facebook", "newspaper"]

st.set_page_config(page_title="Sales Predictor", page_icon="📈", layout="centered")


@st.cache_resource
def load_model(path: str):
    """Load the trained model once and reuse it across reruns."""
    try:
        with open(path, "rb") as f:
            return pickle.load(f)
    except Exception:
        # Fallback in case the model was saved with joblib
        import joblib
        return joblib.load(path)


st.title("📈 Sales Prediction")
st.write("Enter your advertising budgets to predict expected sales.")

try:
    model = load_model(MODEL_PATH)
except FileNotFoundError:
    st.error(f"Model file '{MODEL_PATH}' not found. Place it in the same folder as this app.")
    st.stop()
except Exception as e:
    st.error(f"Could not load the model: {e}")
    st.stop()

with st.form("prediction_form"):
    col1, col2, col3 = st.columns(3)
    youtube = col1.number_input("YouTube budget", min_value=0.0, value=100.0, step=10.0)
    facebook = col2.number_input("Facebook budget", min_value=0.0, value=30.0, step=5.0)
    newspaper = col3.number_input("Newspaper budget", min_value=0.0, value=20.0, step=5.0)
    submitted = st.form_submit_button("Predict Sales", use_container_width=True)

if submitted:
    # Use a DataFrame so column names/order match what the model was trained on
    input_df = pd.DataFrame([[youtube, facebook, newspaper]], columns=FEATURES)
    prediction = model.predict(input_df)[0]

    st.subheader("Result")
    st.metric("Predicted Sales", f"{prediction:,.2f}")

    with st.expander("Show input data"):
        st.dataframe(input_df, hide_index=True)

# Optional: model details
with st.sidebar:
    st.header("Model info")
    if hasattr(model, "coef_"):
        coef_df = pd.DataFrame({"Feature": FEATURES, "Coefficient": model.coef_.ravel()})
        st.dataframe(coef_df, hide_index=True)
        st.caption(f"Intercept: {float(model.intercept_ if not hasattr(model.intercept_, '__len__') else model.intercept_[0]):.4f}")
