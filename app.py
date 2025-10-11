# app.py
import streamlit as st
import pandas as pd
import plotly.express as px
from data_loader import load_adult_data, preprocess_data
from fairness_metrics import train_models, compute_bias
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

st.set_page_config(page_title="Fairness Auditing in Income Prediction", layout="wide")
st.title("💼 Fairness Auditing and Bias Detection in Income Prediction")
st.write("This app detects and analyzes bias across multiple attributes like gender, race, age, education, and more using ML models.")

# Load & preprocess
df = load_adult_data()
st.success(f"✅ Adult dataset loaded: {df.shape}")
st.subheader("📋 Dataset Preview")
st.dataframe(df.head())

if st.button("▶️ Start Processing and Analysis"):

    # Protected attributes
    protected_attributes = ["age", "education", "race", "sex", "marital_status", "occupation"]
    st.write(f"**🔹 Detected Protected Attributes:** {', '.join(protected_attributes)}")
    
    # Preprocess & split
    X_train, X_test, y_train, y_test = preprocess_data(df)

    # Train models
    models = train_models(X_train, y_train)

    # Tab 1: Model performance
    tabs = st.tabs(["📈 Model Performance", "⚖️ Fairness Analysis", "📊 Bias Visualization"])
    with tabs[0]:
        st.subheader("📈 Model Performance")
        for name, model in models.items():
            y_pred = model.predict(X_test)
            st.markdown(f"""
            **{name}:**
            - Accuracy = {accuracy_score(y_test, y_pred):.3f}
            - F1 Score = {f1_score(y_test, y_pred):.3f}
            - Precision = {precision_score(y_test, y_pred):.3f}
            - Recall = {recall_score(y_test, y_pred):.3f}
            """)

    # Tab 2: Fairness Analysis
    with tabs[1]:
        st.subheader("⚖️ Fairness Summary")
        st.info("""
        **DP% → Measures who gets selected more. Higher % = stronger bias.**  
        **EO% → Measures accuracy gap across groups. Higher % = stronger bias.**
        """)
        bias_df = compute_bias(models, X_test, y_test, protected_attributes)
        st.dataframe(bias_df[["Attribute","DP%","EO%","Bias Score","Explanation"]])

        # Overall bias rating
        avg_bias = bias_df["Bias Score"].abs().mean()
        if avg_bias < 5:
            st.success("✅ Low Bias (Fair Model)")
        elif avg_bias < 15:
            st.warning("⚠️ Moderate Bias")
        else:
            st.error("❌ High Bias (Unfair Model Detected)")

    # Tab 3: Bias Visualization
    with tabs[2]:
        st.subheader("📊 Interactive Bias Visualization Across Attributes")
        bias_melt = bias_df.melt(id_vars=["Attribute","Explanation"], value_vars=["DP%","EO%"],
                                 var_name="Metric", value_name="Bias")
        fig = px.bar(
            bias_melt, x="Attribute", y="Bias", color="Metric", barmode="group",
            text="Bias", hover_data=["Explanation"]
        )
        fig.update_layout(title="Bias Across Protected Attributes", yaxis_title="Bias (%)", xaxis_title="Attribute")
        st.plotly_chart(fig, use_container_width=True)

        # Highlight worst attribute
        worst_attr = bias_df.loc[bias_df["Bias Score"].idxmax(), "Attribute"]
        st.info(f"📌 Attribute with **highest detected bias**: `{worst_attr}`")
