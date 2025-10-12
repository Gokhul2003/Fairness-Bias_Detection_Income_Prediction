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

# -------------------- Load & preview dataset --------------------
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

    # Tabs
    tabs = st.tabs(["📈 Model Performance", "⚖️ Fairness Analysis", "📊 Bias Visualization", "🏆 Best Model Summary"])
    
    # -------------------- Tab 1: Model Performance --------------------
    with tabs[0]:
        st.subheader("📈 Model Performance")
        results = []
        for name, model in models.items():
            y_pred = model.predict(X_test)
            acc = accuracy_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred)
            prec = precision_score(y_test, y_pred)
            rec = recall_score(y_test, y_pred)
            results.append([name, acc, f1, prec, rec])
        
        results_df = pd.DataFrame(results, columns=["Model", "Accuracy", "F1 Score", "Precision", "Recall"])
        st.dataframe(results_df)

    # -------------------- Tab 2: Fairness Analysis --------------------
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

    # -------------------- Tab 3: Bias Visualization --------------------
    with tabs[2]:
        st.subheader("📊 Interactive Bias Visualization")
        bias_melt = bias_df.melt(id_vars=["Attribute","Explanation"], value_vars=["DP%","EO%"],
                                 var_name="Metric", value_name="Bias")
        fig = px.bar(bias_melt, x="Attribute", y="Bias", color="Metric", barmode="group",
                     text="Bias", hover_data=["Explanation"])
        fig.update_layout(title="Bias Across Protected Attributes", yaxis_title="Bias (%)")
        st.plotly_chart(fig, use_container_width=True)
        worst_attr = bias_df.loc[bias_df["Bias Score"].idxmax(), "Attribute"]
        st.info(f"📌 Attribute with **highest detected bias**: `{worst_attr}`")
    
    # -------------------- Tab 4: Best Model Summary --------------------
    with tabs[3]:
        st.subheader("🏆 Model Comparison Summary")
        # Sort by Accuracy then F1 Score for tiebreaker
        best_model = results_df.sort_values(["Accuracy","F1 Score"], ascending=False).iloc[0]
        st.success(f"✅ **Best Model:** {best_model['Model']} with Accuracy = {best_model['Accuracy']:.3f} and F1 Score = {best_model['F1 Score']:.3f}")
        
        st.write("### 🔍 Detailed Comparison of All Models")
        st.dataframe(results_df.sort_values(["Accuracy","F1 Score"], ascending=False))
        st.caption("The best-performing model is selected based on highest Accuracy; F1 Score is used as a tiebreaker if needed.")
