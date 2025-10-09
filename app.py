import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from src.data_loader import load_dataset
from src.preprocessing import preprocess
from src.models import get_models, train_and_eval
from src.fairness import fairness_summary, confusion_matrices
from src.mitigation import run_exponentiated_gradient
from src.utils import save_experiment

st.set_page_config(layout='wide', page_title='Fairness Audit - Income Prediction')
st.title('Fairness Auditing & Bias Mitigation in Income Prediction')

# Sidebar controls
st.sidebar.header('Controls')
uploaded = st.sidebar.file_uploader('Upload CSV (optional)', type=['csv'])

if uploaded is not None:
    df = pd.read_csv(uploaded)
else:
    df = load_dataset()

st.sidebar.write('Rows: {} Columns: {}'.format(df.shape[0], df.shape[1]))
st.write('### Dataset preview')
st.dataframe(df.head())

if st.button('Run Full Pipeline'):
    # Preprocessing
    X_train, X_test, y_train, y_test, prot_train, prot_test, preproc, prot_col, target_col = preprocess(df)
    st.success(f'Preprocessing completed. Detected protected attribute: {prot_col} and target: {target_col}')
    
    # Train models
    models = get_models()
    results = train_and_eval(models, X_train, y_train, X_test, y_test)
    
    st.subheader('Model Results')
    for mname, res in results.items():
        st.write(f"{mname}: Accuracy={res['accuracy']:.3f}, F1={res['f1']:.3f}, Precision={res['precision']:.3f}, Recall={res['recall']:.3f}")
    
    # Baseline fairness using LogisticRegression if present
    baseline = 'LogisticRegression' if 'LogisticRegression' in results else list(results.keys())[0]
    preds = results[baseline]['preds']
    fairness = fairness_summary(y_test, preds, prot_test)
    
    # ---- Display Fairness Summary ----
    st.subheader("🔍 Fairness Summary")
    st.json(fairness)

    # ---- Plot Group-wise Fairness Metrics (Improved Style) ----
    st.subheader("📊 Group-wise Fairness Visualization")
    group_df = pd.DataFrame(fairness['by_group']).T  # transpose for plotting

    for metric in group_df.index:
        st.write(f"### {metric.upper()} by Group")
        fig, ax = plt.subplots(figsize=(6, 3))
        sns.barplot(x=group_df.columns, y=group_df.loc[metric].values, palette="pastel", ax=ax)

        # Add value labels above bars
        for i, v in enumerate(group_df.loc[metric].values):
            ax.text(i, v + (0.01 if v >= 0 else -0.01), f"{v:.2f}", ha='center', va='bottom', fontsize=10, color='black')

        # Aesthetic improvements
        ax.set_xlabel('Group', fontsize=10)
        ax.set_ylabel(metric.upper(), fontsize=10)
        ax.set_ylim(min(0, group_df.loc[metric].min() - 0.05), group_df.loc[metric].max() + 0.05)
        sns.despine(ax=ax)
        plt.tight_layout()

        st.pyplot(fig)
        plt.close(fig)

    
    # ---- Fairness Risk Indicator ----
    st.write("### 🚦 Fairness Risk Indicator")
    bias_score = abs(fairness['overall']['demographic_parity_difference'])
    if bias_score < 0.05:
        st.success("✅ Low Bias (Fair Model)")
    elif bias_score < 0.15:
        st.warning("⚠️ Moderate Bias (Needs Attention)")
    else:
        st.error("🚨 High Bias (Unfair Predictions)")

    # Confusion matrices by group
    st.subheader('Confusion Matrices by Group (Baseline)')
    st.json(confusion_matrices(y_test, preds, prot_test))

    # Save experiment summary
    summary = {
        'baseline': baseline,
        'metrics': {k: {'accuracy': v['accuracy'], 'f1': v['f1']} for k, v in results.items()},
        'fairness': fairness
    }
    save_experiment(summary)

    # Mitigation step
    st.subheader('In-processing Mitigation (ExponentiatedGradient)')
    if st.button('Run ExponentiatedGradient - DemographicParity'):
        mitig = run_exponentiated_gradient(X_train, y_train, prot_train, constraint='demographic_parity', eps=0.01)
        mitig_preds = mitig.predict(X_test)
        mitig_fair = fairness_summary(y_test, mitig_preds, prot_test)
        st.write('Fairness after mitigation:')
        st.json(mitig_fair)
