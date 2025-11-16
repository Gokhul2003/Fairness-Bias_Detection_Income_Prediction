============================================================
💼 Fairness Auditing and Bias Detection in Income Prediction
============================================================

Project Overview:
-----------------
This project detects and analyzes bias in machine learning models predicting income (>50K or <=50K) using the UCI Adult dataset. 
It evaluates fairness across protected attributes like age, education, race, sex, marital status, and occupation using metrics like 
Demographic Parity (DP%) and Equalized Odds (EO%). 

The web app is built using Streamlit, visualizes model bias with Plotly, and allows interactive exploration of fairness metrics.

Project Structure:
-----------------
Fairness-Auditing/
│
├─ app.py                  # Main Streamlit application
├─ data_loader.py          # Dataset loading and preprocessing
├─ fairness_metrics.py     # Model training and bias computations
├─ requirements.txt        # Required Python packages
└─ README.txt              # Project documentation

Features:
---------
- Load and preprocess the UCI Adult dataset  
- Train ML models (RandomForest, LightGBM) for income prediction  
- Compute DP% (selection bias) and EO% (accuracy gap) for protected attributes  
- Calculate Bias Score (DP% + EO%) and highlight the most biased attribute  
- Interactive Plotly visualizations for easy bias interpretation  
- Teacher-friendly explanations for each attribute  

Protected Attributes Analyzed:
------------------------------
- Age  
- Education  
- Race  
- Gender (Sex)  
- Marital Status  
- Occupation  

DP% → Measures who gets selected more. Higher % = stronger bias  
EO% → Measures accuracy gap across groups. Higher % = stronger bias

Installation Instructions:
--------------------------
1. Clone the repository:
   git clone https://github.com/Gokhul2003/FairnessAuditing-Bias_Detection_Income_Prediction.git
   cd fairness-auditing

2. Create and activate a Python environment (recommended):
   python -m venv venv
   # Windows
   venv\Scripts\activate
   # macOS/Linux
   source venv/bin/activate

3. Install required packages:
   pip install -r requirements.txt

Required Python Packages:
-------------------------
streamlit
pandas
numpy
scikit-learn
fairlearn
lightgbm
matplotlib
plotly
imblearn

Running the Streamlit App:
--------------------------
streamlit run app.py

- The app will open in your browser.
- Click ▶️ Start Processing and Analysis to run the bias analysis.  
- Explore Model Performance, Fairness Summary, and Interactive Bias Visualization tabs.

How Bias Metrics are Calculated:
--------------------------------
1. Demographic Parity Difference (DP%):
   - Measures how positive outcomes are distributed across groups.
   - Example: Older people predicted high income 72% of the time; younger people rarely. DP% = 72%.

2. Equalized Odds Difference (EO%):
   - Measures whether model accuracy is similar across groups (TPR/FPR).
   - Example: Predictions are 100% accurate for older, almost 0% for younger. EO% = 100%.

3. Bias Score:
   - Sum of DP% + EO% to quantify overall bias for each attribute.

Project Highlights:
-------------------
- Modular design with 3 separate files:
  1. data_loader.py → Load & preprocess dataset  
  2. fairness_metrics.py → Train models & compute bias metrics  
  3. app.py → Streamlit UI and visualization  

- Interactive bar charts with hover explanations  
- Teacher-friendly interpretation of each attribute’s bias  
- Helps understand bias in ML models and promotes fairness-aware AI  

Tips for Users:
---------------
- Use RandomForest or LightGBM for training to see differences in fairness.  
- Ensure imbalanced-learn is installed to apply SMOTE for balancing classes.  
- Hover over bars in the visualization for detailed explanations per attribute.  
- Check Bias Score to quickly identify which attribute is most unfairly treated.  

Contact / Feedback:
------------------
Author:Gokhul Kooranchi 
Email: thegokhul.kooranchi@gmail.com 
GitHub: https://github.com/Gokhul2003  
