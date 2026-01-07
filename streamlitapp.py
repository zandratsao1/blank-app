import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
import plotly.graph_objects as go
import os
from sklearn.metrics import confusion_matrix, recall_score, precision_score, f1_score

# --- 1. PAGE CONFIG & CUSTOM THEME ---
st.set_page_config(page_title="Model Testing Tools", layout="wide")

# Custom CSS for Light Blue Background and Black Text
st.markdown("""
    <style>
    .stApp {
        background-color: #e3f2fd;
        color: #000000;
    }
    h1, h2, h3, h4, h5, h6, p, label, .stMetric, .stSelectbox label, .stSlider label {
        color: #000000 !important;
    }
    .stMetric {
        background-color: #ffffff;
        padding: 15px;
        border-radius: 10px;
        box-shadow: 2px 2px 5px rgba(0,0,0,0.05);
    }
    .stButton>button {
        background-color: #1e88e5;
        color: white;
    }
    </style>
    """, unsafe_allow_html=True)

# --- 2. ASSET LOADING ---
@st.cache_resource
def load_all_assets():
    def safe_load_csv(file_name):
        if os.path.exists(file_name):
            return pd.read_csv(file_name)
        return None

    assets = {
        "Logistic Regression": {
            "model": joblib.load('lr_model.pkl'),
            "scaler": joblib.load('lr_scaler.pkl'),
            "val": safe_load_csv("val_data_lr.csv"),
            "test": safe_load_csv("test_data_lr.csv")
        },
        "Random Forest": {
            "model": joblib.load('rf_model.pkl'),
            "scaler": joblib.load('rf_scaler.pkl'),
            "val": safe_load_csv("val_data_rf.csv"),
            "test": safe_load_csv("test_data_rf.csv")
        },
        "XGBoost": {
            "model": joblib.load('xgb_model.pkl'),
            "scaler": joblib.load('xgb_scaler.pkl'),
            "val": safe_load_csv("val_data_xgb.csv"),
            "test": safe_load_csv("test_data_xgb.csv")
        }
    }
    return assets

try:
    ASSETS = load_all_assets()
except Exception as e:
    st.error(f"Critical Error: Ensure you have run the training script and all .pkl/.csv files exist. {e}")

# --- 3. SIDEBAR NAVIGATION ---
with st.sidebar:
    st.title("🚗 Model Testing Tools")
    page = st.radio("📍 MENU", ["1. Data Overview", "2. Model Performance", "3. Churn Predictor"])
    st.divider()
    threshold = st.slider("Risk Threshold", 0.0, 1.0, 0.5, 0.05)
    st.caption("Adjusting the threshold helps balance Recall and Precision.")

# --- 4. PAGE 1: DATA OVERVIEW ---
if page == "1. Data Overview":
    st.markdown("### 📊 Dataset & Preprocessing Overview")
    
    st.write("""
    This project utilizes the **Waze User Churn Dataset**, containing user behavior data 
    such as driving frequency and app usage patterns. The goal is to predict user churn 
    (label 1) versus retention (label 0).
    """)

    st.markdown("#### 🛠️ Data Strategy & Balancing")
    st.write("""
    The dataset exhibits a significant **Class Imbalance** (only 17.7% churned). To fix this:
    * **Scaling:** Normalized features using `StandardScaler` for accurate model math.
    * **SMOTE:** Synthetically generated new churn samples in the training set to create 
      a 50/50 balance.
    * **Anomaly Capping:** Outliers were capped at the 95th percentile to improve stability.
    """)

    

    st.markdown("#### ❓ Why is the F1 Score low?")
    st.warning("""
    F1 scores in churn prediction often fall between 0.35 and 0.45 due to:
    1. **Weak Feature Correlation:** Human behavior (km driven) doesn't always explain why people leave.
    2. **SMOTE Noise:** Balancing classes creates 'artificial' patterns, increasing False Positives.
    3. **Information Gap:** We lack data on app performance or competitor behavior.
    """)

    dist_df = pd.DataFrame({'Label': ['Stayed (0)', 'Churned (1)'], 'Percentage': [82.3, 17.7]})
    fig = px.pie(dist_df, names='Label', values='Percentage', hole=0.4,
                 color_discrete_sequence=['#1e88e5', '#fb8c00'], 
                 title="Original Class Distribution (Imbalanced)")
    fig.update_layout(margin=dict(t=30, b=0, l=0, r=0), height=300)
    st.plotly_chart(fig, use_container_width=True)

# --- 5. PAGE 2: MODEL PERFORMANCE ---
elif page == "2. Model Performance":
    st.title("🧪 Model Evaluation: Validation vs. Test")
    
    # 1. 扩充后的 Metric Guide (保持不变)
    with st.expander("📖 Metric Guide: What do these numbers mean?"):
        col_m1, col_m2, col_m3 = st.columns(3)
        with col_m1:
            st.markdown("##### **Recall & Precision**")
            st.write("Recall is your 'Catch Rate', while Precision is your 'Certainty'.")
        with col_m2:
            st.markdown("##### **F1 Score**")
            st.write("The balance between Recall and Precision.")
        with col_m3:
            st.markdown("##### **ROC Curve & AUC**")
            st.write("Measures the model's ability to distinguish between classes at all thresholds.")

        st.divider()
        st.markdown("#### 📈 How to read the ROC Curve?")
        st.write("""
        * **What it is:** A plot of the **True Positive Rate** (Recall) vs. **False Positive Rate** (False Alarms).
        * **The Diagonal Line:** Represents a 'Random Guess'.
        * **The Curve:** The closer the curve 'hugs' the top-left corner, the better the model is.
        * **AUC (Area Under Curve):** **0.5:** Random; **0.7-0.8:** Good; **0.9+:** Outstanding.
        """)
        
        st.divider()
        st.markdown("#### 📊 Feature Importance Guide")
        st.write("""
        * **What it shows:** Which input variables (e.g., `activity_days`) had the most influence on the model's predictions.
        * **How to read:** Longer bars mean more importance.
        * **Insights:** Can help you understand what drives churn and focus on key user behaviors.
        """)

    # 2. 模型选择与数据准备
    model_name = st.selectbox("Select Model", ["XGBoost", "Random Forest", "Logistic Regression"])
    data = ASSETS[model_name]
    model = data['model']
    v_df, t_df = data['val'], data['test']

    if v_df is not None and t_df is not None:
        X_v, y_v = v_df.drop(columns=['label2']), v_df['label2']
        X_t, y_t = t_df.drop(columns=['label2']), t_df['label2']
        
        # 预测概率
        v_probs = model.predict_proba(X_v)[:, 1]
        t_probs = model.predict_proba(X_t)[:, 1]
        
        # 计算 ROC 曲线数据
        from sklearn.metrics import roc_curve, auc
        fpr, tpr, _ = roc_curve(y_t, t_probs)
        roc_auc = auc(fpr, tpr)

        # 主要布局：左边是指标和混淆矩阵，右边是 ROC 曲线
        col_main_left, col_main_right = st.columns([1, 1])

        with col_main_left:
            v_pred = (v_probs >= threshold).astype(int)
            t_pred = (t_probs >= threshold).astype(int)
            
            st.subheader("Performance Metrics")
            c1, c2 = st.columns(2)
            c1.metric("F1 Score (Test)", f"{f1_score(y_t, t_pred):.3f}")
            c2.metric("AUC Score (Test)", f"{roc_auc:.3f}")
            
            cm_t = confusion_matrix(y_t, t_pred)
            st.plotly_chart(px.imshow(cm_t, text_auto=True, x=['Stay', 'Churn'], y=['Stay', 'Churn'], 
                                       color_continuous_scale='Blues', title="Test Confusion Matrix"), use_container_width=True)

        with col_main_right:
            st.subheader("ROC Curve (Test Set)")
            fig_roc = go.Figure()
            # 绘制对角线 (Random Guess)
            fig_roc.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode='lines', line=dict(dash='dash', color='gray'), name='Random Guess'))
            # 绘制模型曲线
            fig_roc.add_trace(go.Scatter(x=fpr, y=tpr, mode='lines', line=dict(color='#1e88e5', width=3), name=f'AUC = {roc_auc:.3f}'))
            
            fig_roc.update_layout(
                xaxis_title='False Positive Rate (FPR)',
                yaxis_title='True Positive Rate (TPR / Recall)',
                margin=dict(l=20, r=20, t=40, b=20),
                height=400,
                legend=dict(yanchor="bottom", y=0.01, xanchor="right", x=0.99)
            )
            st.plotly_chart(fig_roc, use_container_width=True)

        st.markdown("---") # 分隔线

        # --- NEW: Feature Importance Section ---
        st.subheader("📊 Feature Importance")
        
        feature_names = X_t.columns.tolist() # 获取特征名称
        
        if model_name in ["XGBoost", "Random Forest"]:
            importances = model.feature_importances_
        elif model_name == "Logistic Regression":
            # 对于 LR，使用系数的绝对值作为重要性
            importances = np.abs(model.coef_[0])
        else:
            importances = np.zeros(len(feature_names)) # Fallback

        feature_importance_df = pd.DataFrame({
            'Feature': feature_names,
            'Importance': importances
        }).sort_values(by='Importance', ascending=False)

        fig_feature_importance = px.bar(
            feature_importance_df.head(10), # 只显示前10个最重要的特征
            x='Importance',
            y='Feature',
            orientation='h',
            title=f"Top 10 Feature Importances for {model_name}",
            color_discrete_sequence=['#1e88e5']
        )
        fig_feature_importance.update_layout(yaxis={'categoryorder':'total ascending'}) # 让最重要的特征在上面
        st.plotly_chart(fig_feature_importance, use_container_width=True)

    else:
        st.warning(f"CSV data for {model_name} not found. Please ensure you have run the training script to export CSV files.")

# --- 6. PAGE 3: CHURN PREDICTOR ---
elif page == "3. Churn Predictor":
    st.title("🔮 Interactive Churn Sandbox")
    
    col_in, col_out = st.columns([1, 1.2])
    
    with col_in:
        st.subheader("1. User Behavior Inputs")
        m_type = st.radio("Choose Model", ["XGBoost", "Random Forest", "Logistic Regression"], horizontal=True)
        
        with st.container(border=True):
            sessions = st.slider("Monthly Sessions", 0, 300, 50)
            drives = st.slider("Total Drives", 0, 300, 20)
            km = st.slider("Total Kilometers Driven", 0, 5000, 800)
            activity_days = st.slider("Total Activity Days", 0, 31, 12) 
            days = st.slider("Driving Days/Month", 0, 31, 10)
            onboarding = st.number_input("Days Since Onboarding", 0, 4000, 500)
            duration = st.slider("Total Duration (Min)", 0, 10000, 1500)

        km_per_day = km / days if days > 0 else 0
        is_pro = 1 if (drives >= 60 and days >= 15) else 0

    with col_out:
        st.subheader("2. Risk Analysis")
        
        input_data = {
            'sessions': sessions, 'drives': drives, 'total_sessions': sessions * 1.2,
            'n_days_after_onboarding': onboarding, 'total_navigations_fav1': 5,
            'total_navigations_fav2': 2, 'driven_km_drives': km,
            'duration_minutes_drives': duration, 'activity_days': activity_days,
            'driving_days': days, 'km_per_driving_day': km_per_day, 'professional_driver': is_pro
        }
        
        input_df = pd.DataFrame([input_data])
        
        if m_type == "Logistic Regression":
            input_df = input_df.drop(columns=['sessions', 'driving_days'])
        
        scaler = ASSETS[m_type]['scaler']
        model = ASSETS[m_type]['model']

        try:
            input_scaled = scaler.transform(input_df)
            prob = model.predict_proba(input_scaled)[0][1]

            fig_gauge = go.Figure(go.Indicator(
                mode = "gauge+number", value = prob*100, number={'suffix': "%"},
                title = {'text': "Churn Probability"},
                gauge = {'axis': {'range': [0, 100]},
                         'bar': {'color': "#1e88e5"},
                         'steps': [
                             {'range': [0, 30], 'color': "#c8e6c9"},
                             {'range': [30, 70], 'color': "#fff9c4"},
                             {'range': [70, 100], 'color': "#ffcdd2"}]}
            ))
            st.plotly_chart(fig_gauge, use_container_width=True)

            if prob >= threshold:
                st.error(f"Prediction: HIGH RISK OF CHURN")
            else:
                st.success(f"Prediction: LOYAL USER")
        except Exception as e:
            st.error(f"Error: {e}")

    st.divider()
    st.write("💡 Insight: Adjusting 'Activity Days' usually has the highest impact on churn risk.")