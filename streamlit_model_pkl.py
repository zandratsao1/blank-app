import numpy as np
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import recall_score, precision_score, f1_score

# --- 1. Data Cleaning & Feature Engineering ---
def get_my_data():
    df = pd.read_csv("waze_dataset.csv")
    df = df.drop('ID', axis=1) 

    df['km_per_driving_day'] = df['driven_km_drives'] / df['driving_days']
    df.loc[df['km_per_driving_day'] == np.inf, 'km_per_driving_day'] = 0

    df['professional_driver'] = np.where((df['drives'] >= 60) & (df['driving_days'] >= 15), 1, 0)
    df = df.dropna(subset=['label'])

    # Anomaly capping (95th percentile)
    for column in ['sessions', 'drives', 'total_sessions', 'total_navigations_fav1',
               'total_navigations_fav2', 'driven_km_drives', 'duration_minutes_drives']:
        threshold = df[column].quantile(0.95)
        df.loc[df[column] > threshold, column] = threshold

    df['label2'] = np.where(df['label'] == 'churned', 1, 0)
    return df

# --- 2. Logistic Regression ---
def train_save_model_LR(df):
    X = df.drop(columns=['label', 'label2', 'device', 'sessions', 'driving_days'])
    y = df['label2']
    
    X_tr, X_test, y_tr, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
    X_train, X_val, y_train, y_val = train_test_split(X_tr, y_tr, test_size=0.25, stratify=y_tr, random_state=42)
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train) 
    X_val_scaled = scaler.transform(X_val)         
    X_test_scaled = scaler.transform(X_test)       
    
    smote = SMOTE(random_state=42)
    X_train_resampled, y_train_resampled = smote.fit_resample(X_train_scaled, y_train)
    
    clf = LogisticRegression(penalty='none', max_iter=500)
    clf.fit(X_train_resampled, y_train_resampled)
    
    # Save model and scaler
    joblib.dump(clf, 'lr_model.pkl')
    joblib.dump(scaler, 'lr_scaler.pkl') 
    
    # Save data for Streamlit
    val_df = pd.concat([pd.DataFrame(X_val_scaled, columns=X.columns), y_val.reset_index(drop=True)], axis=1)
    test_df = pd.concat([pd.DataFrame(X_test_scaled, columns=X.columns), y_test.reset_index(drop=True)], axis=1)
    
    # Save only features used in LR
    val_df.to_csv("val_data_lr.csv", index=False)
    test_df.to_csv("test_data_lr.csv", index=False)
    # --------------------------------------------------
    
    return clf, X_test_scaled, y_test

# --- 3. Random Forest ---
def train_save_model_rf(df):
    X = df.drop(columns=['label', 'label2', 'device'])
    y = df['label2']
    
    X_tr, X_test, y_tr, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
    X_train, X_val, y_train, y_val = train_test_split(X_tr, y_tr, test_size=0.25, stratify=y_tr, random_state=42)
    
    # Using Scaler here to improve SMOTE synthetic sample quality
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)

    smote = SMOTE(random_state=42)
    X_train_resampled, y_train_resampled = smote.fit_resample(X_train_scaled, y_train)

    rf = RandomForestClassifier(random_state=42)
    cv_params = {'max_depth': [None], 'n_estimators': [300], 'min_samples_leaf': [2]}
    
    rf_cv = GridSearchCV(rf, cv_params, scoring='recall', cv=4)
    rf_cv.fit(X_train_resampled, y_train_resampled)
    best_rf = rf_cv.best_estimator_
    
    joblib.dump(best_rf, 'rf_model.pkl') 
    joblib.dump(scaler, 'rf_scaler.pkl')

    # Save data for Streamlit
    val_df = pd.concat([pd.DataFrame(X_val_scaled, columns=X.columns), y_val.reset_index(drop=True)], axis=1)
    test_df = pd.concat([pd.DataFrame(X_test_scaled, columns=X.columns), y_test.reset_index(drop=True)], axis=1)
    val_df.to_csv("val_data_rf.csv", index=False)
    test_df.to_csv("test_data_rf.csv", index=False)

    return best_rf, X_test_scaled, y_test

# --- 4. XGBoost ---
def train_save_model_xgb(df):
    X = df.drop(columns=['label', 'label2', 'device'])
    y = df['label2']
    
    X_tr, X_test, y_tr, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
    X_train, X_val, y_train, y_val = train_test_split(X_tr, y_tr, test_size=0.25, stratify=y_tr, random_state=42)
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)
    
    smote = SMOTE(random_state=42)
    X_train_resampled, y_train_resampled = smote.fit_resample(X_train_scaled, y_train)

    xgb_base = XGBClassifier(random_state=42, eval_metric='logloss')
    cv_params = {'max_depth': [5], 'n_estimators': [300]}
    
    xgb_cv = GridSearchCV(xgb_base, cv_params, scoring='recall', cv=4)
    xgb_cv.fit(X_train_resampled, y_train_resampled)
    best_xgb = xgb_cv.best_estimator_

    joblib.dump(best_xgb, 'xgb_model.pkl')
    joblib.dump(scaler, 'xgb_scaler.pkl') 
    
    # Save data for Streamlit
    val_df = pd.concat([pd.DataFrame(X_val_scaled, columns=X.columns), y_val.reset_index(drop=True)], axis=1)
    test_df = pd.concat([pd.DataFrame(X_test_scaled, columns=X.columns), y_test.reset_index(drop=True)], axis=1)
    val_df.to_csv("val_data_xgb.csv", index=False)
    test_df.to_csv("test_data_xgb.csv", index=False)

    return best_xgb, X_test_scaled, y_test

# --- MAIN EXECUTION ---
df = get_my_data()

print("Training Logistic Regression...")
train_save_model_LR(df)

print("Training Random Forest...")
train_save_model_rf(df)

print("Training XGBoost...")
xgb_model, X_test_final, y_test_final = train_save_model_xgb(df)

# Final Output for Test Set Consistency
pd.DataFrame(X_test_final).to_csv("X_test.csv", index=False)
pd.Series(y_test_final).to_csv("y_test.csv", index=False)

print("All models trained and assets saved successfully.")