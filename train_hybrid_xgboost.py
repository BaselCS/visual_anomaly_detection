import os
import pandas as pd
import numpy as np
import xgboost as xgb
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, roc_curve, precision_score, recall_score, f1_score, accuracy_score

def get_latest_ti_dir(base_dir="trained_models"):
    """تم التعديل لتبحث حصريا عن مجلدات الانعكاس النصي"""
    if not os.path.exists(base_dir):
        raise FileNotFoundError(f"Directory {base_dir} does not exist.")
    max_num = 0
    latest_folder = None
    for folder_name in os.listdir(base_dir):
        if folder_name.startswith("train_ti_"):
            try:
                num = int(folder_name.replace("train_ti_", ""))
                if num > max_num:
                    max_num = num
                    latest_folder = folder_name
            except ValueError:
                continue
    if latest_folder is None:
        raise FileNotFoundError(f"No Textual Inversion directories found in {base_dir}")
    return os.path.join(base_dir, latest_folder)

# ==========================================
# 1. إعداد المسارات وتحميل البيانات
# ==========================================
print("--- Starting Hybrid Model Training (Textual Inversion Mode) ---")
latest_dir = get_latest_ti_dir()
csv_path = os.path.join(latest_dir, 'results_database.csv')

df = pd.read_csv(csv_path).dropna()

# تم التعديل لتطابق القيم المستخرجة في anomal_score_ti.py
TARGET_STRENGTH = 0.40
TARGET_GUIDANCE = 6.5

df_filtered = df[(df['Strength'] == TARGET_STRENGTH) & (df['Guidance'] == TARGET_GUIDANCE)].copy()

print(f"Total Unique Images after filtering: {len(df_filtered)}")

if len(df_filtered) == 0:
    raise ValueError("No data found for the specified Strength and Guidance. Please check your CSV file.")

features = ['L1', 'L2', 'MS_SSIM', 'LPIPS', 'Max_Patch']
X = df_filtered[features].values
y = df_filtered['Label'].values

# ==========================================
# 2. التدريب وإيجاد العتبة الصارمة داخل الطيات
# ==========================================
kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=999)

oof_preds_proba = np.zeros(len(y))
oof_preds_classes = np.zeros(len(y))
fold_aucs = []

print("Training XGBoost with Strict In-Fold Thresholding...")

for fold, (train_idx, test_idx) in enumerate(kf.split(X, y)):
    X_train, y_train = X[train_idx], y[train_idx]
    X_test, y_test = X[test_idx], y[test_idx]
    
    model = xgb.XGBClassifier(
        objective='binary:logistic',
        eval_metric='logloss',
        n_estimators=50,
        learning_rate=0.05,
        max_depth=2,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=999
    )
    
    model.fit(X_train, y_train)
    
    train_preds_proba = model.predict_proba(X_train)[:, 1]
    fpr_train, tpr_train, thresholds_train = roc_curve(y_train, train_preds_proba)
    best_idx_train = np.argmax(tpr_train - fpr_train)
    fold_threshold = thresholds_train[best_idx_train]
    
    test_preds_proba = model.predict_proba(X_test)[:, 1]
    oof_preds_proba[test_idx] = test_preds_proba
    oof_preds_classes[test_idx] = (test_preds_proba >= fold_threshold).astype(int)
    
    fold_auc = roc_auc_score(y_test, test_preds_proba)
    fold_aucs.append(fold_auc)
    print(f"Fold {fold+1} | AUC: {fold_auc:.4f} | Strict Threshold: {fold_threshold:.4f}")

# ==========================================
# 3. التقييم النهائي
# ==========================================
final_auc = roc_auc_score(y, oof_preds_proba)
precision = precision_score(y, oof_preds_classes, zero_division=0)
recall = recall_score(y, oof_preds_classes, zero_division=0)
f1 = f1_score(y, oof_preds_classes, zero_division=0)
accuracy = accuracy_score(y, oof_preds_classes)

print("\n" + "="*40)
print("--- TEXTUAL INVERSION HYBRID MODEL RESULTS ---")
print(f"Target Config: Strength={TARGET_STRENGTH}, Guidance={TARGET_GUIDANCE}")
print(f"Mean Fold AUC: {np.mean(fold_aucs):.4f}")
print(f"Overall AUC:   {final_auc:.4f}")
print(f"Strict F1-Score: {f1:.4f}")
print(f"Strict Precision:{precision:.4f}")
print(f"Strict Recall:   {recall:.4f}")
print(f"Strict Accuracy: {accuracy:.4f}")
print("="*40 + "\n")

# ==========================================
# 4. حفظ الموديل النهائي والعتبة
# ==========================================
import json

fpr, tpr, thresholds = roc_curve(y, oof_preds_proba)
best_idx = np.argmax(tpr - fpr)
optimal_threshold = thresholds[best_idx]

final_model = xgb.XGBClassifier(objective='binary:logistic', n_estimators=50, learning_rate=0.05, max_depth=2, random_state=999)
final_model.fit(X, y)

model_path = os.path.join(latest_dir, f'xgboost_hybrid_S{TARGET_STRENGTH}_G{TARGET_GUIDANCE}.json')
final_model.save_model(model_path)

metadata_path = os.path.join(latest_dir, f'xgboost_metadata_S{TARGET_STRENGTH}_G{TARGET_GUIDANCE}.json')
with open(metadata_path, 'w') as f:
    json.dump({'optimal_threshold': float(optimal_threshold)}, f)

print(f"Final Model saved to: {model_path}")
print(f"Threshold metadata saved to: {metadata_path}")