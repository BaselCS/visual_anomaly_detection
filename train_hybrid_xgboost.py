import os
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import roc_auc_score, roc_curve, precision_score, recall_score, f1_score, accuracy_score
import json

def get_latest_oft_dir(base_dir="trained_models"):
    if not os.path.exists(base_dir): raise FileNotFoundError(f"Directory {base_dir} does not exist.")
    max_num = 0
    latest_folder = None
    for folder_name in os.listdir(base_dir):
        if folder_name.startswith("train_oft_"):
            try:
                num = int(folder_name.replace("train_oft_", ""))
                if num > max_num:
                    max_num = num
                    latest_folder = folder_name
            except ValueError: continue
    if latest_folder is None: raise FileNotFoundError(f"No OFT dirs in {base_dir}")
    return os.path.join(base_dir, latest_folder)

print("--- Starting Hybrid Model Training (Strict Scientific Mode) ---")
latest_dir = get_latest_oft_dir()
csv_path = os.path.join(latest_dir, 'results_database.csv')

df = pd.read_csv(csv_path).dropna()

TARGET_STRENGTH = 0.40
TARGET_GUIDANCE = 6.5
df_filtered = df[(df['Strength'] == TARGET_STRENGTH) & (df['Guidance'] == TARGET_GUIDANCE)].copy()

if len(df_filtered) == 0:
    print(f"Warning: Target {TARGET_STRENGTH}/{TARGET_GUIDANCE} not found. Auto-selecting first available combination...")
    TARGET_STRENGTH = df['Strength'].iloc[0]
    TARGET_GUIDANCE = df['Guidance'].iloc[0]
    df_filtered = df[(df['Strength'] == TARGET_STRENGTH) & (df['Guidance'] == TARGET_GUIDANCE)].copy()

features = ['L1', 'L2', 'MS_SSIM', 'LPIPS', 'Max_Patch']
X = df_filtered[features].values
y = df_filtered['Label'].values

# العزل الصارم: 70% للتدريب و 30% للاختبار الأعمى
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=999, stratify=y)

print(f"Total Images: {len(X)}")
print(f"Training Set: {len(X_train)} images")
print(f"Unseen Test Set: {len(X_test)} images\n")

print("1. Performing Cross-Validation on Training Set (to verify stability)...")
kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=999)
fold_aucs = []

for fold, (train_idx, val_idx) in enumerate(kf.split(X_train, y_train)):
    X_f_train, y_f_train = X_train[train_idx], y_train[train_idx]
    X_f_val, y_f_val = X_train[val_idx], y_train[val_idx]
    
    model = xgb.XGBClassifier(objective='binary:logistic', eval_metric='logloss', n_estimators=50, learning_rate=0.05, max_depth=2, subsample=0.8, colsample_bytree=0.8, random_state=999)
    model.fit(X_f_train, y_f_train)
    
    val_preds = model.predict_proba(X_f_val)[:, 1]
    fold_auc = roc_auc_score(y_f_val, val_preds)
    fold_aucs.append(fold_auc)
    print(f"Fold {fold+1} Validation AUC: {fold_auc:.4f}")

print(f"Mean Validation AUC: {np.mean(fold_aucs):.4f}\n")

print("2. Training Final Model on FULL Training Set and extracting Threshold...")
final_model = xgb.XGBClassifier(objective='binary:logistic', eval_metric='logloss', n_estimators=50, learning_rate=0.05, max_depth=2, random_state=999)
final_model.fit(X_train, y_train)

# استخراج العتبة من بيانات التدريب فقط لمنع الانحياز
train_preds_proba = final_model.predict_proba(X_train)[:, 1]
fpr, tpr, thresholds = roc_curve(y_train, train_preds_proba)
optimal_threshold = thresholds[np.argmax(tpr - fpr)]
print(f"Optimal Threshold calculated from Training Data: {optimal_threshold:.4f}\n")

print("3. Evaluating Final Model on UNSEEN Test Set (True Performance)...")
test_preds_proba = final_model.predict_proba(X_test)[:, 1]
test_preds_classes = (test_preds_proba >= optimal_threshold).astype(int)

test_auc = roc_auc_score(y_test, test_preds_proba)
precision = precision_score(y_test, test_preds_classes, zero_division=0)
recall = recall_score(y_test, test_preds_classes, zero_division=0)
f1 = f1_score(y_test, test_preds_classes, zero_division=0)
accuracy = accuracy_score(y_test, test_preds_classes)

print("="*40)
print("--- SCIENTIFIC TEST SET RESULTS (OFT) ---")
print(f"Overall AUC:     {test_auc:.4f}")
print(f"Strict F1-Score: {f1:.4f}")
print(f"Strict Precision:{precision:.4f}")
print(f"Strict Recall:   {recall:.4f}")
print(f"Strict Accuracy: {accuracy:.4f}")
print("="*40 + "\n")

model_path = os.path.join(latest_dir, f'xgboost_hybrid_S{TARGET_STRENGTH}_G{TARGET_GUIDANCE}.json')
metadata_path = os.path.join(latest_dir, f'xgboost_metadata_S{TARGET_STRENGTH}_G{TARGET_GUIDANCE}.json')

final_model.save_model(model_path)
with open(metadata_path, 'w') as f: json.dump({'optimal_threshold': float(optimal_threshold)}, f)

print(f"Scientifically valid model saved to: {model_path}")