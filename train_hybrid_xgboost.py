import os
import pandas as pd
import numpy as np
import xgboost as xgb
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, roc_curve, precision_score, recall_score, f1_score, accuracy_score

def get_latest_train_folder(base_dir="trained_models"):
    if not os.path.exists(base_dir):
        raise FileNotFoundError(f"Directory {base_dir} does not exist.")
    max_num = 0
    latest_folder = None
    for folder_name in os.listdir(base_dir):
        if folder_name.startswith("train") and os.path.isdir(os.path.join(base_dir, folder_name)):
            try:
                num = int(folder_name.replace("train", ""))
                if num > max_num:
                    max_num = num
                    latest_folder = folder_name
            except ValueError:
                continue
    if latest_folder is None:
        raise FileNotFoundError(f"No train directories found in {base_dir}")
    return os.path.join(base_dir, latest_folder)

# ==========================================
# 1. إعداد المسارات وتحميل البيانات
# ==========================================
print("--- Starting Hybrid Model Training (Absolute Strict Mode) ---")
latest_dir = get_latest_train_folder()
csv_path = os.path.join(latest_dir, 'results_database.csv')

df = pd.read_csv(csv_path).dropna()

TARGET_STRENGTH = 0.42
TARGET_GUIDANCE = 5.5

df_filtered = df[(df['Strength'] == TARGET_STRENGTH) & (df['Guidance'] == TARGET_GUIDANCE)].copy()

print(f"Total Unique Images after filtering: {len(df_filtered)}")

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
    
    # نموذج مبسط جداً لمنع الإفراط في التخصيص لبيانات صغيرة
    model = xgb.XGBClassifier(
        objective='binary:logistic',
        eval_metric='logloss',
        n_estimators=50,        # قللنا الأشجار إلى 50
        learning_rate=0.05,
        max_depth=2,            # قللنا العمق ليكون أبسط
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=999
    )
    
    model.fit(X_train, y_train)
    
    # --- المعالجة الصارمة (Strict Handling) ---
    # 1. نتوقع على بيانات *التدريب* لنجد العتبة
    train_preds_proba = model.predict_proba(X_train)[:, 1]
    fpr_train, tpr_train, thresholds_train = roc_curve(y_train, train_preds_proba)
    best_idx_train = np.argmax(tpr_train - fpr_train)
    fold_threshold = thresholds_train[best_idx_train]
    
    # 2. نتوقع على بيانات *الاختبار* ونطبق العتبة العمياء المستخرجة من التدريب
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
precision = precision_score(y, oof_preds_classes)
recall = recall_score(y, oof_preds_classes)
f1 = f1_score(y, oof_preds_classes)
accuracy = accuracy_score(y, oof_preds_classes)

print("\n" + "="*40)
print("--- BULLETPROOF HYBRID MODEL RESULTS ---")
print(f"Target Config: Strength={TARGET_STRENGTH}, Guidance={TARGET_GUIDANCE}")
print(f"Mean Fold AUC: {np.mean(fold_aucs):.4f}")
print(f"Overall AUC:   {final_auc:.4f}")
print(f"Strict F1-Score: {f1:.4f}")
print(f"Strict Precision:{precision:.4f}")
print(f"Strict Recall:   {recall:.4f}")
print(f"Strict Accuracy: {accuracy:.4f}")
print("="*40 + "\n")

# ==========================================
# 4. حفظ الموديل النهائي
# ==========================================
final_model = xgb.XGBClassifier(objective='binary:logistic', n_estimators=50, learning_rate=0.05, max_depth=2, random_state=999)
final_model.fit(X, y)
model_path = os.path.join(latest_dir, f'xgboost_hybrid_S{TARGET_STRENGTH}_G{TARGET_GUIDANCE}.json')
final_model.save_model(model_path)
print(f"Final Model saved to: {model_path}")