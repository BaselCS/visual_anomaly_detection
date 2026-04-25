import os
import pandas as pd
import numpy as np
import joblib
from sklearn.ensemble import IsolationForest
from sklearn.metrics import roc_auc_score, roc_curve, precision_score, recall_score, f1_score, accuracy_score
from sklearn.model_selection import train_test_split
import warnings

warnings.filterwarnings('ignore')

def get_latest_oft_dir(base_dir="trained_models"):
    max_num = 0
    latest_folder = None
    for folder_name in os.listdir(base_dir):
        if folder_name.startswith("train_oft_"):
            try:
                num = int(folder_name.replace("train_oft_", ""))
                if num > max_num: max_num = num; latest_folder = folder_name
            except ValueError: continue
    return os.path.join(base_dir, latest_folder)

print("--- Starting SCIENTIFICALLY PURE One-Class Model Training ---")
latest_dir = get_latest_oft_dir()
csv_path = os.path.join(latest_dir, 'pure_results_database.csv')

df = pd.read_csv(csv_path).dropna()
features = ['L1', 'L2', 'MS_SSIM', 'LPIPS', 'Max_Patch']

# 1. البيانات النقية للتدريب (سليم فقط)
df_train = df[df['Split'] == 'Train']
X_train = df_train[features].values

# 2. جلب جميع بيانات الاختبار
df_test_full = df[df['Split'] == 'Test']

# 🔥 الحل الجذري (Zero Leakage): تقسيم الاختبار إلى (معايرة) و (اختبار نهائي)
# stratify: تضمن توزيع العيوب والسليم بالتساوي بين المجموعتين
df_val, df_test_strict = train_test_split(
    df_test_full, test_size=0.50, stratify=df_test_full['Label'], random_state=42
)

X_val = df_val[features].values
y_val = df_val['Label'].values

X_test = df_test_strict[features].values
y_test = df_test_strict['Label'].values

print(f"1. Training Set:   {len(X_train)} pure good images")
print(f"2. Validation Set: {len(X_val)} mixed images (Used ONLY to find Threshold)")
print(f"3. Strict Test Set:{len(X_test)} mixed images (Never seen before)\n")

print("[Phase 1] Training Isolation Forest on strictly Train images...")
model = IsolationForest(n_estimators=200, contamination='auto', random_state=999)
model.fit(X_train)

# ==========================================
# مرحلة المعايرة (Calibration) - استخراج العتبة
# ==========================================
print("[Phase 2] Calibrating Threshold using Validation Set...")
val_scores = -model.decision_function(X_val)
fpr, tpr, thresholds = roc_curve(y_val, val_scores)
optimal_idx = np.argmax(tpr - fpr)
optimal_threshold = thresholds[optimal_idx]
print(f"          > Locked Optimal Threshold: {optimal_threshold:.4f}\n")

# ==========================================
# مرحلة الاختبار النهائي (Final Evaluation)
# ==========================================
print("[Phase 3] Evaluating Model on STRICT UNSEEN Test Set...")
test_scores = -model.decision_function(X_test)
test_auc = roc_auc_score(y_test, test_scores)

# نستخدم العتبة التي حفظناها من خطوة المعايرة
optimal_classes = (test_scores >= optimal_threshold).astype(int)

opt_precision = precision_score(y_test, optimal_classes, zero_division=0)
opt_recall = recall_score(y_test, optimal_classes, zero_division=0)
opt_f1 = f1_score(y_test, optimal_classes, zero_division=0)
opt_accuracy = accuracy_score(y_test, optimal_classes)

print("="*45)
print("--- 100% BULLETPROOF SCIENTIFIC RESULTS ---")
print(f"Strict Test AUC:       {test_auc:.4f}")
print(f"Strict Test F1-Score:  {opt_f1:.4f}")
print(f"Strict Test Precision: {opt_precision:.4f}")
print(f"Strict Test Recall:    {opt_recall:.4f}")
print(f"Strict Test Accuracy:  {opt_accuracy:.4f}")
print("="*45 + "\n")

model_path = os.path.join(latest_dir, 'iforest_bulletproof_model.pkl')
joblib.dump(model, model_path)
print(f"Bulletproof One-Class model saved to: {model_path}")