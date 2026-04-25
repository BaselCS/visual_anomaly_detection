import os
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve, auc, confusion_matrix

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

latest_dir = get_latest_oft_dir()
csv_path = os.path.join(latest_dir, 'pure_results_database.csv')
model_path = os.path.join(latest_dir, 'iforest_pure_model.pkl')
output_dir = "report_charts"
os.makedirs(output_dir, exist_ok=True)

# 1. تجهيز البيانات
df = pd.read_csv(csv_path).dropna()
features = ['L1', 'L2', 'MS_SSIM', 'LPIPS', 'Max_Patch']
df_test = df[df['Split'] == 'Test']
X_test = df_test[features].values
y_test = df_test['Label'].values

# 2. تحميل النموذج وحساب التوقعات
model = joblib.load(model_path)
preds_scores = -model.decision_function(X_test)

fpr, tpr, thresholds = roc_curve(y_test, preds_scores)
roc_auc = auc(fpr, tpr)
optimal_idx = np.argmax(tpr - fpr)
optimal_threshold = thresholds[optimal_idx]
optimal_classes = (preds_scores >= optimal_threshold).astype(int)

# ==========================================
# الرسم البياني الأول: منحنى ROC
# ==========================================
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.4f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.scatter(fpr[optimal_idx], tpr[optimal_idx], marker='o', color='red', s=100, label=f'Optimal Threshold')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate', fontsize=12)
plt.ylabel('True Positive Rate', fontsize=12)
plt.title('Receiver Operating Characteristic (ROC) - Pure Isolation Forest', fontsize=14)
plt.legend(loc="lower right")
plt.grid(alpha=0.3)
plt.savefig(os.path.join(output_dir, 'roc_curve.png'), dpi=300, bbox_inches='tight')
plt.close()

# ==========================================
# الرسم البياني الثاني: مصفوفة الارتباك
# ==========================================
cm = confusion_matrix(y_test, optimal_classes)
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False, 
            xticklabels=['Good', 'Anomaly'], yticklabels=['Good', 'Anomaly'],
            annot_kws={"size": 16})
plt.xlabel('Predicted Label', fontsize=12)
plt.ylabel('True Label', fontsize=12)
plt.title('Confusion Matrix (Optimal Threshold)', fontsize=14)
plt.savefig(os.path.join(output_dir, 'confusion_matrix.png'), dpi=300, bbox_inches='tight')
plt.close()

# ==========================================
# الرسم البياني الثالث: مقارنة التقنيات
# ==========================================
techniques = [
    'OFT + XGBoost\n(Supervised)', 
    'DoRA + XGBoost', 
    'DoRA + ControlNet + XGBoost',
    'OFT + iForest\n(Pure Zero-Shot)',
    'Textual Inversion', 
    'TE LoRA'
]
accuracies = [98.10, 93.98, 92.77, 89.16, 87.95, 75.90]

plt.figure(figsize=(10, 6))
bars = plt.barh(techniques[::-1], accuracies[::-1], color=sns.color_palette("viridis", len(techniques)))
plt.xlabel('Accuracy (%)', fontsize=12)
plt.title('QASSAS Project: Anomaly Detection Performance Comparison', fontsize=14)
plt.xlim([60, 100])

for bar in bars:
    plt.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height()/2, 
             f'{bar.get_width():.2f}%', va='center', fontsize=11, fontweight='bold')

plt.grid(axis='x', alpha=0.3)
plt.savefig(os.path.join(output_dir, 'models_comparison.png'), dpi=300, bbox_inches='tight')
plt.close()

print(f"Charts generated successfully in the '{output_dir}' folder!")