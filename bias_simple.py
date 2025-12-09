

import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import json
from datetime import datetime



# Create results folder if not exists
if not os.path.exists('results'):
    os.makedirs('results')
    print("✅ Results folder created")

# ==================== Part 1: Generate Data ====================
print("\n 1. Generating synthetic data...")

np.random.seed(42)
n = 500
healthy = np.random.normal(5, 1, n//2)
disease = np.random.normal(10, 1, n//2)
X = np.concatenate([healthy, disease])
y = np.concatenate([np.zeros(n//2), np.ones(n//2)])

print(f"✅ {n} samples generated")

# ==================== Part 2: Add Bias ====================
print("\n 2. Adding synthetic bias...")

bias = np.zeros(n)
for i in range(n):
    if y[i] == 1:  # If patient
        if np.random.random() < 0.8:  # 80% chance
            bias[i] = 1
    else:  # If healthy
        if np.random.random() < 0.2:  # 20% chance
            bias[i] = 1

print(f" Bias added")

# ==================== Part 3: Train Models ====================
print("\n 3. Training machine learning models...")

X_combined = np.column_stack([X, bias])
X_train, X_test, y_train, y_test = train_test_split(
    X_combined, y, test_size=0.2, random_state=42
)

model_no_bias = LogisticRegression()
model_no_bias.fit(X_train[:, 0].reshape(-1, 1), y_train)
y_pred_no_bias = model_no_bias.predict(X_test[:, 0].reshape(-1, 1))
acc_no_bias = accuracy_score(y_test, y_pred_no_bias)

model_with_bias = LogisticRegression()
model_with_bias.fit(X_train, y_train)
y_pred_with_bias = model_with_bias.predict(X_test)
acc_with_bias = accuracy_score(y_test, y_pred_with_bias)

print(f" Models trained")

# ==================== Part 4: Analyze ====================
print("\n 4. Analyzing model...")

weights = model_with_bias.coef_[0]
bias_weight = model_with_bias.intercept_[0]

total = abs(weights[0]) + abs(weights[1])
disease_importance = (abs(weights[0]) / total) * 100
bias_importance = (abs(weights[1]) / total) * 100

print(f"   Disease weight: {weights[0]:.4f}")
print(f"   Bias weight: {weights[1]:.4f}")
print(f"   Bias importance: {bias_importance:.1f}%")

# ==================== Part 5: Create Plots ====================
print("\n 5. Creating plots...")

plt.figure(figsize=(15, 5))

# Plot 1: Weights
plt.subplot(1, 3, 1)
plt.bar(['Disease', 'Bias'], weights, color=['blue', 'orange'])
plt.title('Model Weights')
plt.ylabel('Value')

# Plot 2: Importance
plt.subplot(1, 3, 2)
plt.pie([disease_importance, bias_importance], 
        labels=['Disease', 'Bias'], 
        colors=['lightblue', 'lightcoral'],
        autopct='%1.1f%%')
plt.title('Feature Importance')

# Plot 3: Accuracy
plt.subplot(1, 3, 3)
plt.bar(['No Bias', 'With Bias'], [acc_no_bias, acc_with_bias], 
        color=['green', 'orange'])
plt.title('Accuracy Comparison')
plt.ylabel('Accuracy')
plt.ylim([0, 1])

plt.tight_layout()
plt.savefig('results/results_simple.png', dpi=100)
plt.show()

# ==================== Part 6: Save Results ====================
print("\n 6. Saving results...")

results = {
    'timestamp': datetime.now().isoformat(),
    'weights': {
        'disease': float(weights[0]),
        'bias': float(weights[1]),
        'intercept': float(bias_weight)
    },
    'importance': {
        'disease_percent': float(disease_importance),
        'bias_percent': float(bias_importance)
    },
    'accuracy': {
        'without_bias': float(acc_no_bias),
        'with_bias': float(acc_with_bias),
        'difference': float(acc_with_bias - acc_no_bias)
    }
}

# Save JSON
with open('results/results_simple.json', 'w', encoding='utf-8') as f:
    json.dump(results, f, ensure_ascii=False, indent=2)

# Text report
report = f""" Project Report
{'='*40}
Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

Results:
• Accuracy with bias: {acc_with_bias:.2%}
• Accuracy without bias: {acc_no_bias:.2%}
• Improvement: {(acc_with_bias-acc_no_bias):+.2%}
• Bias importance: {bias_importance:.1f}%

Analysis:"""
if bias_importance > 30:
    report += "\n Model strongly uses bias (Shortcut Learning detected!)"
elif bias_importance > 15:
    report += "\n  Model moderately uses bias"
else:
    report += "\n Model focuses on actual disease feature"

with open('results/analysis_report.txt', 'w', encoding='utf-8') as f:
    f.write(report)

# ==================== Part 7: Conclusion ====================
print("\n" + "="*60)
print(" Project completed successfully!")
print("="*60)

print("\n Files created in results/ folder:")
print("• results_simple.png  ← Charts")
print("• results_simple.json ← Numerical results")
print("• analysis_report.txt ← Report")

print(f"\n Final result:")
print(f"• Bias importance: {bias_importance:.1f}%")
if bias_importance > 30:
    print("•  Article concept proven: Model learned bias as shortcut!")
else:
    print("•   Bias had little influence")

