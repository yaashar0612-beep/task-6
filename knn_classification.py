"""
Task 6: K-Nearest Neighbors (KNN) Classification
AI & ML Internship - Elevate Labs

Dataset : Iris Dataset (Iris.csv)
Steps   :
    1. Load data & normalize features
    2. Train/test split
    3. Fit KNeighborsClassifier, experiment with different K values
    4. Evaluate using accuracy & confusion matrix
    5. Visualize decision boundaries
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay, classification_report

OUT = "outputs"

# ---------------------------------------------------------------------------
# 1. Load the dataset
# ---------------------------------------------------------------------------
df = pd.read_csv("Iris.csv")
df = df.drop(columns=["Id"])  # Id column is not a feature

print("First 5 rows:\n", df.head())
print("\nClass distribution:\n", df["Species"].value_counts())

feature_cols = ["SepalLengthCm", "SepalWidthCm", "PetalLengthCm", "PetalWidthCm"]
X = df[feature_cols].values
y_raw = df["Species"].values

le = LabelEncoder()
y = le.fit_transform(y_raw)          # Iris-setosa=0, Iris-versicolor=1, Iris-virginica=2
class_names = le.classes_

# ---------------------------------------------------------------------------
# 2. Train/test split + Normalize features (fit scaler on train only)
# ---------------------------------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ---------------------------------------------------------------------------
# 3. Experiment with different values of K
#    Test-set accuracy is shown for reference, but K is CHOSEN using 5-fold
#    cross-validation on the training set only - this avoids picking a K
#    (e.g. K=1) that just happens to fit this particular test split and is
#    more likely to overfit / be sensitive to noise.
# ---------------------------------------------------------------------------
k_values = range(1, 26)
test_accuracies = []
cv_accuracies = []

for k in k_values:
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train_scaled, y_train)
    test_accuracies.append(accuracy_score(y_test, knn.predict(X_test_scaled)))
    cv_scores = cross_val_score(KNeighborsClassifier(n_neighbors=k), X_train_scaled, y_train, cv=5)
    cv_accuracies.append(cv_scores.mean())

best_k = k_values[int(np.argmax(cv_accuracies))]
print(f"\nBest K (via 5-fold CV on training data) = {best_k} "
      f"(CV accuracy = {max(cv_accuracies):.4f}, test accuracy = {test_accuracies[best_k - 1]:.4f})")

plt.figure(figsize=(8, 5))
plt.plot(list(k_values), test_accuracies, marker="o", label="Test accuracy", color="#4C72B0")
plt.plot(list(k_values), cv_accuracies, marker="s", label="5-fold CV accuracy (train)", color="#DD8452")
plt.axvline(best_k, color="red", linestyle="--", alpha=0.6, label=f"Chosen K = {best_k}")
plt.title("KNN Accuracy vs. K value (Iris dataset)")
plt.xlabel("Number of Neighbors (K)")
plt.ylabel("Accuracy")
plt.xticks(list(k_values))
plt.grid(alpha=0.3)
plt.legend()
plt.tight_layout()
plt.savefig(f"{OUT}/accuracy_vs_k.png", dpi=150)
plt.close()

# ---------------------------------------------------------------------------
# 4. Final model evaluation (accuracy + confusion matrix) at best K
# ---------------------------------------------------------------------------
final_knn = KNeighborsClassifier(n_neighbors=best_k)
final_knn.fit(X_train_scaled, y_train)
y_pred = final_knn.predict(X_test_scaled)

final_acc = accuracy_score(y_test, y_pred)
cm = confusion_matrix(y_test, y_pred)
report = classification_report(y_test, y_pred, target_names=class_names)

print(f"\nFinal Accuracy (K={best_k}): {final_acc:.4f}")
print("\nConfusion Matrix:\n", cm)
print("\nClassification Report:\n", report)

with open(f"{OUT}/evaluation_report.txt", "w") as f:
    f.write(f"Best K: {best_k}\n")
    f.write(f"Final Test Accuracy: {final_acc:.4f}\n\n")
    f.write("Confusion Matrix:\n")
    f.write(np.array2string(cm))
    f.write("\n\nClassification Report:\n")
    f.write(report)

disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
fig, ax = plt.subplots(figsize=(6, 5))
disp.plot(ax=ax, cmap="Blues", colorbar=True)
plt.title(f"Confusion Matrix (K={best_k}, Accuracy={final_acc:.2%})")
plt.tight_layout()
plt.savefig(f"{OUT}/confusion_matrix.png", dpi=150)
plt.close()

# ---------------------------------------------------------------------------
# 5. Visualize decision boundaries
#    (using Petal Length & Petal Width -> the 2 most discriminative features)
# ---------------------------------------------------------------------------
feat_a, feat_b = "PetalLengthCm", "PetalWidthCm"
X2 = df[[feat_a, feat_b]].values
X2_train, X2_test, y2_train, y2_test = train_test_split(
    X2, y, test_size=0.2, random_state=42, stratify=y
)

scaler2 = StandardScaler()
X2_train_s = scaler2.fit_transform(X2_train)
X2_test_s = scaler2.transform(X2_test)

def plot_decision_boundary(ax, model, X, y, title):
    cmap_light = ListedColormap(["#FFDDDD", "#DDFFDD", "#DDDDFF"])
    cmap_bold = ["#FF0000", "#00AA00", "#0000FF"]

    x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
    y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 300), np.linspace(y_min, y_max, 300))

    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    ax.contourf(xx, yy, Z, cmap=cmap_light, alpha=0.8)
    for i, color in enumerate(cmap_bold):
        idx = y == i
        ax.scatter(X[idx, 0], X[idx, 1], c=color, label=class_names[i],
                   edgecolor="k", s=35)
    ax.set_xlabel(f"{feat_a} (scaled)")
    ax.set_ylabel(f"{feat_b} (scaled)")
    ax.set_title(title)
    ax.legend(fontsize=8)

# Single plot at best_k
fig, ax = plt.subplots(figsize=(7, 6))
knn2 = KNeighborsClassifier(n_neighbors=best_k)
knn2.fit(X2_train_s, y2_train)
plot_decision_boundary(ax, knn2, X2_train_s, y2_train, f"KNN Decision Boundary (K={best_k})")
plt.tight_layout()
plt.savefig(f"{OUT}/decision_boundary.png", dpi=150)
plt.close()

# Comparison grid: effect of K on the boundary (underfitting vs overfitting)
fig, axes = plt.subplots(2, 2, figsize=(12, 10))
for ax, k in zip(axes.ravel(), [1, 5, best_k, 25]):
    knn_k = KNeighborsClassifier(n_neighbors=k)
    knn_k.fit(X2_train_s, y2_train)
    plot_decision_boundary(ax, knn_k, X2_train_s, y2_train, f"K={k}")
plt.suptitle("How K changes the KNN decision boundary", fontsize=14)
plt.tight_layout()
plt.savefig(f"{OUT}/decision_boundary_multi_k.png", dpi=150)
plt.close()

print("\nAll outputs saved to the 'outputs/' folder.")
