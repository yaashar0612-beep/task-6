# Task 6: K-Nearest Neighbors (KNN) Classification

AI & ML Internship — Elevate Labs

## Objective
Understand and implement KNN for a classification problem using the Iris dataset.

## Tools
Python, Pandas, Scikit-learn, Matplotlib

## Files
```
knn_classification.py           # Main script
Iris.csv                        # Dataset (150 rows, 3 balanced classes)
outputs/
  accuracy_vs_k.png             # Test accuracy & CV accuracy across K = 1..25
  confusion_matrix.png          # Confusion matrix at the chosen K
  decision_boundary.png         # Decision boundary at the chosen K (2 features)
  decision_boundary_multi_k.png # How the boundary changes for K = 1, 5, chosen K, 25
  evaluation_report.txt         # Text version of the metrics below
```

## What the script does
1. **Loads** `Iris.csv` and drops the `Id` column.
2. **Encodes** the `Species` labels and **splits** the data 80/20 (stratified, so all
   three classes are represented proportionally in train and test).
3. **Normalizes** the 4 numeric features with `StandardScaler` (fit on the training
   set only, then applied to the test set — this avoids data leakage).
4. **Experiments with K = 1 to 25.** For each K it records:
   - the accuracy on the held-out test set, and
   - the mean accuracy from 5-fold cross-validation on the training set.
   The final K is chosen from the **cross-validation** curve rather than the raw
   test-set curve, because picking K purely to maximize one test split can
   reward a K that just got lucky (K=1 is especially prone to this).
5. **Evaluates** the final model with accuracy, a confusion matrix, and a full
   precision/recall/F1 classification report.
6. **Visualizes decision boundaries** using Petal Length & Petal Width (the two
   most discriminative features for Iris), including a side-by-side of K=1
   (jagged, overfit-prone), K=5, the chosen K, and K=25 (smoother, more biased)
   so the underfitting/overfitting trade-off is visible directly.

## Results (this run)
- **Chosen K:** 5 (via 5-fold CV, CV accuracy ≈ 96.7%)
- **Test accuracy:** ≈ 93.3% (28/30 correct)
- All errors were between **Iris-versicolor** and **Iris-virginica** — the two
  classes that overlap in feature space; **Iris-setosa** is perfectly separated
  from the other two in every run, which matches the well-known structure of
  this dataset.

Exact numbers can vary slightly with the random train/test split seed, but the
overall pattern (Setosa easy, Versicolor/Virginica occasionally confused) is
consistent.

## Interview Questions

**1. How does the KNN algorithm work?**
KNN is an instance-based ("lazy") learner: it stores the entire training set and
does no explicit training/fitting step. To classify a new point, it computes the
distance from that point to every training point, picks the K closest ones,
and assigns the class that is most common (majority vote) among those K
neighbors.

**2. How do you choose the right K?**
Try a range of K values and pick the one that gives the best validation/
cross-validation performance rather than test-set performance directly. Small
K (like 1–2) fits noise and overfits; large K oversmooths and can underfit,
washing out real class boundaries. A common heuristic is to start around
K = √n and use odd values for binary problems to avoid tie votes, then tune
with cross-validation, as this script does.

**3. Why is normalization important in KNN?**
KNN classifies based on distance between points. If one feature has a much
larger numeric range than another (e.g., a feature in the thousands vs. one
between 0 and 1), it will dominate the distance calculation regardless of its
actual relevance. Scaling all features to a comparable range (e.g., with
StandardScaler or Min-Max scaling) ensures every feature contributes fairly.

**4. What is the time complexity of KNN?**
- **Training:** O(1) — it just stores the data.
- **Prediction (naive/brute-force):** O(n·d) per query, where n is the number
  of training points and d is the number of features, since it must compute
  the distance to every stored point.
- This can be sped up with spatial index structures like KD-Trees or Ball
  Trees, which reduce average query time to roughly O(log n) for
  low-to-moderate dimensional data (their benefit shrinks in very
  high-dimensional spaces).

**5. What are pros and cons of KNN?**
- *Pros:* simple and intuitive; no training phase; naturally handles
  multi-class problems; makes no assumption about the underlying data
  distribution (non-parametric); decision boundaries can be arbitrarily
  complex.
- *Cons:* slow at prediction time on large datasets; sensitive to feature
  scale and irrelevant features; sensitive to noisy data/outliers (especially
  at small K); performance degrades in high dimensions (the "curse of
  dimensionality," since distances become less meaningful); needs the entire
  training set kept in memory.

**6. Is KNN sensitive to noise?**
Yes, particularly at small K. With K=1, a single mislabeled or noisy training
point can directly flip the prediction for any new point that lands near it.
Increasing K averages over more neighbors, which dilutes the influence of any
one noisy point, at the cost of potentially blurring genuine boundaries if K
is too large.

**7. How does KNN handle multi-class problems?**
Natively and without modification — the majority-vote step simply counts votes
across however many classes exist and picks the class with the most votes
among the K neighbors (ties can be broken e.g. by falling back to the nearest
single neighbor). This script uses all 3 Iris classes directly with
`KNeighborsClassifier`, no one-vs-rest wrapping needed.

**8. What's the role of distance metrics in KNN?**
The distance metric defines what "nearest" means, so it directly determines
which points are considered neighbors and therefore the predicted class.
Euclidean distance (straight-line distance, the default) works well for
continuous, similarly-scaled features. Manhattan distance can be more robust
in high dimensions or with grid-like data; Minkowski generalizes both; and
metrics like Hamming distance are used for categorical/binary features.
Choosing an inappropriate metric for the data type can significantly hurt
accuracy even with an otherwise well-tuned K.

## How to run
```bash
pip install pandas scikit-learn matplotlib
python knn_classification.py
```
