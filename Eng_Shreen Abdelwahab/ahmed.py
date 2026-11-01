
# Machine Learning - Lecture 1
# Concepts and Evaluation Metrics

# =====================================
# 1. Supervised Learning
# =====================================

# Classification Example
actual_labels = ["Spam", "Not Spam", "Spam", "Not Spam"]
predicted_labels = ["Spam", "Not Spam", "Not Spam", "Not Spam"]

print("Supervised Learning")
print("Actual:", actual_labels)
print("Predicted:", predicted_labels)


# =====================================
# 2. Unsupervised Learning
# =====================================

from sklearn.cluster import KMeans
import numpy as np

# Example data
X = np.array([
    [1, 2],
    [2, 3],
    [8, 9],
    [9, 10]
])

model = KMeans(n_clusters=2, random_state=42, n_init=10)
model.fit(X)

print("\nUnsupervised Learning")
print("Cluster Labels:", model.labels_)


# =====================================
# 3. Reinforcement Learning
# =====================================

# Simple reward example
rewards = [10, -5, 20, 15]

total_reward = sum(rewards)

print("\nReinforcement Learning")
print("Total Reward:", total_reward)


# =====================================
# 4. Confusion Matrix Values
# From the lecture example
# =====================================

TP = 90
FN = 10
FP = 5
TN = 895

print("\nConfusion Matrix")
print("True Positive:", TP)
print("False Negative:", FN)
print("False Positive:", FP)
print("True Negative:", TN)


# =====================================
# 5. Classification Metrics
# =====================================

total = TP + TN + FP + FN

accuracy = (TP + TN) / total
precision = TP / (TP + FP)
recall = TP / (TP + FN)
f1_score = 2 * precision * recall / (precision + recall)

print("\nClassification Metrics")
print(f"Accuracy:  {accuracy * 100:.1f}%")
print(f"Precision: {precision * 100:.1f}%")
print(f"Recall:    {recall * 100:.1f}%")
print(f"F1 Score:  {f1_score * 100:.1f}%")


# =====================================
# 6. Regression Metrics
# =====================================

actual = np.array([100, 200, 300, 400])
predicted = np.array([110, 190, 320, 380])

errors = actual - predicted

MSE = np.mean(errors ** 2)
RMSE = np.sqrt(MSE)
MAE = np.mean(np.abs(errors))

print("\nRegression Metrics")
print("MSE:", MSE)
print("RMSE:", RMSE)
print("MAE:", MAE)






# Machine Learning - Lecture 2
# Linear Regression: Concepts & Worked Example

import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# ==========================================
# 1. Dataset from the lecture
# ==========================================
X = np.array([2, 4, 6, 8, 10], dtype=float)  # Study hours
y = np.array([40, 50, 65, 80, 90], dtype=float)  # Exam grades

n = len(X)
x_mean = np.mean(X)
y_mean = np.mean(y)

print("STEP 1: Means")
print("Mean of X:", x_mean)
print("Mean of Y:", y_mean)

# ==========================================
# 2. Calculate slope and intercept manually
# ==========================================
numerator = np.sum((X - x_mean) * (y - y_mean))
denominator = np.sum((X - x_mean) ** 2)

slope = numerator / denominator       # beta_1
intercept = y_mean - slope * x_mean   # beta_0

print("\nSTEP 2: Slope and Intercept")
print("Numerator:", numerator)
print("Denominator:", denominator)
print("Slope (beta_1):", slope)
print("Intercept (beta_0):", intercept)
print(f"Regression equation: y_hat = {intercept} + {slope}x")

# ==========================================
# 3. Predictions and residuals
# ==========================================
y_pred = intercept + slope * X
residuals = y - y_pred

print("\nSTEP 3: Predictions and Residuals")
print("Hours | Actual | Predicted | Residual")
for hours, actual, predicted, error in zip(X, y, y_pred, residuals):
    print(f"{hours:5.0f} | {actual:6.1f} | {predicted:9.1f} | {error:8.1f}")

# ==========================================
# 4. Regression evaluation metrics
# ==========================================
mae = mean_absolute_error(y, y_pred)
mse = mean_squared_error(y, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y, y_pred)

print("\nSTEP 4: Evaluation Metrics")
print(f"MAE:  {mae:.2f}")
print(f"MSE:  {mse:.2f}")
print(f"RMSE: {rmse:.2f}")
print(f"R^2:  {r2:.4f} ({r2 * 100:.2f}%)")

# ==========================================
# 5. Predict a grade for 7 study hours
# ==========================================
hours_to_predict = 7
predicted_grade = intercept + slope * hours_to_predict
print(f"\nPrediction for {hours_to_predict} study hours: {predicted_grade:.1f}")

# ==========================================
# 6. Verify using scikit-learn
# ==========================================
model = LinearRegression()
model.fit(X.reshape(-1, 1), y)

print("\nSTEP 5: scikit-learn verification")
print("Intercept:", model.intercept_)
print("Slope:", model.coef_[0])
print("Prediction for 7 hours:", model.predict([[7]])[0])




/__codex_browser_temporary_files__/450136bf-0dc0-4fad-b701-3c0cd030ed3b/lecture3_kmeans.py






import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

# 1. تحميل بيانات تجريبية (أو يمكن استبدالها ببياناتك الخاصة مثل جدول الطلاب)
data = load_iris()
X, y = data.data, data.target

# 2. تقسيم البيانات إلى بيانات تدريب (Train) وبيانات اختبار (Test) بنسبة 70% إلى 30%
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# 3. بناء وتدريب نموذج شجرة القرار (Decision Tree) مع تحديد أقصى عمق لتجنب الإفراط في التعلّم (Overfitting)
tree_model = DecisionTreeClassifier(max_depth=3, random_state=42)
tree_model.fit(X_train, y_train)

# 4. بناء وتدريب نموذج الغابة العشوائية (Random Forest) مع تحديد عدد الأشجار (n_estimators)
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

# 5. قياس دقة النماذج على بيانات الاختبار
tree_accuracy = tree_model.score(X_test, y_test)
rf_accuracy = rf_model.score(X_test, y_test)

print(f"Decision Tree Accuracy: {tree_accuracy:.2f}")
print(f"Random Forest Accuracy: {rf_accuracy:.2f}")

# 6. التنبؤ بنتيجة عينة جديدة باستخدام الغابة العشوائية واحتمالات التصويت (Soft Voting)
sample_prediction = rf_model.predict_proba([X_test[0]])
print(f"Prediction Probabilities for the first test sample: {sample_prediction}")







from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline

# 1. بيانات التدريب (أمثلة لرسائل بريد إلكتروني وتصنيفاتها: spam أو ham)
emails = [
    "free offer now", 
    "meeting at noon", 
    "win free prize", 
    "project report"
]
labels = ["spam", "ham", "spam", "ham"]

# 2. بناء نموذج يدمج تحويل الكلمات إلى أعداد (CountVectorizer) مع مصنّف Naïve Bayes (MultinomialNB)
# مع استخدام Laplace smoothing الافتراضي (alpha = 1.0)
model = make_pipeline(
    CountVectorizer(), 
    MultinomialNB(alpha=1.0)
)

# 3. تدريب النموذج على البيانات
model.fit(emails, labels)

# 4. التنبؤ بتصنيف إيميل جديد يحتوي على كلمات جديدة
new_email = ["free prize offer"]
prediction = model.predict(new_email)
probabilities = model.predict_proba(new_email)

print(f"Predicted Class: {prediction[0]}")
print(f"Prediction Probabilities [Ham, Spam]: {probabilities}")