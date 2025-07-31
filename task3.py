import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns

# 1. Load Dataset
df = pd.read_csv("C:/Users/Rahul/Documents/New folder (3)/bank-full.csv", sep=';')  # Use correct path

# 2. Explore
print(df.head())
print("\nMissing values:\n", df.isnull().sum())
print("\nTarget class distribution:\n", df['y'].value_counts())

# 3. Encode categorical variables
df_encoded = pd.get_dummies(df, drop_first=True)

# 4. Split features and target
X = df_encoded.drop('y_yes', axis=1)
y = df_encoded['y_yes']

# 5. Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 6. Train Decision Tree
model = DecisionTreeClassifier(random_state=42, max_depth=5)  # You can tune max_depth
model.fit(X_train, y_train)

# 7. Predict and Evaluate
y_pred = model.predict(X_test)
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))
print("Accuracy:", accuracy_score(y_test, y_pred))

# 8. Visualize Decision Tree
plt.figure(figsize=(20,10))
plot_tree(model, feature_names=X.columns, class_names=["No", "Yes"], filled=True, fontsize=10)
plt.title("Decision Tree Classifier for Bank Marketing")
plt.show()
