import pandas as pd

df = pd.read_csv('Crop_recommendation.csv')
df
df.info()
df.head()
df.describe()

#Cleaning data
df = df.drop_duplicates()
df
df.isnull().sum()
df = df.rename(columns={'N': 'Nitrogen', 'P': 'Phosphorus', 'K':'Potassium'})
df
df['label'].unique()
df['label'].nunique()
count_labels = df['label'].value_counts()
count_labels

#Visulizing data
import matplotlib.pyplot as plt
import seaborn as sns

Title_font = {'family': 'serif', 'color': 'black', 'size': 14 , 'weight' : 'bold'}
Label_font = {'family': 'serif', 'color': 'black', 'size': 12}
Values_font = {'family': 'serif', 'color': 'black', 'size': 8}

avg_features = df.groupby('label')[['Nitrogen', 'Phosphorus', 'Potassium', 'temperature', 'humidity', 'ph', 'rainfall']].mean()

plt.figure(figsize=(15, 10))
sns.heatmap(avg_features, annot=True, cmap='YlGnBu', fmt=".1f")
plt.title("Average Environmental Features per Crop", fontdict=Title_font)
plt.xlabel("Feature", fontdict=Label_font)
plt.ylabel("Crop", fontdict=Label_font)
plt.tight_layout()
plt.show()

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

Features = df.drop('label', axis=1)
Output = df['label']
le = LabelEncoder()
output_encoded = le.fit_transform(Output)

Features_train, Features_test, Output_train, Output_test = train_test_split(
    Features, output_encoded, test_size=0.3, random_state=50, stratify=output_encoded
)

import numpy as np
print(np.bincount(Output_train))
print(np.bincount(Output_test))

from sklearn.metrics import accuracy_score, classification_report
from sklearn.tree import DecisionTreeClassifier

model_dt = DecisionTreeClassifier(random_state = 50)
model_dt.fit(Features_train, Output_train)
output_pred_dt = model_dt.predict(Features_test)
accuracy_dt = accuracy_score(Output_test, output_pred_dt)
print("Decision Tree accuracy:", accuracy_dt)
print(classification_report(Output_test, output_pred_dt, target_names=le.classes_))

from sklearn.svm import SVC
model_svm = SVC(kernel='rbf', random_state=50)
model_svm.fit(Features_train, Output_train)
output_pred_svm = model_svm.predict(Features_test)
accuracy_svm = accuracy_score(Output_test, output_pred_svm)
print("SVM accuracy:", accuracy_svm)
print(classification_report(Output_test, output_pred_svm, target_names=le.classes_))

from sklearn.ensemble import RandomForestClassifier
model_rf = RandomForestClassifier(random_state=50)
model_rf.fit(Features_train, Output_train)
output_pred_rf = model_rf.predict(Features_test)
accuracy_rf = accuracy_score(Output_test, output_pred_rf)
print("Random Forest accuracy:", accuracy_rf)
print(classification_report(Output_test, output_pred_rf, target_names=le.classes_))

from xgboost import XGBClassifier
model_xgb = XGBClassifier(use_label_encoder=False, eval_metric='mlogloss', random_state=50)
model_xgb.fit(Features_train, Output_train)
output_pred_xgb = model_xgb.predict(Features_test)
accuracy_xgb = accuracy_score(Output_test, output_pred_xgb)
print("XGBoost accuracy:", accuracy_xgb)
print(classification_report(Output_test, output_pred_xgb, target_names=le.classes_))

from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

#DT
output_pred_dt = model_dt.predict(Features_test)
dt_cm = confusion_matrix(Output_test, output_pred_dt)
sns.heatmap(dt_cm, fmt="d", cmap="Reds", cbar=True)
plt.title(f"Confusion Matrix - SVM (Accuracy: {accuracy_dt:.3f})", fontdict=Title_font)
plt.xlabel("Predicted Label", fontdict = Label_font)
plt.ylabel("True Label", fontdict = Label_font)
plt.tight_layout()
plt.show()

#SVM
output_pred_svm = model_svm.predict(Features_test)
svm_cm = confusion_matrix(Output_test, output_pred_svm)
sns.heatmap(svm_cm, fmt="d", cmap="Reds", cbar=True)
plt.title(f"Confusion Matrix - SVM (Accuracy: {accuracy_svm:.3f})", fontdict=Title_font)
plt.xlabel("Predicted Label", fontdict = Label_font)
plt.ylabel("True Label", fontdict = Label_font)
plt.tight_layout()
plt.show()

#RF
rf_cm = confusion_matrix(Output_test, output_pred_rf)
sns.heatmap(rf_cm, fmt="d", cmap="Greens", cbar=True)
plt.title(f"Confusion Matrix - SVM (Accuracy: {accuracy_rf:.3f})", fontdict=Title_font)
plt.xlabel("Predicted Label", fontdict=Label_font)
plt.ylabel("True Label", fontdict=Label_font)
plt.tight_layout()
plt.show()

#XGB
xgb_cm = confusion_matrix(Output_test, output_pred_xgb)
sns.heatmap(xgb_cm, fmt="d", cmap="Reds", cbar=True)
plt.title(f"Confusion Matrix - SVM (Accuracy: {accuracy_xgb:.3f})", fontdict=Title_font)
plt.xlabel("Predicted Label", fontdict=Label_font)
plt.ylabel("True Label", fontdict=Label_font)
plt.tight_layout()
plt.show()

import joblib
joblib.dump(model_rf, "best_crop_model.pkl")
joblib.dump(le, "label_encoder.pkl")
print("Model and label encoder saved successfully!")