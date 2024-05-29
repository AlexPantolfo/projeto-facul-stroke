from flask import Flask, request, jsonify
import pandas as pd
from sklearn import metrics
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, roc_auc_score, precision_score, recall_score, f1_score
from sklearn.model_selection import cross_val_score
from datetime import datetime

app = Flask(__name__)

# Load and preprocess the dataset
avc = pd.read_csv('healthcare-dataset-stroke-data-teste.csv')
avc = avc.drop(['id'], axis=1)
avc['bmi'].fillna(avc['bmi'].mean(), inplace=True)
avc = avc.replace({
    'gender': {'Male': 0, 'Female': 1, 'Other': 2},
    'ever_married': {'Yes': 0, 'No': 1},
    'work_type': {'Private': 0, 'Self-employed': 1, 'Govt_job': 2, 'children': 3, 'Never_worked': 4},
    'smoking_status': {'formerly smoked': 0, 'never smoked': 1, 'smokes': 2, 'Unknown': 3},
    'Residence_type': {'Urban': 0, 'Rural': 1}
})

# Oversampling
oversample = SMOTE(random_state=42)
avc_x = avc.drop(['stroke'], axis=1)
avc_y = avc['stroke']
x_train_res, y_train_res = oversample.fit_resample(avc_x, avc_y.ravel())
x_train, x_test, y_train, y_test = train_test_split(x_train_res, y_train_res, test_size=0.2, random_state=42)

# Train models
models = [
    ['Logistic Regression', LogisticRegression(random_state=42, max_iter=200)],
    ['SVM', SVC(random_state=42)],
    ['KNeighbors', KNeighborsClassifier()],
    ['Decision Tree', DecisionTreeClassifier(random_state=42)],
    ['Random Forest', RandomForestClassifier(random_state=42)]
]

def evaluate_models(x_test, y_test):
    results = []
    for name, model in models:
        start_time = datetime.now()
        model.fit(x_train, y_train)
        y_pred = model.predict(x_test)
        end_time = datetime.now()
        execution_time = (end_time - start_time).total_seconds()

        accuracy = accuracy_score(y_test, y_pred)
        roc = roc_auc_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)

        results.append({
            'Model': name,
            'Accuracy': accuracy,
            'ROC AUC': roc,
            'Precision': precision,
            'Recall': recall,
            'F1': f1,
            'Execution Time (s)': execution_time
        })
    return results

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    input_df = pd.DataFrame([data])
    input_df = input_df.replace({
        'gender': {'Male': 0, 'Female': 1, 'Other': 2},
        'ever_married': {'Yes': 0, 'No': 1},
        'work_type': {'Private': 0, 'Self-employed': 1, 'Govt_job': 2, 'children': 3, 'Never_worked': 4},
        'smoking_status': {'formerly smoked': 0, 'never smoked': 1, 'smokes': 2, 'Unknown': 3},
        'Residence_type': {'Urban': 0, 'Rural': 1}
    })

    # Prediction
    results = evaluate_models(input_df, [0])  # The second parameter is a dummy placeholder
    return jsonify(results)

if __name__ == '__main__':
    app.run(debug=True)
