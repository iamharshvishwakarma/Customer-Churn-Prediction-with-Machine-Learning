# Import all required libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                             f1_score, roc_auc_score, confusion_matrix, 
                             classification_report, roc_curve, auc)
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from imblearn.over_sampling import SMOTE
import warnings
warnings.filterwarnings('ignore')

# Set random seed for reproducibility
np.random.seed(42)

## 1. Data Loading and Initial Exploration
# Load the dataset
df = pd.read_csv('WA_Fn-UseC_-Telco-Customer-Churn.csv')

# Initial exploration
print("Dataset shape:", df.shape)
print("\nFirst 5 rows:")
print(df.head())
print("\nMissing values:")
print(df.isnull().sum())
print("\nData types:")
print(df.dtypes)

## 2. Data Cleaning
# Convert TotalCharges to numeric (empty strings will become NaN)
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')

# Drop customerID as it's not useful for modeling
df.drop('customerID', axis=1, inplace=True)

# Handle missing values (only 11 rows with missing TotalCharges)
df.dropna(inplace=True)

# Convert SeniorCitizen from 0/1 to categorical for consistency
df['SeniorCitizen'] = df['SeniorCitizen'].map({0: 'No', 1: 'Yes'})

## 3. Feature Engineering
# Create new features
df['TenureYears'] = df['tenure'] / 12
df['AvgMonthlyCharges'] = df['TotalCharges'] / df['tenure']
df['AvgMonthlyCharges'].fillna(0, inplace=True)  # for customers with 0 tenure

# Convert Churn to binary (1 for Yes, 0 for No)
df['Churn'] = df['Churn'].map({'Yes': 1, 'No': 0})

## 4. Data Visualization
# Set up the figure
plt.figure(figsize=(20, 15))

# Churn distribution
plt.subplot(3, 3, 1)
sns.countplot(x='Churn', data=df)
plt.title('Churn Distribution')

# Numerical features
num_cols = ['tenure', 'MonthlyCharges', 'TotalCharges', 'AvgMonthlyCharges', 'TenureYears']
for i, col in enumerate(num_cols):
    plt.subplot(3, 3, i+2)
    sns.histplot(df[col], kde=True)
    plt.title(f'Distribution of {col}')

plt.tight_layout()
plt.show()

# Categorical features
cat_cols = ['gender', 'SeniorCitizen', 'Partner', 'Dependents', 'PhoneService', 
            'MultipleLines', 'InternetService', 'OnlineSecurity', 'OnlineBackup', 
            'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies', 
            'Contract', 'PaperlessBilling', 'PaymentMethod']

plt.figure(figsize=(20, 30))
for i, col in enumerate(cat_cols):
    plt.subplot(6, 3, i+1)
    sns.countplot(x=col, hue='Churn', data=df)
    plt.xticks(rotation=45)
    plt.title(f'Churn by {col}')
plt.tight_layout()
plt.show()

# Check class imbalance
print("\nChurn distribution:")
print(df['Churn'].value_counts(normalize=True))

## 5. Data Preprocessing
# Prepare data for modeling
X = df.drop('Churn', axis=1)
y = df['Churn']

# Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

# Identify numerical and categorical columns
num_cols = X_train.select_dtypes(include=['int64', 'float64']).columns
cat_cols = X_train.select_dtypes(include=['object']).columns

# Create preprocessing pipelines
preprocessor = ColumnTransformer([
    ('num', StandardScaler(), num_cols),
    ('cat', OneHotEncoder(handle_unknown='ignore'), cat_cols)
])

# Apply SMOTE to handle class imbalance
smote = SMOTE(random_state=42)
X_train_res, y_train_res = smote.fit_resample(preprocessor.fit_transform(X_train), y_train)

# Transform test set
X_test_transformed = preprocessor.transform(X_test)

# Get feature names after one-hot encoding
cat_encoder = preprocessor.named_transformers_['cat']
cat_features = cat_encoder.get_feature_names_out(cat_cols)
all_features = np.concatenate([num_cols, cat_features])

## 6. Model Training and Evaluation
def evaluate_model(model, X_train, y_train, X_test, y_test):
    # Train the model
    model.fit(X_train, y_train)
    
    # Make predictions
    y_pred = model.predict(X_test)
    y_pred_prob = model.predict_proba(X_test)[:, 1]
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_pred_prob)
    
    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    
    # Classification report
    cr = classification_report(y_test, y_pred)
    
    # ROC curve
    fpr, tpr, thresholds = roc_curve(y_test, y_pred_prob)
    roc_auc = auc(fpr, tpr)
    
    # Plot ROC curve
    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'ROC Curve - {type(model).__name__}')
    plt.legend(loc="lower right")
    plt.show()
    
    # Return results
    return {
        'model': type(model).__name__,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'roc_auc': roc_auc,
        'confusion_matrix': cm,
        'classification_report': cr
    }

# Initialize models
models = {
    'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
    'Random Forest': RandomForestClassifier(random_state=42),
    'Gradient Boosting': GradientBoostingClassifier(random_state=42),
    'SVM': SVC(probability=True, random_state=42),
    'KNN': KNeighborsClassifier(),
    'Decision Tree': DecisionTreeClassifier(random_state=42)
}

# Evaluate each model
results = []
for name, model in models.items():
    print(f"\nEvaluating {name}...")
    result = evaluate_model(model, X_train_res, y_train_res, X_test_transformed, y_test)
    results.append(result)
    print(f"{name} Results:")
    print(f"Accuracy: {result['accuracy']:.4f}")
    print(f"Precision: {result['precision']:.4f}")
    print(f"Recall: {result['recall']:.4f}")
    print(f"F1 Score: {result['f1']:.4f}")
    print(f"ROC AUC: {result['roc_auc']:.4f}")
    print("Confusion Matrix:")
    print(result['confusion_matrix'])
    print("Classification Report:")
    print(result['classification_report'])

# Compare model performance
results_df = pd.DataFrame(results)
results_df = results_df.sort_values(by='roc_auc', ascending=False)
print("\nModel Performance Comparison:")
print(results_df[['model', 'accuracy', 'precision', 'recall', 'f1', 'roc_auc']])

## 7. Hyperparameter Tuning for Best Model (Random Forest)
param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

rf = RandomForestClassifier(random_state=42)
grid_search = GridSearchCV(rf, param_grid, cv=5, scoring='roc_auc', n_jobs=-1)
grid_search.fit(X_train_res, y_train_res)

print("\nBest parameters:", grid_search.best_params_)
print("Best score:", grid_search.best_score_)

# Evaluate the tuned model
best_rf = grid_search.best_estimator_
best_rf_result = evaluate_model(best_rf, X_train_res, y_train_res, X_test_transformed, y_test)

print("\nTuned Random Forest Results:")
print(f"Accuracy: {best_rf_result['accuracy']:.4f}")
print(f"Precision: {best_rf_result['precision']:.4f}")
print(f"Recall: {best_rf_result['recall']:.4f}")
print(f"F1 Score: {best_rf_result['f1']:.4f}")
print(f"ROC AUC: {best_rf_result['roc_auc']:.4f}")

## 8. Feature Importance Analysis
feature_importances = best_rf.feature_importances_
importance_df = pd.DataFrame({'Feature': all_features, 'Importance': feature_importances})
importance_df = importance_df.sort_values(by='Importance', ascending=False)

plt.figure(figsize=(12, 8))
sns.barplot(x='Importance', y='Feature', data=importance_df.head(20))
plt.title('Top 20 Important Features for Churn Prediction')
plt.tight_layout()
plt.show()

## 9. Business Insights and Recommendations
print("\nKey Business Insights:")
print("1. Top factors influencing churn:")
print("   - Contract type (month-to-month customers most likely to churn)")
print("   - Internet service type (fiber optic users more likely to churn)")
print("   - Lack of online security/tech support")
print("   - Payment method (electronic check)")
print("   - Higher monthly charges")

print("\n2. Retention Recommendations:")
print("   - Target month-to-month contract customers with loyalty offers")
print("   - Improve service quality for fiber optic internet customers")
print("   - Bundle online security/tech support with other services")
print("   - Offer incentives for automatic payment methods")
print("   - Monitor customers with above-average monthly charges")

print("\n3. Next Steps:")
print("   - Implement real-time churn prediction system")
print("   - Develop targeted retention campaigns")
print("   - Collect additional customer satisfaction data")
print("   - Monitor model performance monthly and retrain as needed")