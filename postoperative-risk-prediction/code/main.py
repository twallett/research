#%%
!pip install -q imbalanced-learn xgboost

import pandas as pd
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from imblearn.over_sampling import SMOTE
import os

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier 

from models import Models
from tensorflow.keras.callbacks import EarlyStopping

from sklearn.metrics import roc_curve, roc_auc_score, confusion_matrix, accuracy_score, precision_score, recall_score, f1_score

os.makedirs('results/tables', exist_ok=True)
os.makedirs('results/cm', exist_ok=True)
os.makedirs('results/roc', exist_ok=True)

df = pd.read_csv("data/LapGenSurgOnly_2022.csv")

INPUT_FEATURES = [
    "Age", "SEX", "RACE_NEW", "BMI", "INOUT", "ASACLAS", "CPT", "ANESTHES",
    "DIABETES", "SMOKE", "FNSTATUS2", "HXCOPD", "ASCITES", "HXCHF", "HYPERMED",
    "DIALYSIS", "DISCANCR", "STEROID", "TRANSFUS"
]
TARGET_FEATURES = ['CDARREST', 'CDMI', 'PULEMBOL', 'REINTUB', 'OUPNEUMO', 'FAILWEAN']

CM_LABELS = {"CDARREST": ["No Complication", "Cardiac Arrest"]}
for target in TARGET_FEATURES[1:]:
    CM_LABELS[target] = [df[TARGET_FEATURES][target].unique()[0],df[TARGET_FEATURES][target].unique()[1]]

MODEL_NAMES = [
    "2layer_mlp", "2layer_cnn", "4layer_mlp", "4layer_cnn", "8layer_mlp", "8layer_cnn",
    "rf", "lr", "knn", "dt", "nb", "xgb" 
]

colors = plt.cm.tab10(np.linspace(0, 1, len(MODEL_NAMES)))

def get_early_stopping():
    return EarlyStopping(
        monitor='val_loss',         
        mode='min',                
        patience=20,                
        verbose=1,                  
        restore_best_weights=True 
    )

for target in TARGET_FEATURES:
    
    print(f"\n{'='*30} Target: {target} {'='*30}")
    
    X = df[INPUT_FEATURES].copy()
    y = df[target].copy()

    X = X.replace("Unknown", np.nan)
    data = pd.concat([X, y], axis=1).dropna()
    X = data[INPUT_FEATURES]
    y = data[target]

    label_encoders = {}
    for col in X.select_dtypes(include='object').columns:
        le = LabelEncoder()
        X.loc[:, col] = le.fit_transform(X[col])
        label_encoders[col] = le

    if y.dtype == object:
        y = LabelEncoder().fit_transform(y)
        if target == 'CDMI':
            y = 1 - y

    X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.2, random_state=42)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    oversampler = SMOTE(random_state=42)
    X_train_res, y_train_res = oversampler.fit_resample(X_train_scaled, y_train)

    early_stopping = get_early_stopping()
    
    nn_models = [
        Models.create_2layer_mlp(X_train_res.shape[1]),
        Models.create_2layer_cnn(X_train_res.shape[1]),
        Models.create_4layer_mlp(X_train_res.shape[1]),
        Models.create_4layer_cnn(X_train_res.shape[1]),
        Models.create_8layer_mlp(X_train_res.shape[1]),
        Models.create_8layer_cnn(X_train_res.shape[1])
    ]
    
    nn_predictions_prob = []
    nn_predictions = []
    nn_training_histories = []
    
    for i, nn_model in enumerate(nn_models):
        
        print(f"Training {MODEL_NAMES[i]}...")
        
        history = nn_model.fit(
            X_train_res, y_train_res, 
            epochs=500,                     
            batch_size=1024, 
            validation_split=0.2, 
            verbose=1,
            callbacks=[early_stopping]      
        )
        
        nn_training_histories.append(history)
        
        prob = nn_model.predict(X_test_scaled).ravel()
        pred = (prob > 0.5).astype(int)
        nn_predictions_prob.append(prob)
        nn_predictions.append(pred)

    classic_models = [
        RandomForestClassifier(),
        LogisticRegression(),
        KNeighborsClassifier(),
        DecisionTreeClassifier(),
        GaussianNB(),
        XGBClassifier(use_label_encoder=False, eval_metric='logloss')  # Added XGBoost classifier
    ]
    
    predictions_prob = nn_predictions_prob.copy()
    predictions = nn_predictions.copy()

    print("Training classical models...")
    for clf in tqdm(classic_models):
        clf.fit(X_train_res, y_train_res)
        
        if isinstance(clf, XGBClassifier):
            prob = clf.predict_proba(X_test_scaled)
            predictions_prob.append(prob[:, 1])
        else:
            prob = clf.predict_proba(X_test_scaled)
            predictions_prob.append(prob[:, 1])
            
        predictions.append(clf.predict(X_test_scaled))

    metrics_list = []
    
    plt.figure(figsize=(12, 8))
    plt.plot([0, 1], [0, 1], color='gray', linestyle='--', label='Random Classifier (AUC = 0.5)')
    
    for i, (name, prob, pred) in enumerate(zip(MODEL_NAMES, predictions_prob, predictions)):
        
        fpr, tpr, _ = roc_curve(y_test, prob)
        auc_score = roc_auc_score(y_test, prob)

        plt.plot(fpr, tpr, lw=2, color=colors[i], label=f'{name} (AUC = {auc_score:.4f})')

        auc_score = roc_auc_score(y_test, prob)
        cm = confusion_matrix(y_test, pred)
        accuracy = accuracy_score(y_test, pred)
        precision_val = precision_score(y_test, pred, zero_division=0)
        recall_val = recall_score(y_test, pred, zero_division=0)
        f1 = f1_score(y_test, pred, zero_division=0)
        
        metrics_list.append({
            "Model": name,
            "AUC": auc_score,
            "Accuracy": accuracy,
            "Precision": precision_val,
            "Recall": recall_val,
            "F1-Score": f1
        })

    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'ROC Curve - {target}')
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend(loc='best')
    
    plt.savefig(f"results/roc/{target}_roc_curve.pdf")
    plt.show()

    metrics_df = pd.DataFrame(metrics_list).set_index("Model").round(4)
    
    metrics_df.to_csv(f"results/tables/{target}_metrics.csv")
    print(metrics_df)

    # Find model with best AUC and best Recall
    best_auc_idx = np.argmax([m['AUC'] for m in metrics_list])
    best_recall_idx = np.argmax([m['Recall'] for m in metrics_list])

    best_auc_name = metrics_list[best_auc_idx]['Model']
    best_recall_name = metrics_list[best_recall_idx]['Model']

    best_auc_pred = predictions[best_auc_idx]
    best_recall_pred = predictions[best_recall_idx]

    # Create confusion matrices for both best models
    best_auc_cm = confusion_matrix(y_test, best_auc_pred)
    best_recall_cm = confusion_matrix(y_test, best_recall_pred)

    # Save confusion matrix for best AUC model
    plt.figure(figsize=(8, 6))
    plt.imshow(best_auc_cm, interpolation='nearest', cmap='Blues')
    plt.title(f'Confusion Matrix - {target} - Best AUC Model: {best_auc_name} (AUC = {metrics_list[best_auc_idx]["AUC"]:.4f})')
    plt.colorbar()

    classes = CM_LABELS[target]
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes)
    plt.yticks(tick_marks, classes)

    thresh = best_auc_cm.max() / 2
    for i in range(best_auc_cm.shape[0]):
        for j in range(best_auc_cm.shape[1]):
            plt.text(j, i, format(best_auc_cm[i, j], 'd'),
                     ha="center", va="center",
                     color="white" if best_auc_cm[i, j] > thresh else "black")

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()

    plt.savefig(f"results/cm/{target}_best_auc_confusion_matrix.pdf")
    plt.show()

    # Save confusion matrix for best Recall model
    plt.figure(figsize=(8, 6))
    plt.imshow(best_recall_cm, interpolation='nearest', cmap='Blues')
    plt.title(f'Confusion Matrix - {target} - Best Recall Model: {best_recall_name} (Recall = {metrics_list[best_recall_idx]["Recall"]:.4f})')
    plt.colorbar()

    plt.xticks(tick_marks, classes)
    plt.yticks(tick_marks, classes)

    thresh = best_recall_cm.max() / 2
    for i in range(best_recall_cm.shape[0]):
        for j in range(best_recall_cm.shape[1]):
            plt.text(j, i, format(best_recall_cm[i, j], 'd'),
                     ha="center", va="center",
                     color="white" if best_recall_cm[i, j] > thresh else "black")

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()

    plt.savefig(f"results/cm/{target}_best_recall_confusion_matrix.pdf")
    plt.show()

    # Print results for both metrics
    print(f"\nBest AUC model for {target}: {best_auc_name} (AUC = {metrics_list[best_auc_idx]['AUC']:.4f})")
    print(f"AUC Confusion Matrix:\n{best_auc_cm}")

    print(f"\nBest Recall model for {target}: {best_recall_name} (Recall = {metrics_list[best_recall_idx]['Recall']:.4f})")
    print(f"Recall Confusion Matrix:\n{best_recall_cm}")
# %%

