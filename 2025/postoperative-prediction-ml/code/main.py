#%%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from imblearn.over_sampling import SMOTE
import os

!pip install -q imbalanced-learn xgboost keras-tuner

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier 

from tensorflow.keras.callbacks import EarlyStopping
from keras_tuner import RandomSearch
from models import (
    TwoLayerMLPHyperModel, FourLayerMLPHyperModel, EightLayerMLPHyperModel,
    TwoLayerCNNHyperModel, FourLayerCNNHyperModel, EightLayerCNNHyperModel
)

from sklearn.metrics import roc_curve, roc_auc_score, confusion_matrix, accuracy_score, precision_score, recall_score, f1_score

os.makedirs('results/tables', exist_ok=True)
os.makedirs('results/cm', exist_ok=True)
os.makedirs('results/roc', exist_ok=True)
os.makedirs('results/hyperparameters', exist_ok=True)
os.makedirs('results/best_models', exist_ok=True)

df = pd.read_csv("data/LapGenSurgOnly_2022.csv")

INPUT_FEATURES = [
    "Age", "SEX", "RACE_NEW", "BMI", "INOUT", "ASACLAS", "CPT",
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

dt_param_grid = {
    'max_depth': [10, 20, 30],
    'criterion': ['gini', 'entropy']
}

rf_param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [10, 20, 30]
}

xgb_param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [10, 20, 30],
    'learning_rate': [0.01, 0.05, 0.1]
}

knn_param_grid = {
    'n_neighbors': [3, 5, 7, 9, 11],
    'metric': ['euclidean', 'manhattan']
}

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
    
    best_hyperparameters = {}
    
    predictions_prob = []
    predictions = []
    
    # STEP 1: NEURAL NETWORK HYPERPARAMETER TUNING AND TRAINING
    print("\nTuning Neural Network Hyperparameters...\n")
    
    input_shape = X_train_res.shape[1]
    
    hypermodels = [
        (TwoLayerMLPHyperModel(input_shape), "2layer_mlp"),
        (TwoLayerCNNHyperModel(input_shape), "2layer_cnn"),
        (FourLayerMLPHyperModel(input_shape), "4layer_mlp"),
        (FourLayerCNNHyperModel(input_shape), "4layer_cnn"),
        (EightLayerMLPHyperModel(input_shape), "8layer_mlp"),
        (EightLayerCNNHyperModel(input_shape), "8layer_cnn")
    ]
    
    for hypermodel, model_name in hypermodels:
        print(f"\nTuning {model_name}...")
        
        # 1. Find optimal hyperparameters with cross-validation
        tuner_dir = f'tuner/{target}/{model_name}'
        os.makedirs(tuner_dir, exist_ok=True)
        
        tuner = RandomSearch(
            hypermodel,
            objective='val_loss',
            max_trials=3,  
            executions_per_trial=1,
            directory=tuner_dir,
            project_name=f'{target}_{model_name}'
        )
        
        tuner.search(
            X_train_res, y_train_res,
            epochs=50, 
            batch_size=1024,
            validation_split=0.2,
            callbacks=[get_early_stopping()]
        )
        
        best_hps = tuner.get_best_hyperparameters(1)[0]
        best_hyperparameters[model_name] = best_hps
        
        with open(f'results/hyperparameters/{target}_{model_name}_best_hps.txt', 'w') as f:
            f.write(str(best_hps.values))
        
        # 2. Build model with best hyperparameters
        best_model = tuner.hypermodel.build(best_hps)
        
        # 3. Train on full training set with best hyperparameters
        print(f"Training {model_name} with best hyperparameters...")
        history = best_model.fit(
            X_train_res, y_train_res,
            epochs=500,
            batch_size=1024,
            validation_split=0.2,
            callbacks=[get_early_stopping()],
            verbose=1
        )
        
        best_model.save(f'results/best_models/{target}_{model_name}_best_model.h5')
        
        # 4. Evaluate on test set
        prob = best_model.predict(X_test_scaled).ravel()
        pred = (prob > 0.5).astype(int)
        predictions_prob.append(prob)
        predictions.append(pred)

    # STEP 2: CLASSICAL MODEL HYPERPARAMETER TUNING AND TRAINING
    print("\nTuning Classical Model Hyperparameters and Training Final Models...\n")
    
    # Special case for Naive Bayes (not much to tune)
    print("\nTraining Naive Bayes...")
    nb = GaussianNB()
    nb.fit(X_train_res, y_train_res)
    nb_prob = nb.predict_proba(X_test_scaled)[:, 1]
    nb_pred = nb.predict(X_test_scaled)
    best_hyperparameters["nb"] = "No hyperparameters to tune"
    
    # Special case for Logistic Regression (no tuning per request)
    print("\nTraining Logistic Regression (no hyperparameter tuning)...")
    lr = LogisticRegression(random_state=42)
    lr.fit(X_train_res, y_train_res)
    lr_prob = lr.predict_proba(X_test_scaled)[:, 1]
    lr_pred = lr.predict(X_test_scaled)
    best_hyperparameters["lr"] = "No hyperparameters tuned - trained with default parameters"
    
    # Track predictions based on MODEL_NAMES order
    lr_prediction_prob = lr_prob
    lr_prediction = lr_pred
    nb_prediction_prob = nb_prob
    nb_prediction = nb_pred
    
    # Define the classic models to tune
    classic_models = [
        (RandomForestClassifier(random_state=42), rf_param_grid, "rf"),
        (KNeighborsClassifier(), knn_param_grid, "knn"),
        (DecisionTreeClassifier(random_state=42), dt_param_grid, "dt"),
        (XGBClassifier(random_state=42, use_label_encoder=False, eval_metric='logloss'), xgb_param_grid, "xgb")
    ]
    
    # For each classical model (excluding LR and NB)
    for model, param_grid, model_name in classic_models:
        print(f"\nTuning {model_name}...")
        
        # 1. Find optimal hyperparameters with cross-validation
        grid_search = GridSearchCV(
            model, param_grid, 
            cv=3,  # Using 5-fold cross-validation for finding best parameters
            scoring='roc_auc', 
            n_jobs=-1, verbose=1
        )
        
        grid_search.fit(X_train_res, y_train_res)
        
        best_params = grid_search.best_params_
        best_hyperparameters[model_name] = best_params
        
        with open(f'results/hyperparameters/{target}_{model_name}_best_params.txt', 'w') as f:
            f.write(str(best_params))
        
        # 2. Create a new model with best hyperparameters
        if model_name == "rf":
            final_model = RandomForestClassifier(random_state=42, **best_params)
        elif model_name == "knn":
            final_model = KNeighborsClassifier(**best_params)
        elif model_name == "dt":
            final_model = DecisionTreeClassifier(random_state=42, **best_params)
        elif model_name == "xgb":
            final_model = XGBClassifier(random_state=42, use_label_encoder=False, eval_metric='logloss', **best_params)
        
        # 3. Train on full training set with best hyperparameters
        print(f"Training final {model_name} model with best hyperparameters...")
        final_model.fit(X_train_res, y_train_res)
        
        # 4. Evaluate on test set
        if hasattr(final_model, "predict_proba"):
            prob = final_model.predict_proba(X_test_scaled)[:, 1]
        else:
            prob = final_model.decision_function(X_test_scaled)
            
        pred = final_model.predict(X_test_scaled)
        
        # Add predictions in order matching MODEL_NAMES
        if model_name == "rf":
            predictions_prob.append(prob)
            predictions.append(pred)
            # Add LR after RF
            predictions_prob.append(lr_prediction_prob)
            predictions.append(lr_prediction)
        elif model_name == "knn":
            predictions_prob.append(prob)
            predictions.append(pred)
        elif model_name == "dt":
            predictions_prob.append(prob)
            predictions.append(pred)
            # Add NB after DT
            predictions_prob.append(nb_prediction_prob)
            predictions.append(nb_prediction)
        elif model_name == "xgb":
            predictions_prob.append(prob)
            predictions.append(pred)
    
    # STEP 3: EVALUATION AND VISUALIZATION
    metrics_list = []
    
    plt.figure(figsize=(10, 7)) 
    plt.plot([0, 1], [0, 1], color='gray', linestyle='--', label='Random Classifier (AUC = 0.5)')

    markers = ['o', '^', 's', 'D', 'v', '<', '>', 'p', '*', 'h', 'x', '+']

    for i, (name, prob, pred) in enumerate(zip(MODEL_NAMES, predictions_prob, predictions)):
        fpr, tpr, _ = roc_curve(y_test, prob)
        auc_score = roc_auc_score(y_test, prob)

        plt.plot(fpr, tpr, lw=1.5, color=colors[i], 
                marker=markers[i % len(markers)], markevery=0.1, markersize=6,
                label=f'{name} (AUC = {auc_score:.4f})')
        
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
    plt.title(f'ROC Curve - {target} (Best Models)')
    plt.grid(True, linestyle='--', alpha=0.6)

    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5), 
              frameon=True, fancybox=False, edgecolor='black', 
              fontsize=9)

    plt.tight_layout()
    plt.savefig(f"results/roc/{target}_roc_curve_best_models.pdf", bbox_inches='tight')
    plt.show()

    metrics_df = pd.DataFrame(metrics_list).set_index("Model").round(4)
    
    metrics_df.to_csv(f"results/tables/{target}_metrics_best_models.csv")
    print("\nModel Performance Metrics:")
    print(metrics_df)

    best_auc_idx = np.argmax([m['AUC'] for m in metrics_list])
    best_recall_idx = np.argmax([m['Recall'] for m in metrics_list])

    best_auc_name = metrics_list[best_auc_idx]['Model']
    best_recall_name = metrics_list[best_recall_idx]['Model']

    best_auc_pred = predictions[best_auc_idx]
    best_recall_pred = predictions[best_recall_idx]

    best_auc_cm = confusion_matrix(y_test, best_auc_pred)
    best_recall_cm = confusion_matrix(y_test, best_recall_pred)

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

    print(f"\nBest AUC model for {target}: {best_auc_name} (AUC = {metrics_list[best_auc_idx]['AUC']:.4f})")
    print(f"AUC Confusion Matrix:\n{best_auc_cm}")

    print(f"\nBest Recall model for {target}: {best_recall_name} (Recall = {metrics_list[best_recall_idx]['Recall']:.4f})")
    print(f"Recall Confusion Matrix:\n{best_recall_cm}")
    
    with open(f'results/hyperparameters/{target}_all_best_params.txt', 'w') as f:
        for model_name, params in best_hyperparameters.items():
            f.write(f"{model_name}: {params}\n")
    
    print(f"\nBest hyperparameters for all models saved to results/hyperparameters/{target}_all_best_params.txt")

# %%
