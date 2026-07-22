# models.py
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier

class ModelTrainer:
    def __init__(self):
        self.xgb_model = XGBClassifier(
            objective="multi:softprob", num_class=3, random_state=42,
            eval_metric="mlogloss", learning_rate=0.03, max_depth=3,
            max_delta_step=3, n_estimators=150
        )
        self.cat_model = CatBoostClassifier(
            iterations=150, learning_rate=0.03, depth=4,
            loss_function="MultiClass", random_seed=42, verbose=0
        )
        self.lgb_model = LGBMClassifier(
            n_estimators=150, learning_rate=0.02, num_leaves=15,
            max_depth=4, random_state=42, verbosity=-1
        )

    def fit_all(self, X_train, Y_train):
        self.xgb_model.fit(X_train, Y_train)
        self.cat_model.fit(X_train, Y_train)
        self.lgb_model.fit(X_train, Y_train)

    def evaluate_model(self, y_true, y_pred, model_name):
        accuracy = accuracy_score(y_true, y_pred)
        precision_w = precision_score(y_true, y_pred, average='weighted')
        recall_w = recall_score(y_true, y_pred, average='weighted')
        f1_w = f1_score(y_true, y_pred, average='weighted')
        
        print(f"\\n{'='*55}\\n📊  {model_name}  -  Performance Report\\n{'='*55}")
        print(f"✅ Accuracy: {accuracy:.4f} | ⚖️ F1-Score (Weighted): {f1_w:.4f}")
        print(classification_report(y_true, y_pred, digits=4))
        
        return {'Model': model_name, 'Accuracy': accuracy, 'F1 (W)': f1_w}
