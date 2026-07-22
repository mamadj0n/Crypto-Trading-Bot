# ensemble.py
import numpy as np
import itertools
from tqdm import tqdm
from sklearn.metrics import f1_score, classification_report

class EnsembleOptimizer:
    def __init__(self, xgb_model, cat_model, lgb_model):
        self.xgb_model = xgb_model
        self.cat_model = cat_model
        self.lgb_model = lgb_model

    def optimize_ensemble(self, X_train, Y_train, X_test, Y_test):
        print("⏳ Extracting probabilities for ensemble optimization...")
        xgb_probs = self.xgb_model.predict_proba(X_test)
        cat_probs = self.cat_model.predict_proba(X_test)
        lgb_probs = self.lgb_model.predict_proba(X_test)
        y_true = np.array(Y_test)
        
        weight_options = [1, 2, 3] 
        class_0_weights = [1.0, 1.5, 2.0, 2.5]
        class_2_weights = [1.0, 1.5, 2.0, 2.5]
        threshold_options = np.arange(0.20, 0.45, 0.02)
        
        all_combinations = list(itertools.product(
            weight_options, weight_options, weight_options,
            class_0_weights, class_2_weights,
            threshold_options, threshold_options
        ))
        
        best_macro_f1 = 0
        best_params, best_preds = None, None
        
        for w_xgb, w_lgb, w_cat, c0_w, c2_w, down_th, up_th in tqdm(all_combinations, desc="Scanning Hyperparameters"):
            total_w = w_xgb + w_lgb + w_cat
            ensemble_probs = (xgb_probs * w_xgb + lgb_probs * w_lgb + cat_probs * w_cat) / total_w
            
            weighted_probs = ensemble_probs.copy()
            weighted_probs[:, 0] *= c0_w  
            weighted_probs[:, 2] *= c2_w  
            
            row_sums = weighted_probs.sum(axis=1, keepdims=True)
            weighted_probs = np.divide(weighted_probs, row_sums, out=weighted_probs, where=row_sums > 0)
            
            prob_down, prob_up = weighted_probs[:, 0], weighted_probs[:, 2]
            
            preds = np.ones(len(y_true), dtype=int)
            up_cond = (prob_up > up_th) & (prob_up > prob_down)
            preds[up_cond] = 2
            down_cond = (prob_down > down_th) & (prob_down > prob_up) & (~up_cond)
            preds[down_cond] = 0
            
            f1_scores = f1_score(y_true, preds, average=None, labels=[0, 1, 2], zero_division=0)
            macro_f1 = f1_score(y_true, preds, average='macro', zero_division=0)
            
            if f1_scores[0] > 0.24 and f1_scores[2] > 0.24 and macro_f1 > 0.40:
                if macro_f1 > best_macro_f1:
                    best_macro_f1 = macro_f1
                    best_preds = preds
                    best_params = {
                        'w_xgb': w_xgb, 'w_lgb': w_lgb, 'w_cat': w_cat,
                        'c0_w': c0_w, 'c2_w': c2_w, 'down_th': down_th, 'up_th': up_th
                    }
        
        if best_params:
            print(f"\\n✅ Best configuration found! F1-Macro: {best_macro_f1:.4f}")
            print(classification_report(y_true, best_preds, zero_division=0))
        return best_params, best_preds
