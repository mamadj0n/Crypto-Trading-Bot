import warnings
warnings.filterwarnings('ignore')

from data_pipeline import DataPipeline
from models import ModelTrainer
from ensemble import EnsembleOptimizer
from backtester import InstitutionalBacktester

def main():
    print("🚀 Starting Quantitative Trading Pipeline...")
    
    # 1. Pipeline Execution
    pipeline = DataPipeline()
    df_raw = pipeline.load_and_merge()
    df_clean = pipeline.create_targets_and_clean(df_raw)
    
    X_train_s, X_test_s, y_train, y_test, feature_names = pipeline.split_and_scale(df_clean)
    
    # 2. Model Training
    print("\\n🧠 Training Base Models...")
    trainer = ModelTrainer()
    trainer.fit_all(X_train_s, y_train)
    
    # Evaluate baseline
    xgb_preds = trainer.xgb_model.predict(X_test_s)
    trainer.evaluate_model(y_test, xgb_preds, "XGBoost Baseline")
    
    # 3. Optimize Ensemble
    print("\\n⚙️ Optimizing Ensemble Parameters...")
    optimizer = EnsembleOptimizer(trainer.xgb_model, trainer.cat_model, trainer.lgb_model)
    best_params, best_preds = optimizer.optimize_ensemble(X_train_s, y_train, X_test_s, y_test)
    
    # 4. Integrate Predictions to DataFrame and Run Backtest
    # (Extracting testing timeline mapping)
    test_timeline_df = df_clean.iloc[len(y_train):].copy()
    test_timeline_df['Signal'] = best_preds
    
    print("\\n📈 Running Institutional Backtest...")
    backtester = SingleBacktester(test_timeline_df)
    equity_curve, trades_df = backtester.run()
    backtester.calculate_professional_metrics()
    
if __name__ == "__main__":
    main()
