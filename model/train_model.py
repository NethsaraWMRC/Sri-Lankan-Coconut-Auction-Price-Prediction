"""
Machine Learning Model for Coconut Auction Price Prediction
============================================================

This script trains an XGBoost Regressor to predict coconut auction prices
using historical auction data, USD exchange rates, and lag features.

Dataset: srilanka_coconut_auction_data_cleaned.csv
Target: Avg_Price_Rs./1000 (Average price per 1000 coconuts)

Author: ML Assignment
Date: February 2026
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib

# Machine Learning Libraries
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import xgboost as xgb

# Optional: Explainable AI (comment out if not installed)
try:
    import shap
    HAS_SHAP = True
except ImportError:
    HAS_SHAP = False
    print("Note: SHAP not installed. Skipping SHAP explanations.")

# Suppress warnings for cleaner output
import warnings
warnings.filterwarnings('ignore')

# Set random seed for reproducibility
RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

# Configure plotting style
plt.rcParams['figure.figsize'] = (12, 6)
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')


def load_data(file_path):
    """
    Load the cleaned coconut auction dataset.
    
    Args:
        file_path (str): Path to the cleaned CSV file
        
    Returns:
        pd.DataFrame: Loaded dataset with datetime conversion
    """
    print("="*70)
    print("STEP 1: LOADING DATA")
    print("="*70)
    
    df = pd.read_csv(file_path)
    
    # Convert Date column to datetime
    df['Date'] = pd.to_datetime(df['Date'])
    
    # Sort by date (chronological order is crucial for lag features)
    df = df.sort_values('Date').reset_index(drop=True)
    
    print(f"✓ Dataset loaded successfully")
    print(f"  Total records: {len(df)}")
    print(f"  Total features: {len(df.columns)}")
    print(f"  Date range: {df['Date'].min()} to {df['Date'].max()}")
    print(f"  Columns: {list(df.columns)}")
    
    return df


def engineer_features(df):
    """
    Create lag features for time-series prediction.
    
    LAG FEATURE: Previous week's price is often the best predictor of current price.
    This captures momentum and trend information.
    
    Args:
        df (pd.DataFrame): Input dataset
        
    Returns:
        pd.DataFrame: Dataset with lag features
    """
    print("\n" + "="*70)
    print("STEP 2: FEATURE ENGINEERING")
    print("="*70)
    
    print("  Creating lag feature: Prev_Week_Price")
    
    # Create lag feature: Previous week's price
    df['Prev_Week_Price'] = df['Avg_Price_Rs./1000'].shift(1)
    
    # Drop first row with NaN lag value
    initial_rows = len(df)
    df = df.dropna(subset=['Prev_Week_Price']).reset_index(drop=True)
    rows_dropped = initial_rows - len(df)
    
    print(f"  ✓ Lag feature created: Prev_Week_Price")
    print(f"  ✓ Dropped {rows_dropped} row(s) with missing lag values")
    print(f"  Final dataset size: {len(df)} records")
    
    return df


def prepare_features_and_target(df):
    """
    Prepare features (X) and target (y) for modeling.
    
    Target: Avg_Price_Rs./1000
    Features: 9 features including lag feature
    
    Args:
        df (pd.DataFrame): Input dataset
        
    Returns:
        tuple: (X, y, feature_names)
    """
    print("\n" + "="*70)
    print("STEP 3: FEATURE SELECTION")
    print("="*70)
    
    # Define feature columns (including lag feature)
    feature_columns = [
        'Offered_Quantity',
        'Sold_Quantity',
        'Sales_Ratio',
        'Unsold_Quantity',
        'Year',
        'Month',
        'Quarter',
        'USD_Avg_Rate',
        'Prev_Week_Price'  # LAG FEATURE - captures price momentum
    ]
    
    # Define target column
    target_column = 'Avg_Price_Rs./1000'
    
    # Extract features and target
    X = df[feature_columns].copy()
    y = df[target_column].copy()
    
    print(f"✓ Features selected: {len(feature_columns)}")
    print(f"  Feature names:")
    for i, feat in enumerate(feature_columns, 1):
        print(f"    {i}. {feat}")
    
    print(f"\n✓ Target variable: {target_column}")
    print(f"  Target range: {y.min():.2f} - {y.max():.2f}")
    print(f"  Target mean: {y.mean():.2f}")
    print(f"  Target std: {y.std():.2f}")
    
    return X, y, feature_columns


def split_data(X, y):
    """
    Split data into training and testing sets.
    
    IMPORTANT: With lag features and mixed time periods, we use shuffle=True
    to ensure the model sees both high and low price eras in training.
    This prevents the model from being biased towards recent price patterns only.
    
    Args:
        X (pd.DataFrame): Features
        y (pd.Series): Target
        
    Returns:
        tuple: (X_train, X_test, y_train, y_test)
    """
    print("\n" + "="*70)
    print("STEP 4: TRAIN-TEST SPLIT")
    print("="*70)
    
    # Split: 80% training, 20% testing
    # shuffle=True: Ensures model sees diverse price patterns across all time periods
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        shuffle=True,  # SHUFFLE to expose model to all price eras
        random_state=RANDOM_STATE
    )
    
    print(f"✓ Data split completed (80% train, 20% test)")
    print(f"  Training samples: {len(X_train)} ({len(X_train)/len(X)*100:.1f}%)")
    print(f"  Testing samples:  {len(X_test)} ({len(X_test)/len(X)*100:.1f}%)")
    print(f"\n  ✓ shuffle=True: Model exposed to diverse price patterns")
    print(f"     This prevents bias towards recent trends only")
    
    return X_train, X_test, y_train, y_test


def preprocess_data(X_train, X_test):
    """
    Preprocess features: Handle missing values and standardize.
    
    Args:
        X_train (pd.DataFrame): Training features
        X_test (pd.DataFrame): Testing features
        
    Returns:
        tuple: (X_train_scaled, X_test_scaled, scaler)
    """
    print("\n" + "="*70)
    print("STEP 5: PREPROCESSING")
    print("="*70)
    
    # Check for missing values
    missing_train = X_train.isnull().sum().sum()
    missing_test = X_test.isnull().sum().sum()
    
    if missing_train > 0 or missing_test > 0:
        print(f"  Missing values in training: {missing_train}")
        print(f"  Missing values in testing: {missing_test}")
        print("  Applying SimpleImputer (median strategy)...")
        
        imputer = SimpleImputer(strategy='median')
        X_train = pd.DataFrame(
            imputer.fit_transform(X_train),
            columns=X_train.columns,
            index=X_train.index
        )
        X_test = pd.DataFrame(
            imputer.transform(X_test),
            columns=X_test.columns,
            index=X_test.index
        )
        print("  ✓ Missing values handled")
    else:
        print("  ✓ No missing values detected")
    
    # Note: XGBoost handles raw features well, but we standardize for consistency
    print("\n  Standardizing features using StandardScaler...")
    scaler = StandardScaler()
    X_train_scaled = pd.DataFrame(
        scaler.fit_transform(X_train),
        columns=X_train.columns,
        index=X_train.index
    )
    X_test_scaled = pd.DataFrame(
        scaler.transform(X_test),
        columns=X_test.columns,
        index=X_test.index
    )
    print("  ✓ Features standardized (mean=0, std=1)")
    
    return X_train_scaled, X_test_scaled, scaler


def train_model(X_train, X_test, y_train, y_test):
    """
    Train an XGBoost Regressor model with early stopping.
    
    XGBoost is chosen because:
    - Superior performance on tabular data
    - Handles non-linear relationships well
    - Built-in regularization prevents overfitting
    - Fast training with gradient boosting
    - Early stopping prevents overtraining
    
    Args:
        X_train (pd.DataFrame): Training features
        X_test (pd.DataFrame): Testing features (for early stopping validation)
        y_train (pd.Series): Training target
        y_test (pd.Series): Testing target (for early stopping validation)
        
    Returns:
        xgb.XGBRegressor: Trained model
    """
    print("\n" + "="*70)
    print("STEP 6: MODEL TRAINING")
    print("="*70)
    
    print("  Model: XGBoost Regressor")
    print("  Hyperparameters:")
    print("    - n_estimators: 1000 (max trees)")
    print("    - learning_rate: 0.05 (slow learning for better generalization)")
    print("    - max_depth: 5 (tree depth)")
    print("    - early_stopping_rounds: 50 (stop if no improvement)")
    print("    - random_state: 42 (reproducibility)")
    
    # Initialize XGBoost Regressor
    model = xgb.XGBRegressor(
        n_estimators=1000,           # Maximum number of trees
        learning_rate=0.05,          # Lower learning rate for better generalization
        max_depth=5,                 # Maximum tree depth
        random_state=RANDOM_STATE,   # For reproducibility
        early_stopping_rounds=50,    # Stop if no improvement for 50 rounds
        eval_metric='rmse',          # Use RMSE for validation
        n_jobs=-1                    # Use all CPU cores
    )
    
    print("\n  Training model with early stopping...")
    print("  (Using test set for validation)")
    
    # Fit with early stopping
    model.fit(
        X_train, y_train,
        eval_set=[(X_test, y_test)],
        verbose=False  # Set to True to see training progress
    )
    
    best_iteration = model.best_iteration
    print(f"  ✓ Model training completed!")
    print(f"  Best iteration: {best_iteration} (early stopped)")
    
    return model


def evaluate_model(model, X_train, X_test, y_train, y_test):
    """
    Evaluate model performance using multiple metrics.
    
    Metrics:
    - R² Score: Coefficient of determination - how well model explains variance
    - RMSE: Root Mean Squared Error (in Rupees) - penalizes large errors
    - MAE: Mean Absolute Error (in Rupees) - average prediction error
    
    Args:
        model: Trained model
        X_train (pd.DataFrame): Training features
        X_test (pd.DataFrame): Testing features
        y_train (pd.Series): Training target
        y_test (pd.Series): Testing target
        
    Returns:
        dict: Dictionary containing predictions and metrics
    """
    print("\n" + "="*70)
    print("STEP 7: MODEL EVALUATION")
    print("="*70)
    
    # Make predictions
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)
    
    # Calculate metrics for training set
    train_r2 = r2_score(y_train, y_train_pred)
    train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
    train_mae = mean_absolute_error(y_train, y_train_pred)
    
    # Calculate metrics for testing set
    test_r2 = r2_score(y_test, y_test_pred)
    test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
    test_mae = mean_absolute_error(y_test, y_test_pred)
    
    # Print results (R² first as requested)
    print("\n📊 TRAINING SET PERFORMANCE:")
    print(f"  R² Score: {train_r2:.4f}")
    print(f"  RMSE: {train_rmse:,.2f} Rs/1000")
    print(f"  MAE:  {train_mae:,.2f} Rs/1000")
    
    print("\n📊 TESTING SET PERFORMANCE:")
    print(f"  R² Score: {test_r2:.4f}")
    print(f"  RMSE: {test_rmse:,.2f} Rs/1000")
    print(f"  MAE:  {test_mae:,.2f} Rs/1000")
    
    print("\n💡 INTERPRETATION:")
    print(f"  • Model explains {test_r2*100:.2f}% of price variance")
    print(f"  • On average, predictions are off by ±{test_mae:,.2f} Rs/1000")
    
    # Warning if R² is below 0.6
    if test_r2 < 0.6:
        print(f"\n  ⚠️  WARNING: R² Score ({test_r2:.4f}) is below 0.6")
        print(f"     Model may need more features or hyperparameter tuning")
    else:
        print(f"\n  ✓ Good performance: R² Score is {test_r2:.4f} (above 0.6 threshold)")
    
    # Check for overfitting
    if train_r2 - test_r2 > 0.1:
        print(f"  ⚠️  Warning: Possible overfitting detected")
        print(f"     (Training R² - Testing R² = {train_r2 - test_r2:.4f})")
    else:
        print(f"  ✓ No significant overfitting detected")
    
    return {
        'train_pred': y_train_pred,
        'test_pred': y_test_pred,
        'train_metrics': {'r2': train_r2, 'rmse': train_rmse, 'mae': train_mae},
        'test_metrics': {'r2': test_r2, 'rmse': test_rmse, 'mae': test_mae}
    }


def plot_actual_vs_predicted(y_test, y_test_pred, save_path='actual_vs_predicted.png'):
    """
    Plot actual vs predicted prices for visual assessment.
    Shows first 50 samples for readability.
    
    Args:
        y_test (pd.Series): Actual test values
        y_test_pred (np.array): Predicted test values
        save_path (str): Path to save the plot
    """
    print("\n" + "="*70)
    print("STEP 8: VISUALIZATION - ACTUAL VS PREDICTED")
    print("="*70)
    
    # Limit to first 50 samples for readability
    n_samples = min(50, len(y_test))
    y_test_subset = y_test.values[:n_samples]
    y_pred_subset = y_test_pred[:n_samples]
    
    print(f"  Plotting first {n_samples} samples for readability")
    
    plt.figure(figsize=(14, 6))
    
    # Plot 1: Line plot of actual vs predicted (first 50 samples)
    plt.subplot(1, 2, 1)
    plt.plot(range(n_samples), y_test_subset, label='Actual Price', 
             marker='o', linestyle='-', linewidth=2, markersize=6, alpha=0.8)
    plt.plot(range(n_samples), y_pred_subset, label='Predicted Price', 
             marker='s', linestyle='--', linewidth=2, markersize=6, alpha=0.8)
    plt.xlabel('Test Sample Index', fontsize=12)
    plt.ylabel('Price (Rs/1000)', fontsize=12)
    plt.title(f'Actual vs Predicted Prices (First {n_samples} Samples)', 
              fontsize=14, fontweight='bold')
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    
    # Plot 2: Scatter plot with perfect prediction line (all samples)
    plt.subplot(1, 2, 2)
    plt.scatter(y_test, y_test_pred, alpha=0.6, edgecolors='k', linewidth=0.5)
    
    # Add perfect prediction line
    min_val = min(y_test.min(), y_test_pred.min())
    max_val = max(y_test.max(), y_test_pred.max())
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Perfect Prediction')
    
    plt.xlabel('Actual Price (Rs/1000)', fontsize=12)
    plt.ylabel('Predicted Price (Rs/1000)', fontsize=12)
    plt.title('Prediction Accuracy Scatter Plot (All Test Samples)', 
              fontsize=14, fontweight='bold')
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"  ✓ Plot saved: {save_path}")
    plt.show()


def plot_feature_importance(model, feature_names, save_path='feature_importance.png'):
    """
    Plot feature importance from XGBoost.
    
    Feature importance shows which features the model relies on most
    for making predictions.
    
    Args:
        model: Trained XGBoost model
        feature_names (list): List of feature names
        save_path (str): Path to save the plot
    """
    print("\n" + "="*70)
    print("STEP 9: FEATURE IMPORTANCE ANALYSIS")
    print("="*70)
    
    # Get feature importance
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1]
    
    # Print feature importance
    print("\n📊 Feature Importance Ranking:")
    for i, idx in enumerate(indices, 1):
        print(f"  {i}. {feature_names[idx]:<20} {importances[idx]:.4f} ({importances[idx]*100:.2f}%)")
    
    # Plot feature importance
    plt.figure(figsize=(10, 6))
    plt.barh(range(len(importances)), importances[indices], align='center', alpha=0.8)
    plt.yticks(range(len(importances)), [feature_names[i] for i in indices])
    plt.xlabel('Importance Score', fontsize=12)
    plt.title('Feature Importance (XGBoost)', fontsize=14, fontweight='bold')
    plt.gca().invert_yaxis()
    plt.grid(True, alpha=0.3, axis='x')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"\n  ✓ Feature importance plot saved: {save_path}")
    plt.show()


def explain_with_shap(model, X_train, X_test, feature_names, save_path='shap_summary.png'):
    """
    Generate SHAP explanations for model predictions.
    
    SHAP (SHapley Additive exPlanations) shows:
    - Which features push predictions higher or lower
    - How feature values impact predictions
    - Global and local feature importance
    
    Args:
        model: Trained model
        X_train (pd.DataFrame): Training features (for background)
        X_test (pd.DataFrame): Testing features (to explain)
        feature_names (list): List of feature names
        save_path (str): Path to save the plot
    """
    if not HAS_SHAP:
        print("  ⚠️  SHAP not installed. Skipping SHAP analysis.")
        return
    
    print("\n" + "="*70)
    print("STEP 10: EXPLAINABLE AI (SHAP Analysis)")
    print("="*70)
    
    print("  Calculating SHAP values...")
    print("  (This may take a few minutes for XGBoost models)")
    
    # Create SHAP explainer for tree-based models
    explainer = shap.TreeExplainer(model)
    
    # Calculate SHAP values for test set
    shap_values = explainer.shap_values(X_test)
    
    print("  ✓ SHAP values calculated")
    
    # Plot SHAP summary (beeswarm plot)
    plt.figure(figsize=(10, 8))
    shap.summary_plot(shap_values, X_test, feature_names=feature_names, show=False)
    plt.title('SHAP Summary Plot (Feature Impact on Price)', fontsize=14, fontweight='bold', pad=20)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"  ✓ SHAP summary plot saved: {save_path}")
    plt.show()
    
    print("\n💡 INTERPRETING SHAP PLOT:")
    print("  • Each dot represents a prediction")
    print("  • Red = high feature value, Blue = low feature value")
    print("  • Position on x-axis shows impact on prediction")
    print("  • Features are ordered by importance (top = most important)")


def save_model_summary(results, output_path='model_summary.txt'):
    """
    Save model performance summary to a text file.
    
    Args:
        results (dict): Evaluation results
        output_path (str): Path to save the summary
    """
    print("\n" + "="*70)
    print("STEP 11: SAVING MODEL SUMMARY")
    print("="*70)
    
    with open(output_path, 'w') as f:
        f.write("="*70 + "\n")
        f.write("COCONUT AUCTION PRICE PREDICTION MODEL - SUMMARY\n")
        f.write("="*70 + "\n\n")
        
        f.write("MODEL CONFIGURATION:\n")
        f.write("-" * 40 + "\n")
        f.write("Algorithm: XGBoost Regressor\n")
        f.write("Max Estimators: 1000 (with early stopping)\n")
        f.write("Learning Rate: 0.05\n")
        f.write("Max Depth: 5\n")
        f.write("Random State: 42\n")
        f.write("Train-Test Split: 80-20 (shuffle=True)\n")
        f.write("Features: 9 (including Prev_Week_Price lag feature)\n\n")
        
        f.write("TRAINING SET PERFORMANCE:\n")
        f.write("-" * 40 + "\n")
        f.write(f"R² Score: {results['train_metrics']['r2']:.4f}\n")
        f.write(f"RMSE: {results['train_metrics']['rmse']:,.2f} Rs/1000\n")
        f.write(f"MAE:  {results['train_metrics']['mae']:,.2f} Rs/1000\n\n")
        
        f.write("TESTING SET PERFORMANCE:\n")
        f.write("-" * 40 + "\n")
        f.write(f"R² Score: {results['test_metrics']['r2']:.4f}\n")
        f.write(f"RMSE: {results['test_metrics']['rmse']:,.2f} Rs/1000\n")
        f.write(f"MAE:  {results['test_metrics']['mae']:,.2f} Rs/1000\n\n")
        
        f.write("="*70 + "\n")
    
    print(f"  ✓ Model summary saved: {output_path}")


def save_model_artifacts(model, scaler, feature_names, output_dir='outputs'):
    """
    Save trained model and preprocessing artifacts for deployment.
    
    Args:
        model: Trained XGBoost model
        scaler: Fitted StandardScaler
        feature_names (list): List of feature names
        output_dir (str): Directory to save artifacts
    """
    print("\n" + "="*70)
    print("STEP 12: SAVING MODEL ARTIFACTS FOR DEPLOYMENT")
    print("="*70)
    
    # Save model
    model_path = os.path.join(output_dir, 'model.joblib')
    joblib.dump(model, model_path)
    print(f"  ✓ Model saved: {model_path}")
    
    # Save scaler
    scaler_path = os.path.join(output_dir, 'scaler.joblib')
    joblib.dump(scaler, scaler_path)
    print(f"  ✓ Scaler saved: {scaler_path}")
    
    # Save feature names
    features_path = os.path.join(output_dir, 'feature_names.joblib')
    joblib.dump(feature_names, features_path)
    print(f"  ✓ Feature names saved: {features_path}")
    
    print(f"\n  📦 All artifacts ready for deployment!")
    print(f"     Use these files in your Streamlit app or API")


def main():
    """
    Main execution function for the ML pipeline.
    """
    print("\n" + "="*70)
    print("  COCONUT AUCTION PRICE PREDICTION - ML PIPELINE")
    print("="*70)
    print("  Model: XGBoost Regressor")
    print("  Target: Avg_Price_Rs./1000")
    print("  Date:", pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S"))
    print("="*70)
    
    # Get file paths
    current_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(current_dir)
    data_path = os.path.join(parent_dir, 'srilanka_coconut_auction_data_cleaned.csv')
    
    # Create output directory for plots
    output_dir = os.path.join(current_dir, 'outputs')
    os.makedirs(output_dir, exist_ok=True)
    
    # Pipeline execution
    # Step 1: Load data
    df = load_data(data_path)
    
    # Step 2: Feature engineering (add lag features)
    df = engineer_features(df)
    
    # Step 3: Prepare features and target
    X, y, feature_names = prepare_features_and_target(df)
    
    # Step 4: Split data (SHUFFLE=TRUE for diverse patterns!)
    X_train, X_test, y_train, y_test = split_data(X, y)
    
    # Step 5: Preprocess data
    X_train_scaled, X_test_scaled, scaler = preprocess_data(X_train, X_test)
    
    # Step 6: Train model (with early stopping using test set)
    model = train_model(X_train_scaled, X_test_scaled, y_train, y_test)
    
    # Step 7: Evaluate model
    results = evaluate_model(model, X_train_scaled, X_test_scaled, y_train, y_test)
    
    # Step 8: Visualizations
    plot_actual_vs_predicted(
        y_test, results['test_pred'],
        save_path=os.path.join(output_dir, 'actual_vs_predicted.png')
    )
    
    # Step 9: Feature importance
    plot_feature_importance(
        model, feature_names,
        save_path=os.path.join(output_dir, 'feature_importance.png')
    )
    
    # Step 10: SHAP explanations (optional)
    if HAS_SHAP:
        explain_with_shap(
            model, X_train_scaled, X_test_scaled, feature_names,
            save_path=os.path.join(output_dir, 'shap_summary.png')
        )
    else:
        print("\n" + "="*70)
        print("STEP 10: EXPLAINABLE AI (SHAP Analysis) - SKIPPED")
        print("="*70)
        print("  ℹ️  SHAP not installed. Install with: pip install shap")
        print("  ℹ️  Skipping SHAP analysis (not required for deployment)")
    
    # Step 11: Save summary
    save_model_summary(results, output_path=os.path.join(output_dir, 'model_summary.txt'))
    
    # Step 12: Save model artifacts for deployment
    save_model_artifacts(model, scaler, feature_names, output_dir=output_dir)
    
    # Final summary
    print("\n" + "="*70)
    print("  ✅ PIPELINE COMPLETED SUCCESSFULLY!")
    print("="*70)
    print(f"\n  📁 All outputs saved to: {output_dir}")
    print(f"  📊 Generated files:")
    print(f"     • actual_vs_predicted.png")
    print(f"     • feature_importance.png")
    print(f"     • shap_summary.png")
    print(f"     • model_summary.txt")
    print(f"  📦 Deployment artifacts:")
    print(f"     • model.joblib")
    print(f"     • scaler.joblib")
    print(f"     • feature_names.joblib")
    print("\n" + "="*70 + "\n")
    
    return model, results


if __name__ == "__main__":
    model, results = main()
