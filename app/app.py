"""
Coconut Auction Price Prediction App
=====================================

A Streamlit web application for predicting coconut auction prices
using an XGBoost machine learning model with SHAP explainability.

Author: ML Assignment
Date: February 2026
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
from datetime import datetime
import os
import pickle
import xgboost as xgb
import warnings
warnings.filterwarnings('ignore')

# Try to import SHAP (optional)
try:
    import shap
    HAS_SHAP = True
except ImportError:
    HAS_SHAP = False

# Page configuration
st.set_page_config(
    page_title="Coconut Price Predictor",
    page_icon="🥥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
    <style>
    .main-header {
        font-size: 3rem;
        color: #2E7D32;
        text-align: center;
        font-weight: bold;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #666;
        text-align: center;
        margin-bottom: 2rem;
    }
    .prediction-box {
        background-color: #E8F5E9;
        padding: 2rem;
        border-radius: 10px;
        border-left: 5px solid #2E7D32;
        margin: 1rem 0;
    }
    .prediction-value {
        font-size: 3rem;
        color: #1B5E20;
        font-weight: bold;
        text-align: center;
    }
    .info-box {
        background-color: #E3F2FD;
        padding: 1rem;
        border-radius: 5px;
        border-left: 4px solid #1976D2;
        margin: 1rem 0;
        color: #1a1a1a;
    }
    .info-box strong {
        color: #0d47a1;
    }
    .warning-box {
        background-color: #FFF3E0;
        padding: 1rem;
        border-radius: 5px;
        border-left: 4px solid #F57C00;
        margin: 1rem 0;
        color: #1a1a1a;
    }
    .warning-box strong {
        color: #e65100;
    }
    </style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_model_artifacts():
    """
    Load the trained model, scaler, and feature names.
    Uses caching to load only once.
    
    Returns:
        tuple: (model, scaler, feature_names)
    """
    try:
        # Adjust path based on where artifacts are stored
        # Development: ../model/outputs/
        # Docker: /app/model/outputs/
        app_dir = os.path.dirname(__file__)
        
        # Try Docker path first
        model_dir = '/app/model/outputs'
        if not os.path.exists(model_dir):
            # Fall back to development path
            model_dir = os.path.join(app_dir, '..', 'model', 'outputs')
        
        # Load model artifacts
        model = joblib.load(os.path.join(model_dir, 'model.joblib'))
        scaler = joblib.load(os.path.join(model_dir, 'scaler.joblib'))
        feature_names = joblib.load(os.path.join(model_dir, 'feature_names.joblib'))
        
        return model, scaler, feature_names
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        st.info("Please train the model first using: python train_model.py")
        st.stop()


def calculate_derived_features(offered_qty, sold_qty, selected_date):
    """
    Calculate derived features from user inputs.
    
    Args:
        offered_qty (float): Offered quantity
        sold_qty (float): Sold quantity
        selected_date (datetime): Selected date
        
    Returns:
        dict: Dictionary of derived features
    """
    # Calculate sales ratio
    sales_ratio = sold_qty / offered_qty if offered_qty > 0 else 0
    sales_ratio = min(sales_ratio, 1.0)  # Cap at 1.0
    
    # Calculate unsold quantity
    unsold_qty = offered_qty - sold_qty
    
    # Extract date features
    year = selected_date.year
    month = selected_date.month
    quarter = (month - 1) // 3 + 1
    
    return {
        'Sales_Ratio': sales_ratio,
        'Unsold_Quantity': unsold_qty,
        'Year': year,
        'Month': month,
        'Quarter': quarter
    }


def prepare_features(user_inputs, feature_names):
    """
    Prepare features in the correct order for model prediction.
    
    Args:
        user_inputs (dict): User input values
        feature_names (list): List of feature names in correct order
        
    Returns:
        pd.DataFrame: Features ready for prediction
    """
    # Create feature dictionary in correct order
    features = {}
    for feature in feature_names:
        if feature in user_inputs:
            features[feature] = user_inputs[feature]
        else:
            st.error(f"Missing feature: {feature}")
            return None
    
    # Convert to DataFrame
    df = pd.DataFrame([features])
    return df


def generate_shap_explanation(model, features_df, feature_names):
    """
    Generate SHAP explanation for the prediction.
    
    Args:
        model: Trained model
        features_df (pd.DataFrame): Features for prediction
        feature_names (list): List of feature names
        
    Returns:
        tuple: (waterfall_fig, summary_fig, shap_values, expected_value)
    """
    if not HAS_SHAP:
        raise ImportError("SHAP is not installed. Install it with: pip install shap")
    
    # Create SHAP explainer
    explainer = shap.TreeExplainer(model)
    
    # Calculate SHAP values
    shap_values = explainer.shap_values(features_df)
    expected_value = explainer.expected_value
    
    # Create waterfall plot (shows how each feature contributes to the prediction)
    waterfall_fig = plt.figure(figsize=(14, 10))
    shap.waterfall_plot(
        shap.Explanation(
            values=shap_values[0],
            base_values=expected_value,
            data=features_df.iloc[0].values,
            feature_names=feature_names
        ),
        show=False,
        max_display=10
    )
    # Adjust layout to prevent text overlap
    plt.subplots_adjust(left=0.25, right=0.95, top=0.95, bottom=0.1)
    plt.tight_layout(pad=2.0)
    
    # Create summary bar plot (overall feature importance)
    summary_fig = plt.figure(figsize=(12, 7))
    shap.summary_plot(
        shap_values,
        features_df,
        feature_names=feature_names,
        plot_type="bar",
        show=False
    )
    plt.tight_layout()
    
    return waterfall_fig, summary_fig, shap_values, expected_value


def predict_multiple_weeks(model, scaler, feature_names, base_inputs, num_weeks=4):
    """
    Predict prices for multiple future weeks using rolling predictions.
    
    Args:
        model: Trained model
        scaler: Fitted scaler
        feature_names: List of feature names
        base_inputs: Dictionary of base input values
        num_weeks: Number of weeks to predict
        
    Returns:
        pd.DataFrame: Predictions with dates
    """
    predictions = []
    current_date = base_inputs['date']
    prev_price = base_inputs['Prev_Week_Price']
    
    for week in range(1, num_weeks + 1):
        # Update date (add 7 days per week)
        current_date = current_date + pd.Timedelta(days=7)
        year = current_date.year
        month = current_date.month
        quarter = (month - 1) // 3 + 1
        
        # Prepare features for this week
        week_inputs = base_inputs.copy()
        week_inputs['Year'] = year
        week_inputs['Month'] = month
        week_inputs['Quarter'] = quarter
        week_inputs['Prev_Week_Price'] = prev_price
        
        # Make prediction
        features_df = prepare_features(week_inputs, feature_names)
        features_scaled = scaler.transform(features_df)
        prediction = model.predict(features_scaled)[0]
        
        predictions.append({
            'Week': f'Week {week}',
            'Date': current_date.strftime('%Y-%m-%d'),
            'Predicted_Price': prediction,
            'Change_from_Previous': ((prediction - prev_price) / prev_price * 100) if prev_price > 0 else 0
        })
        
        # Update prev_price for next iteration (rolling prediction)
        prev_price = prediction
    
    return pd.DataFrame(predictions)


def main():
    """
    Main application function.
    """
    # Header
    st.markdown('<p class="main-header">🥥 Coconut Auction Price Predictor</p>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Predict coconut auction prices using AI with explainable predictions</p>', unsafe_allow_html=True)
    
    # About section
    with st.expander("ℹ️ About This Application", expanded=False):
        st.markdown("""
        ### What is this?
        This is an **AI-powered price prediction system** for Sri Lankan coconut auctions. It uses a machine learning model 
        (XGBoost) trained on historical auction data to forecast prices based on market conditions.
        
        ### Why does this exist?
        - **Help farmers & traders** make informed decisions about when to sell coconuts
        - **Provide price transparency** in the coconut auction market
        - **Predict market trends** using historical patterns and seasonal factors
        - **Support agricultural planning** with data-driven insights
        
        ### What can you do?
        1. **Enter auction parameters** (quantity offered, sold, date, exchange rate, etc.)
        2. **Get instant price predictions** for 1000 coconuts in LKR
        3. **Forecast multiple weeks** ahead (4-week or 8-week forecasts)
        4. **Understand predictions with SHAP analysis** - see exactly how each feature impacts the price
        5. **View interactive explanations** with waterfall plots and feature contributions
        6. **Analyze market factors** affecting coconut prices
        
        ### How it works
        The model analyzes 9 key features including:
        - Quantity metrics (offered, sold, unsold, sales ratio)
        - Time factors (year, month, quarter)
        - Currency exchange rates (USD/LKR)
        - Historical price trends (previous week's price)
        
        📊 **Model Performance:** Trained on 227 auction records with high accuracy
        """)
    
    st.markdown("---")
    
    # Load model artifacts
    with st.spinner("Loading model..."):
        model, scaler, feature_names = load_model_artifacts()
    
    st.success("✅ Model loaded successfully!")
    
    # Sidebar for inputs
    st.sidebar.header("📊 Input Parameters")
    st.sidebar.markdown("---")
    
    # Prediction mode
    st.sidebar.subheader("🔮 Prediction Mode")
    prediction_mode = st.sidebar.radio(
        "Select prediction type:",
        ["Single Week", "Multi-Week Forecast (4 weeks)", "Extended Forecast (8 weeks)"],
        help="Choose to predict one week or multiple future weeks"
    )
    
    st.sidebar.markdown("---")
    
    # Auction Date
    st.sidebar.subheader("📅 Auction Date")
    selected_date = st.sidebar.date_input(
        "Select auction date",
        value=datetime.now(),
        help="Select the date for prediction. Year, Month, and Quarter will be auto-extracted."
    )
    
    st.sidebar.markdown("---")
    
    # Auction Quantities
    st.sidebar.subheader("📦 Auction Quantities")
    
    offered_quantity = st.sidebar.number_input(
        "Offered Quantity",
        min_value=0,
        max_value=3000000,
        value=600000,
        step=10000,
        help="Total quantity of coconuts offered at auction (in units)"
    )
    
    sold_quantity = st.sidebar.number_input(
        "Sold Quantity",
        min_value=0,
        max_value=int(offered_quantity),
        value=int(offered_quantity * 0.75),
        step=10000,
        help="Total quantity of coconuts sold at auction (must be ≤ offered quantity)"
    )
    
    # Validate sold quantity
    if sold_quantity > offered_quantity:
        st.sidebar.error("❌ Sold quantity cannot exceed offered quantity!")
    
    st.sidebar.markdown("---")
    
    # Previous Week's Price
    st.sidebar.subheader("💰 Historical Price")
    
    st.sidebar.markdown("""
    <div class="info-box">
        <strong>ℹ️ Previous Week's Price:</strong><br>
        Enter last week's auction price. This is crucial as recent prices 
        strongly influence current market rates.
    </div>
    """, unsafe_allow_html=True)
    
    prev_week_price = st.sidebar.number_input(
        "Previous Week's Price (Rs/1000)",
        min_value=0.0,
        max_value=500000.0,
        value=120000.0,
        step=1000.0,
        help="Average price from last week's auction (in Rs per 1000 coconuts)"
    )
    
    st.sidebar.markdown("---")
    
    # Exchange Rate
    st.sidebar.subheader("💵 Exchange Rate")
    
    usd_avg_rate = st.sidebar.number_input(
        "USD Average Rate (LKR)",
        min_value=0.0,
        max_value=1000.0,
        value=305.0,
        step=0.5,
        help="Current USD to LKR exchange rate (average of buy and sell rates)"
    )
    
    st.sidebar.markdown("---")
    
    # Predict button
    predict_button = st.sidebar.button("🔮 Predict Price", type="primary", use_container_width=True)
    
    # Main content area
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("📋 Input Summary")
        
        # Calculate derived features
        derived = calculate_derived_features(offered_quantity, sold_quantity, selected_date)
        
        # Display input summary
        summary_data = {
            "Parameter": [
                "Auction Date",
                "Offered Quantity",
                "Sold Quantity",
                "Sales Ratio",
                "Unsold Quantity",
                "Previous Week Price",
                "USD Exchange Rate",
                "Year",
                "Month",
                "Quarter"
            ],
            "Value": [
                selected_date.strftime("%Y-%m-%d"),
                f"{offered_quantity:,}",
                f"{sold_quantity:,}",
                f"{derived['Sales_Ratio']:.2%}",
                f"{derived['Unsold_Quantity']:,}",
                f"Rs {prev_week_price:,.2f}",
                f"Rs {usd_avg_rate:.2f}",
                str(derived['Year']),
                str(derived['Month']),
                str(derived['Quarter'])
            ]
        }
        
        summary_df = pd.DataFrame(summary_data)
        st.dataframe(summary_df, use_container_width=True, hide_index=True)
        
        # Auto-calculated features info
        st.markdown("""
        <div class="info-box">
            <strong>ℹ️ Auto-Calculated Features:</strong><br>
            • <strong>Sales Ratio</strong> = Sold / Offered<br>
            • <strong>Unsold Quantity</strong> = Offered - Sold<br>
            • <strong>Year, Month, Quarter</strong> extracted from selected date
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.subheader("🎯 Prediction Results")
        
        if predict_button:
            if sold_quantity > offered_quantity:
                st.error("❌ Please correct the quantities: Sold cannot exceed Offered!")
            else:
                with st.spinner("Making prediction..."):
                    try:
                        # Prepare all features
                        user_inputs = {
                            'Offered_Quantity': offered_quantity,
                            'Sold_Quantity': sold_quantity,
                            'Sales_Ratio': derived['Sales_Ratio'],
                            'Unsold_Quantity': derived['Unsold_Quantity'],
                            'Year': derived['Year'],
                            'Month': derived['Month'],
                            'Quarter': derived['Quarter'],
                            'USD_Avg_Rate': usd_avg_rate,
                            'Prev_Week_Price': prev_week_price,
                            'date': pd.Timestamp(selected_date)
                        }
                        
                        # Prepare features dataframe (needed for both prediction and SHAP)
                        features_df = prepare_features(user_inputs, feature_names)
                        
                        # Determine number of weeks based on mode
                        if "4 weeks" in prediction_mode:
                            num_weeks = 4
                        elif "8 weeks" in prediction_mode:
                            num_weeks = 8
                        else:
                            num_weeks = 1
                        
                        if num_weeks == 1:
                            # Single week prediction
                            if features_df is not None:
                                # Scale features
                                features_scaled = scaler.transform(features_df)
                                
                                # Make prediction
                                prediction = model.predict(features_scaled)[0]
                                
                                # Display prediction
                                st.markdown("""
                                <div class="prediction-box">
                                    <p style="text-align: center; margin: 0; color: #666;">Predicted Price</p>
                                    <p class="prediction-value">Rs {:.2f}</p>
                                    <p style="text-align: center; margin: 0; color: #666; font-size: 1.1rem;">per 1000 coconuts</p>
                                </div>
                                """.format(prediction), unsafe_allow_html=True)
                                
                                # Price insights
                                if prev_week_price > 0:
                                    price_change = ((prediction - prev_week_price) / prev_week_price) * 100
                                    
                                    if price_change > 0:
                                        st.success(f"📈 Price increased by {price_change:.2f}% from last week")
                                    elif price_change < 0:
                                        st.info(f"📉 Price decreased by {abs(price_change):.2f}% from last week")
                                    else:
                                        st.info("➡️ Price remains stable from last week")
                        
                        else:
                            # Multi-week forecast
                            forecast_df = predict_multiple_weeks(model, scaler, feature_names, user_inputs, num_weeks)
                            
                            # Display forecast summary
                            st.markdown(f"""
                            <div class="prediction-box">
                                <p style="text-align: center; margin: 0; color: #666; font-size: 1.2rem;">
                                    {num_weeks}-Week Price Forecast
                                </p>
                                <p class="prediction-value">Rs {forecast_df['Predicted_Price'].iloc[0]:.2f}</p>
                                <p style="text-align: center; margin: 0; color: #666;">Next Week (Week 1)</p>
                            </div>
                            """, unsafe_allow_html=True)
                            
                            # Show forecast table
                            st.subheader("📊 Detailed Forecast")
                            display_df = forecast_df.copy()
                            display_df['Predicted_Price'] = display_df['Predicted_Price'].apply(lambda x: f"Rs {x:,.2f}")
                            display_df['Change_from_Previous'] = display_df['Change_from_Previous'].apply(lambda x: f"{x:+.2f}%")
                            st.dataframe(display_df, use_container_width=True, hide_index=True)
                            
                            # Plot forecast trend
                            st.subheader("📈 Price Trend Forecast")
                            fig, ax = plt.subplots(figsize=(12, 6))
                            
                            weeks = list(range(0, num_weeks + 1))
                            prices = [prev_week_price] + forecast_df['Predicted_Price'].tolist()
                            
                            ax.plot(weeks, prices, marker='o', linewidth=2, markersize=8, color='#2E7D32')
                            ax.fill_between(weeks, prices, alpha=0.3, color='#4CAF50')
                            
                            ax.set_xlabel('Week', fontsize=12, fontweight='bold')
                            ax.set_ylabel('Price (Rs/1000)', fontsize=12, fontweight='bold')
                            ax.set_title(f'{num_weeks}-Week Price Forecast', fontsize=14, fontweight='bold', pad=20)
                            ax.grid(True, alpha=0.3, linestyle='--')
                            
                            # Add value labels on points
                            for i, (week, price) in enumerate(zip(weeks, prices)):
                                label = "Current" if week == 0 else f"Week {week}"
                                ax.annotate(f'Rs {price:,.0f}', 
                                           xy=(week, price), 
                                           xytext=(0, 10),
                                           textcoords='offset points',
                                           ha='center',
                                           fontweight='bold',
                                           fontsize=9)
                            
                            plt.tight_layout()
                            st.pyplot(fig)
                            
                            # Forecast insights
                            avg_price = forecast_df['Predicted_Price'].mean()
                            max_price = forecast_df['Predicted_Price'].max()
                            min_price = forecast_df['Predicted_Price'].min()
                            max_week = forecast_df.loc[forecast_df['Predicted_Price'].idxmax(), 'Week']
                            min_week = forecast_df.loc[forecast_df['Predicted_Price'].idxmin(), 'Week']
                            
                            col_a, col_b, col_c = st.columns(3)
                            with col_a:
                                st.metric("Average Price", f"Rs {avg_price:,.2f}")
                            with col_b:
                                st.metric("Highest Price", f"Rs {max_price:,.2f}", delta=f"{max_week}")
                            with col_c:
                                st.metric("Lowest Price", f"Rs {min_price:,.2f}", delta=f"{min_week}")
                    
                    except Exception as e:
                        st.error(f"Prediction error: {str(e)}")
        else:
            st.info("👈 Click 'Predict Price' in the sidebar to get a prediction")
    
    # SHAP Analysis Section
    if predict_button and sold_quantity <= offered_quantity:
        st.markdown("---")
        st.subheader("🔍 SHAP Analysis - Explainable AI")
        
        if not HAS_SHAP:
            st.warning("""
            ⚠️ **SHAP is not installed.** To enable explainable AI features, install SHAP:
            ```
            pip install shap
            ```
            """)
        else:
            # Check if features_df exists (it should be created during prediction)
            try:
                if 'features_df' not in locals() or features_df is None:
                    st.warning("⚠️ Features not available. Please make a prediction first.")
                else:
                    st.markdown("""
                    <div class="info-box">
                        <strong>Understanding the Prediction with SHAP:</strong><br>
                        SHAP (SHapley Additive exPlanations) shows exactly how each feature contributed 
                        to this specific prediction. Red bars push the price higher, blue bars push it lower.
                    </div>
                    """, unsafe_allow_html=True)
                    
                    with st.spinner("Generating SHAP explanation..."):
                        try:
                            # Generate SHAP explanation
                            waterfall_fig, summary_fig, shap_values, expected_value = generate_shap_explanation(
                                model, features_df, feature_names
                            )
                            
                            # Display waterfall plot
                            st.subheader("📊 SHAP Waterfall Plot")
                            st.markdown("""
                            This chart shows how each feature **pushed the prediction** up or down from the baseline:
                            - **Baseline (E[f(X)])**: Average prediction across all data
                            - **Red bars**: Features that increased the price
                            - **Blue bars**: Features that decreased the price
                            - **f(x)**: Final prediction
                            """)
                            st.pyplot(waterfall_fig)
                            
                            # Display summary bar plot
                            st.markdown("---")
                            st.subheader("📈 Feature Importance (SHAP Values)")
                            st.markdown("""
                            Overall feature importance based on SHAP values - shows which features 
                            have the most impact on predictions:
                            """)
                            st.pyplot(summary_fig)
                            
                            # Feature contribution summary
                            col1, col2 = st.columns(2)
                            
                            with col1:
                                st.markdown("**🔴 Top Features Increasing Price:**")
                                top_positive = []
                                for i, (fname, sval) in enumerate(zip(feature_names, shap_values[0])):
                                    if sval > 0:
                                        top_positive.append((fname, sval))
                                top_positive.sort(key=lambda x: x[1], reverse=True)
                                
                                if top_positive:
                                    for fname, sval in top_positive[:3]:
                                        st.markdown(f"- **{fname}**: +Rs {abs(sval):,.2f}")
                                else:
                                    st.markdown("*No features increasing price*")
                            
                            with col2:
                                st.markdown("**🔵 Top Features Decreasing Price:**")
                                top_negative = []
                                for i, (fname, sval) in enumerate(zip(feature_names, shap_values[0])):
                                    if sval < 0:
                                        top_negative.append((fname, sval))
                                top_negative.sort(key=lambda x: x[1])
                                
                                if top_negative:
                                    for fname, sval in top_negative[:3]:
                                        st.markdown(f"- **{fname}**: Rs {sval:,.2f}")
                                else:
                                    st.markdown("*No features decreasing price*")
                            
                            # Display prediction breakdown
                            st.markdown("---")
                            st.markdown("**📊 Prediction Breakdown:**")
                            
                            prediction_breakdown = pd.DataFrame({
                                'Feature': feature_names,
                                'Value': features_df.iloc[0].values,
                                'SHAP Impact': shap_values[0],
                                'Direction': ['↑ Increase' if v > 0 else '↓ Decrease' if v < 0 else '→ Neutral' 
                                             for v in shap_values[0]]
                            })
                            prediction_breakdown = prediction_breakdown.sort_values('SHAP Impact', 
                                                                                    key=abs, 
                                                                                    ascending=False)
                            prediction_breakdown['SHAP Impact'] = prediction_breakdown['SHAP Impact'].apply(
                                lambda x: f"{x:+,.2f}"
                            )
                            
                            st.dataframe(prediction_breakdown, use_container_width=True, hide_index=True)
                            
                            st.info(f"""
                            **Baseline Price**: Rs {expected_value:,.2f}  
                            **Your Prediction**: Rs {expected_value + sum(shap_values[0]):,.2f}  
                            **Net SHAP Impact**: Rs {sum(shap_values[0]):+,.2f}
                            """)
                            
                        except Exception as e:
                            st.error(f"Could not generate SHAP explanation: {str(e)}")
                            st.info("Make sure SHAP is installed: `pip install shap`")
            except NameError:
                st.warning("⚠️ Features not available. Please make a prediction first.")
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #666; padding: 1rem;">
        <p><strong>Coconut Auction Price Predictor</strong> | Powered by XGBoost + SHAP</p>
        <p>ML Assignment 2026 | Built with Streamlit</p>
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
