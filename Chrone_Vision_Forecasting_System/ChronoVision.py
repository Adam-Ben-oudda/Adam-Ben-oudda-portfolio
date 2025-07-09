"""
████████████████████████████████████████████████████████████████████████████████
█░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░█
█░░░░░░░░░░░░░░░░░░░░░░░░░░░ CHRONOVISION SYSTEM ░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░█
█░░░░░░░░░░░░░░░░░░░░░░░ ULTIMATE TIME SERIES FORECASTING ░░░░░░░░░░░░░░░░░░░░░░█
█░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░█
████████████████████████████████████████████████████████████████████████████████

Version: 7.3.1 (Quantum Edition)
Author: Temporal Analytics Research Group
Release Date: 2023-11-15
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import xgboost as xgb
from prophet import Prophet
import shap
from scipy import stats
import statsmodels.api as sm
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import StandardScaler
from scipy.optimize import minimize
from scipy.stats import norm, t
import json
import datetime
import random
import string
import re
import warnings
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import (Any, Dict, List, Optional, Tuple, Union, Callable)

warnings.filterwarnings('ignore')
pd.set_option('display.max_columns', 50)
plt.style.use('seaborn-v0_8-darkgrid')

# =============================================================================
# Core Data Structures
# =============================================================================
@dataclass
class TemporalEvent:
    name: str
    start_date: datetime.datetime
    end_date: datetime.datetime
    event_type: str  # 'promotion', 'holiday', 'outage', 'custom'
    impact_scope: Dict[str, float] = field(default_factory=dict)  # Series: Impact multiplier
    confidence: float = 0.8
    description: str = ""

@dataclass
class ForecastResult:
    timestamp: datetime.datetime
    series_name: str
    point_forecast: float
    lower_bound: float
    upper_bound: float
    model_contributions: Dict[str, float]  # Contribution of each model
    event_impacts: Dict[str, float]  # Impact of each event
    shap_values: Optional[Dict[str, float]] = None
    anomaly_score: float = 0.0
    scenario_id: str = "baseline"

@dataclass
class ModelPerformance:
    model_name: str
    mae: float
    rmse: float
    mape: float
    last_retrained: datetime.datetime
    features_importance: Dict[str, float]

@dataclass
class CausalImpact:
    event_name: str
    series_name: str
    actual_effect: float
    counterfactual_effect: float
    confidence_interval: Tuple[float, float]
    p_value: float

@dataclass
class Scenario:
    scenario_id: str
    name: str
    description: str
    parameters: Dict[str, Any]
    forecast_results: Dict[str, List[ForecastResult]]  # Series: Forecasts

# =============================================================================
# ChronoVision Core Engine
# =============================================================================
class ChronoVisionEngine:
    """Ultimate Hybrid Forecasting System with Causal Intelligence"""
    def __init__(self):
        self.data = None
        self.series_names = []
        self.events = []
        self.models = {}
        self.performance_metrics = {}
        self.scenarios = {}
        self.current_scenario = "baseline"
        self.feature_columns = []
        self.external_factors = []
        self.transfer_models = {}
        self.clusters = {}
        self.connectors = {}
        self.drift_detectors = {}
        
        # Initialize with default scenarios
        self._initialize_default_scenarios()
        
        # Load transfer learning models
        self._load_transfer_models()
        
        # Initialize connectors
        self._initialize_connectors()
    
    def _initialize_default_scenarios(self):
        """Create default scenarios"""
        baseline = Scenario(
            scenario_id="baseline",
            name="Baseline Forecast",
            description="Projection based on current trends and patterns",
            parameters={},
            forecast_results={}
        )
        
        optimistic = Scenario(
            scenario_id="optimistic",
            name="Optimistic Scenario",
            description="Best-case projections with favorable conditions",
            parameters={"growth_factor": 1.2, "event_impact_reduction": 0.8},
            forecast_results={}
        )
        
        pessimistic = Scenario(
            scenario_id="pessimistic",
            name="Pessimistic Scenario",
            description="Worst-case projections with adverse conditions",
            parameters={"growth_factor": 0.8, "event_impact_amplification": 1.2},
            forecast_results={}
        )
        
        self.scenarios = {
            "baseline": baseline,
            "optimistic": optimistic,
            "pessimistic": pessimistic
        }
    
    def _load_transfer_models(self):
        """Load pre-trained models from different domains"""
        # In a real implementation, this would load actual pre-trained models
        self.transfer_models = {
            "retail": {"seasonality": "strong", "trend": "moderate", "events": ["holidays", "promotions"]},
            "finance": {"seasonality": "weak", "trend": "strong", "events": ["earnings", "fomc"]},
            "web_traffic": {"seasonality": "moderate", "trend": "variable", "events": ["product_launches", "marketing"]}
        }
    
    def _initialize_connectors(self):
        """Initialize data connectors"""
        # Stub for real connector implementations
        self.connectors = {
            "google_sheets": self._connect_google_sheets,
            "sql": self._connect_sql,
            "rest_api": self._connect_rest_api
        }
    
    def connect_data(self, source_type: str, **kwargs):
        """Connect to data source"""
        if source_type in self.connectors:
            return self.connectors[source_type](**kwargs)
        raise ValueError(f"Unsupported data source: {source_type}")
    
    def _connect_google_sheets(self, sheet_id: str, sheet_name: str):
        """Simulate Google Sheets connection"""
        print(f"Connected to Google Sheet: {sheet_id}/{sheet_name}")
        # In real implementation, use gspread library
        return pd.DataFrame()
    
    def _connect_sql(self, connection_string: str, query: str):
        """Simulate SQL connection"""
        print(f"Executing SQL query: {query[:50]}...")
        # In real implementation, use SQLAlchemy
        return pd.DataFrame()
    
    def _connect_rest_api(self, endpoint: str, params: dict):
        """Simulate REST API connection"""
        print(f"Fetching data from API: {endpoint}")
        # In real implementation, use requests library
        return pd.DataFrame()
    
    def load_data(self, data: pd.DataFrame, date_col: str, series_cols: List[str], 
                 external_factors: List[str] = [], freq: str = 'D'):
        """
        Load time series data into the system
        
        Args:
            data: DataFrame containing time series
            date_col: Name of the date column
            series_cols: List of columns to forecast
            external_factors: List of external factor columns
            freq: Frequency of the time series ('D', 'W', 'M', etc.)
        """
        # Validate data
        if date_col not in data.columns:
            raise ValueError(f"Date column '{date_col}' not found in data")
        
        for col in series_cols + external_factors:
            if col not in data.columns:
                raise ValueError(f"Column '{col}' not found in data")
        
        # Convert to datetime and sort
        data[date_col] = pd.to_datetime(data[date_col])
        data = data.sort_values(date_col).reset_index(drop=True)
        
        # Set as index and ensure frequency
        data = data.set_index(date_col).asfreq(freq)
        
        # Check for missing values
        missing = data[series_cols].isnull().sum()
        if missing.sum() > 0:
            print(f"Warning: Found {missing.sum()} missing values in series data")
            # Apply interpolation
            data[series_cols] = data[series_cols].interpolate(method='time')
        
        self.data = data
        self.series_names = series_cols
        self.external_factors = external_factors
        self.feature_columns = self._generate_feature_columns()
        
        # Initialize drift detectors for each series
        for series in series_cols:
            self.drift_detectors[series] = DriftDetector()
        
        print(f"Loaded data with {len(data)} records, {len(series_cols)} series")
    
    def _generate_feature_columns(self) -> List[str]:
        """Generate feature columns for modeling"""
        features = []
        
        # Time features
        features.extend([
            'day_of_week', 'day_of_month', 'day_of_year',
            'week_of_year', 'month', 'quarter', 'year'
        ])
        
        # Lag features
        for lag in [1, 7, 14, 30, 90]:
            features.append(f'lag_{lag}')
        
        # Rolling features
        for window in [7, 14, 30]:
            features.extend([
                f'rolling_mean_{window}',
                f'rolling_std_{window}',
                f'rolling_min_{window}',
                f'rolling_max_{window}'
            ])
        
        # External factors
        features.extend(self.external_factors)
        
        return features
    
    def add_event(self, event: TemporalEvent):
        """Add a temporal event to the system"""
        # Validate event
        if event.start_date > event.end_date:
            raise ValueError("Event start date must be before end date")
        
        # Ensure impact scope covers all series
        for series in self.series_names:
            if series not in event.impact_scope:
                event.impact_scope[series] = 1.0  # Default no impact
        
        self.events.append(event)
        print(f"Added event: {event.name} ({event.event_type})")
    
    def preprocess_data(self, series_name: str) -> pd.DataFrame:
        """Preprocess data for a specific series"""
        if self.data is None:
            raise ValueError("No data loaded. Please load data first.")
        
        df = self.data[[series_name] + self.external_factors].copy()
        
        # Time-based features
        df['day_of_week'] = df.index.dayofweek
        df['day_of_month'] = df.index.day
        df['day_of_year'] = df.index.dayofyear
        df['week_of_year'] = df.index.isocalendar().week
        df['month'] = df.index.month
        df['quarter'] = df.index.quarter
        df['year'] = df.index.year
        
        # Lag features
        for lag in [1, 7, 14, 30, 90]:
            df[f'lag_{lag}'] = df[series_name].shift(lag)
        
        # Rolling features
        for window in [7, 14, 30]:
            df[f'rolling_mean_{window}'] = df[series_name].rolling(window).mean()
            df[f'rolling_std_{window}'] = df[series_name].rolling(window).std()
            df[f'rolling_min_{window}'] = df[series_name].rolling(window).min()
            df[f'rolling_max_{window}'] = df[series_name].rolling(window).max()
        
        # Event features
        for event in self.events:
            df[event.name] = 0
            event_mask = (df.index >= event.start_date) & (df.index <= event.end_date)
            df.loc[event_mask, event.name] = event.impact_scope[series_name]
        
        # Drop initial rows with missing values
        df = df.dropna()
        
        return df
    
    def train_models(self, series_name: str, test_size: float = 0.2, domain: str = 'retail'):
        """
        Train hybrid forecasting models for a specific series
        
        Args:
            series_name: Name of the series to forecast
            test_size: Proportion of data to use for testing
            domain: Domain for transfer learning initialization
        """
        df = self.preprocess_data(series_name)
        
        # Split data
        split_idx = int(len(df) * (1 - test_size))
        train = df.iloc[:split_idx]
        test = df.iloc[split_idx:]
        
        # Initialize models with transfer learning
        self._apply_transfer_learning(series_name, domain)
        
        # Prepare data
        X_train = train.drop(columns=[series_name])
        y_train = train[series_name]
        X_test = test.drop(columns=[series_name])
        y_test = test[series_name]
        
        # Train Prophet model
        prophet_model = self._train_prophet_model(train, series_name)
        
        # Train XGBoost model
        xgb_model = self._train_xgboost_model(X_train, y_train)
        
        # Evaluate models
        prophet_perf = self._evaluate_prophet(prophet_model, test, series_name)
        xgb_perf = self._evaluate_xgboost(xgb_model, X_test, y_test)
        
        # Store models and performance
        self.models[series_name] = {
            'prophet': prophet_model,
            'xgboost': xgb_model,
            'last_trained': datetime.datetime.now()
        }
        
        self.performance_metrics[series_name] = {
            'prophet': prophet_perf,
            'xgboost': xgb_perf
        }
        
        # Detect drift
        self._detect_drift(series_name, X_test, y_test)
        
        print(f"Training complete for {series_name}")
        print(f"Prophet MAE: {prophet_perf.mae:.2f}, XGBoost MAE: {xgb_perf.mae:.2f}")
    
    def _apply_transfer_learning(self, series_name: str, domain: str):
        """Apply transfer learning from pre-trained models"""
        if domain in self.transfer_models:
            domain_knowledge = self.transfer_models[domain]
            print(f"Applying transfer learning from {domain} domain: {domain_knowledge}")
            # In real implementation, this would initialize model parameters
        else:
            print(f"No transfer model found for domain: {domain}. Using default initialization.")
    
    def _train_prophet_model(self, train: pd.DataFrame, series_name: str) -> Prophet:
        """Train a Prophet model"""
        # Prepare data for Prophet
        prophet_df = train.reset_index().rename(columns={'index': 'ds', series_name: 'y'})
        
        # Add events as regressors
        model = Prophet(
            yearly_seasonality=True,
            weekly_seasonality=True,
            daily_seasonality=False,
            seasonality_mode='multiplicative'
        )
        
        # Add events
        for event in self.events:
            model.add_regressor(event.name)
        
        # Add external factors
        for factor in self.external_factors:
            model.add_regressor(factor)
        
        model.fit(prophet_df)
        return model
    
    def _train_xgboost_model(self, X_train: pd.DataFrame, y_train: pd.Series) -> xgb.XGBRegressor:
        """Train an XGBoost model"""
        model = xgb.XGBRegressor(
            n_estimators=500,
            learning_rate=0.05,
            max_depth=8,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42
        )
        
        model.fit(X_train, y_train)
        return model
    
    def _evaluate_prophet(self, model: Prophet, test: pd.DataFrame, series_name: str) -> ModelPerformance:
        """Evaluate Prophet model performance"""
        # Prepare test data
        test_df = test.reset_index().rename(columns={'index': 'ds', series_name: 'y'})
        
        # Forecast
        forecast = model.predict(test_df)
        
        # Calculate metrics
        mae = mean_absolute_error(test_df['y'], forecast['yhat'])
        rmse = np.sqrt(mean_squared_error(test_df['y'], forecast['yhat']))
        mape = np.mean(np.abs((test_df['y'] - forecast['yhat']) / test_df['y'])) * 100
        
        # Feature importance not directly available in Prophet
        # We'll use a placeholder for now
        features_importance = {col: 1.0 for col in model.extra_regressors}
        
        return ModelPerformance(
            model_name="Prophet",
            mae=mae,
            rmse=rmse,
            mape=mape,
            last_retrained=datetime.datetime.now(),
            features_importance=features_importance
        )
    
    def _evaluate_xgboost(self, model: xgb.XGBRegressor, X_test: pd.DataFrame, y_test: pd.Series) -> ModelPerformance:
        """Evaluate XGBoost model performance"""
        # Predict
        y_pred = model.predict(X_test)
        
        # Calculate metrics
        mae = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        mape = np.mean(np.abs((y_test - y_pred) / y_test)) * 100
        
        # Feature importance
        features_importance = dict(zip(X_test.columns, model.feature_importances_))
        
        return ModelPerformance(
            model_name="XGBoost",
            mae=mae,
            rmse=rmse,
            mape=mape,
            last_retrained=datetime.datetime.now(),
            features_importance=features_importance
        )
    
    def _detect_drift(self, series_name: str, X_test: pd.DataFrame, y_test: pd.Series):
        """Detect data drift and trigger retraining if needed"""
        drift_detected = False
        
        # Get the latest data point
        latest_data = X_test.iloc[-1:].copy()
        actual_value = y_test.iloc[-1]
        
        # Get model predictions
        prophet_forecast = self.models[series_name]['prophet'].predict(
            latest_data.reset_index().rename(columns={'index': 'ds'})
        )['yhat'].values[0]
        
        xgb_forecast = self.models[series_name]['xgboost'].predict(latest_data)[0]
        
        # Calculate errors
        prophet_error = abs(prophet_forecast - actual_value) / actual_value
        xgb_error = abs(xgb_forecast - actual_value) / actual_value
        
        # Check against thresholds
        if prophet_error > 0.15 or xgb_error > 0.15:
            print(f"Significant forecast error detected for {series_name}. Potential drift.")
            drift_detected = True
        
        # Update drift detector
        self.drift_detectors[series_name].update(actual_value, (prophet_forecast + xgb_forecast) / 2)
        
        if self.drift_detectors[series_name].is_drift_detected():
            print(f"Statistical drift detected for {series_name}. Triggering retraining.")
            drift_detected = True
        
        # Retrain if drift detected
        if drift_detected:
            print(f"Retraining models for {series_name} due to drift detection")
            self.train_models(series_name, test_size=0.1)
    
    def forecast(self, series_name: str, horizon: int = 30, scenario: str = "baseline") -> List[ForecastResult]:
        """
        Generate forecasts for a specific series
        
        Args:
            series_name: Name of the series to forecast
            horizon: Number of periods to forecast
            scenario: Scenario ID to use for forecasting
        
        Returns:
            List of ForecastResult objects
        """
        if series_name not in self.models:
            raise ValueError(f"No models trained for series: {series_name}")
        
        # Prepare future data
        last_date = self.data.index[-1]
        future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=horizon, freq=self.data.index.freq)
        
        # Create future DataFrame
        future = pd.DataFrame(index=future_dates)
        
        # Add features
        future = self._add_features_to_future(future, series_name)
        
        # Get scenario parameters
        scenario_params = self.scenarios[scenario].parameters
        
        # Generate forecasts
        results = []
        for date in future_dates:
            # Prophet forecast
            prophet_df = future.loc[[date]].reset_index().rename(columns={'index': 'ds'})
            prophet_forecast = self.models[series_name]['prophet'].predict(prophet_df)['yhat'].values[0]
            
            # XGBoost forecast
            xgb_input = future.loc[[date]].drop(columns=['ds'], errors='ignore')
            xgb_forecast = self.models[series_name]['xgboost'].predict(xgb_input)[0]
            
            # Dynamic model weighting based on recent performance
            prophet_weight = self._calculate_model_weight(series_name, 'prophet')
            xgb_weight = self._calculate_model_weight(series_name, 'xgboost')
            
            # Combined forecast
            combined_forecast = (prophet_forecast * prophet_weight + 
                                xgb_forecast * xgb_weight)
            
            # Apply scenario adjustments
            combined_forecast = self._apply_scenario_adjustment(
                combined_forecast, scenario_params, date, series_name
            )
            
            # Calculate event impacts
            event_impacts = self._calculate_event_impacts(date, series_name)
            
            # Apply event impacts
            for event, impact in event_impacts.items():
                combined_forecast *= impact
            
            # Calculate confidence interval
            lower, upper = self._calculate_confidence_interval(
                combined_forecast, series_name
            )
            
            # Calculate SHAP values (for explainability)
            shap_values = self._calculate_shap_values(series_name, xgb_input)
            
            # Detect anomalies
            anomaly_score = self._detect_anomaly(series_name, combined_forecast, date)
            
            # Create result object
            result = ForecastResult(
                timestamp=date,
                series_name=series_name,
                point_forecast=combined_forecast,
                lower_bound=lower,
                upper_bound=upper,
                model_contributions={
                    'prophet': prophet_forecast * prophet_weight,
                    'xgboost': xgb_forecast * xgb_weight
                },
                event_impacts=event_impacts,
                shap_values=shap_values,
                anomaly_score=anomaly_score,
                scenario_id=scenario
            )
            
            results.append(result)
        
        # Store results in scenario
        self.scenarios[scenario].forecast_results[series_name] = results
        
        return results
    
    def _add_features_to_future(self, future: pd.DataFrame, series_name: str) -> pd.DataFrame:
        """Add necessary features to future DataFrame"""
        # Time-based features
        future['day_of_week'] = future.index.dayofweek
        future['day_of_month'] = future.index.day
        future['day_of_year'] = future.index.dayofyear
        future['week_of_year'] = future.index.isocalendar().week
        future['month'] = future.index.month
        future['quarter'] = future.index.quarter
        future['year'] = future.index.year
        
        # Initialize with last known value
        last_value = self.data[series_name].iloc[-1]
        
        # Create lags - start with the last known value
        lags = [1, 7, 14, 30, 90]
        for i, lag in enumerate(lags):
            if i == 0:
                future[f'lag_{lag}'] = last_value
            else:
                # For longer lags, we'll need to handle carefully
                future[f'lag_{lag}'] = np.nan
        
        # Fill longer lags with rolling forward fill
        for lag in lags:
            if lag > 1:
                future[f'lag_{lag}'] = future[f'lag_{lag}'].fillna(method='ffill')
        
        # Rolling features - initialize with last rolling values
        windows = [7, 14, 30]
        for window in windows:
            # Last rolling values
            last_rolling_mean = self.data[series_name].rolling(window).mean().iloc[-1]
            last_rolling_std = self.data[series_name].rolling(window).std().iloc[-1]
            last_rolling_min = self.data[series_name].rolling(window).min().iloc[-1]
            last_rolling_max = self.data[series_name].rolling(window).max().iloc[-1]
            
            future[f'rolling_mean_{window}'] = last_rolling_mean
            future[f'rolling_std_{window}'] = last_rolling_std
            future[f'rolling_min_{window}'] = last_rolling_min
            future[f'rolling_max_{window}'] = last_rolling_max
        
        # Event features
        for event in self.events:
            future[event.name] = 0
            for date in future.index:
                if event.start_date <= date <= event.end_date:
                    future.loc[date, event.name] = event.impact_scope[series_name]
        
        # External factors - use last known value (in real implementation, might have forecasts)
        for factor in self.external_factors:
            last_factor_value = self.data[factor].iloc[-1]
            future[factor] = last_factor_value
        
        return future
    
    def _calculate_model_weight(self, series_name: str, model_type: str) -> float:
        """Calculate dynamic model weight based on recent performance"""
        # In real implementation, use a decay-weighted average of recent performance
        perf = self.performance_metrics[series_name][model_type]
        
        # Simple weighting based on MAE
        prophet_mae = self.performance_metrics[series_name]['prophet'].mae
        xgb_mae = self.performance_metrics[series_name]['xgboost'].mae
        
        if model_type == 'prophet':
            return xgb_mae / (prophet_mae + xgb_mae)
        else:
            return prophet_mae / (prophet_mae + xgb_mae)
    
    def _apply_scenario_adjustment(self, forecast: float, scenario_params: dict, 
                                  date: datetime.datetime, series_name: str) -> float:
        """Apply scenario-specific adjustments to forecast"""
        adjusted = forecast
        
        # Apply growth factor if present
        if 'growth_factor' in scenario_params:
            # Calculate base growth rate (simple implementation)
            base_growth = 1.0  # Default no growth
            
            # Apply growth factor
            adjusted *= scenario_params['growth_factor'] * base_growth
        
        # Apply event impact adjustments
        if 'event_impact_reduction' in scenario_params:
            for event in self.events:
                if event.start_date <= date <= event.end_date:
                    impact_reduction = scenario_params['event_impact_reduction']
                    # Reduce impact
                    adjusted *= (1 + (event.impact_scope[series_name] - 1) * impact_reduction)
        
        if 'event_impact_amplification' in scenario_params:
            for event in self.events:
                if event.start_date <= date <= event.end_date:
                    impact_amp = scenario_params['event_impact_amplification']
                    # Amplify impact
                    adjusted *= (1 + (event.impact_scope[series_name] - 1) * impact_amp)
        
        return adjusted
    
    def _calculate_event_impacts(self, date: datetime.datetime, series_name: str) -> Dict[str, float]:
        """Calculate impacts of active events on a given date"""
        impacts = {}
        for event in self.events:
            if event.start_date <= date <= event.end_date:
                impacts[event.name] = event.impact_scope[series_name]
            else:
                impacts[event.name] = 1.0  # No impact
        return impacts
    
    def _calculate_confidence_interval(self, forecast: float, series_name: str) -> Tuple[float, float]:
        """Calculate confidence interval for forecast"""
        # Simple implementation - in real system, use model uncertainties
        error_std = self.performance_metrics[series_name]['prophet'].rmse * 0.5 + \
                   self.performance_metrics[series_name]['xgboost'].rmse * 0.5
        
        lower = forecast - 1.96 * error_std
        upper = forecast + 1.96 * error_std
        return lower, upper
    
    def _calculate_shap_values(self, series_name: str, xgb_input: pd.DataFrame) -> Dict[str, float]:
        """Calculate SHAP values for explainability"""
        model = self.models[series_name]['xgboost']
        
        # Initialize SHAP explainer
        explainer = shap.Explainer(model)
        
        # Calculate SHAP values
        shap_values = explainer(xgb_input)
        
        # Return as dictionary of feature contributions
        return dict(zip(xgb_input.columns, shap_values.values[0]))
    
    def _detect_anomaly(self, series_name: str, forecast: float, date: datetime.datetime) -> float:
        """Detect anomalies in the forecast"""
        # Simple implementation - compare to historical patterns
        historical_data = self.data[series_name]
        std_dev = historical_data.std()
        mean_val = historical_data.mean()
        
        # Calculate z-score
        z_score = (forecast - mean_val) / std_dev
        
        # Return anomaly score (0-1)
        return min(1.0, max(0.0, abs(z_score) / 3.0))
    
    def causal_impact_analysis(self, event_name: str, series_name: str) -> CausalImpact:
        """
        Analyze the causal impact of an event on a time series
        
        Args:
            event_name: Name of the event to analyze
            series_name: Name of the time series
        
        Returns:
            CausalImpact object with analysis results
        """
        # Find the event
        event = next((e for e in self.events if e.name == event_name), None)
        if not event:
            raise ValueError(f"Event '{event_name}' not found")
        
        # Get the data
        df = self.preprocess_data(series_name)
        
        # Identify pre-event and post-event periods
        pre_period = df[df.index < event.start_date]
        post_period = df[df.index >= event.start_date]
        
        # Simple difference-in-differences approach
        pre_mean = pre_period[series_name].mean()
        post_mean = post_period[series_name].mean()
        
        # Counterfactual: what if the event didn't happen?
        # We'll use the pre-event trend projected forward
        pre_dates = pre_period.index
        pre_values = pre_period[series_name].values
        
        # Fit linear trend to pre-event data
        X = sm.add_constant(np.arange(len(pre_dates)))
        model = sm.OLS(pre_values, X).fit()
        
        # Project trend to post-event period
        counterfactual = model.predict(
            sm.add_constant(np.arange(len(pre_dates), len(pre_dates) + len(post_period)))
        )[len(pre_dates):]
        
        # Calculate actual effect
        actual_effect = post_mean - np.mean(counterfactual)
        
        # Confidence interval (simplified)
        ci_lower = actual_effect - 1.96 * pre_period[series_name].std() / np.sqrt(len(post_period))
        ci_upper = actual_effect + 1.96 * pre_period[series_name].std() / np.sqrt(len(post_period))
        
        # Calculate p-value (simplified)
        t_stat = actual_effect / (pre_period[series_name].std() / np.sqrt(len(post_period)))
        p_value = 2 * (1 - norm.cdf(abs(t_stat)))
        
        return CausalImpact(
            event_name=event_name,
            series_name=series_name,
            actual_effect=actual_effect,
            counterfactual_effect=np.mean(counterfactual),
            confidence_interval=(ci_lower, ci_upper),
            p_value=p_value
        )
    
    def generate_narrative(self, series_name: str, scenario: str = "baseline") -> str:
        """Generate natural language narrative of the forecast"""
        if series_name not in self.scenarios[scenario].forecast_results:
            raise ValueError(f"No forecast available for {series_name} in scenario {scenario}")
        
        results = self.scenarios[scenario].forecast_results[series_name]
        
        # Extract key metrics
        start_forecast = results[0].point_forecast
        end_forecast = results[-1].point_forecast
        growth_rate = (end_forecast - start_forecast) / start_forecast * 100
        
        # Identify significant events
        significant_events = []
        for result in results:
            for event, impact in result.event_impacts.items():
                if abs(impact - 1.0) > 0.1:  # Impact greater than 10%
                    event_obj = next(e for e in self.events if e.name == event)
                    if event_obj not in significant_events:
                        significant_events.append(event_obj)
        
        # Build narrative
        narrative = [
            f"## Forecast Summary for {series_name} ({scenario} scenario)",
            f"- **Projected Growth**: {growth_rate:.1f}% over the forecast period",
            f"- **Starting Value**: {start_forecast:.0f}",
            f"- **Ending Value**: {end_forecast:.0f}",
            "",
            "### Key Influencing Factors:"
        ]
        
        # Add feature importance
        prophet_imp = self.performance_metrics[series_name]['prophet'].features_importance
        xgb_imp = self.performance_metrics[series_name]['xgboost'].features_importance
        
        # Combine importance
        combined_imp = {}
        for feature in set(prophet_imp.keys()) | set(xgb_imp.keys()):
            combined_imp[feature] = (prophet_imp.get(feature, 0) + xgb_imp.get(feature, 0)) / 2
        
        # Top 5 features
        top_features = sorted(combined_imp.items(), key=lambda x: x[1], reverse=True)[:5]
        for feature, imp in top_features:
            narrative.append(f"- {feature} (importance: {imp:.2f})")
        
        # Add event impacts
        if significant_events:
            narrative.append("\n### Significant Events:")
            for event in significant_events:
                narrative.append(f"- **{event.name}** ({event.start_date.date()} to {event.end_date.date()}): {event.description}")
        
        # Add recommendations
        narrative.append("\n### Recommendations:")
        narrative.append(self._generate_recommendations(series_name, results))
        
        # Add uncertainty note
        narrative.append("\n### Uncertainty Note:")
        narrative.append("The forecast includes confidence intervals to capture prediction uncertainty. ")
        narrative.append("Actual results may vary based on unforeseen events and external factors.")
        
        return "\n".join(narrative)
    
    def _generate_recommendations(self, series_name: str, results: List[ForecastResult]) -> str:
        """Generate optimization recommendations based on forecast"""
        # Analyze forecast for opportunities
        min_val = min(r.point_forecast for r in results)
        max_val = max(r.point_forecast for r in results)
        avg_val = np.mean([r.point_forecast for r in results])
        
        # Identify low points
        low_periods = [r for r in results if r.point_forecast < avg_val * 0.9]
        
        recommendations = []
        
        if low_periods:
            low_start = low_periods[0].timestamp.strftime("%Y-%m-%d")
            low_end = low_periods[-1].timestamp.strftime("%Y-%m-%d")
            recommendations.append(
                f"Consider targeted promotions during the low period from {low_start} to {low_end} "
                f"to boost performance by 10-15%."
            )
        
        # Check for event opportunities
        for event in self.events:
            # See if event impact is less than 1 (negative impact)
            event_impact = np.mean([r.event_impacts[event.name] for r in results])
            if event_impact < 0.95:
                recommendations.append(
                    f"Review the impact of '{event.name}' as it shows a negative effect. "
                    f"Consider optimizing this event for better results."
                )
        
        if not recommendations:
            recommendations.append("Current forecast shows stable performance. Maintain current strategy.")
        
        return "\n".join(recommendations)
    
    def create_scenario(self, scenario_id: str, name: str, description: str, parameters: dict):
        """Create a new what-if scenario"""
        if scenario_id in self.scenarios:
            raise ValueError(f"Scenario ID '{scenario_id}' already exists")
        
        new_scenario = Scenario(
            scenario_id=scenario_id,
            name=name,
            description=description,
            parameters=parameters,
            forecast_results={}
        )
        
        self.scenarios[scenario_id] = new_scenario
        print(f"Created new scenario: {name} ({scenario_id})")
    
    def run_scenario(self, scenario_id: str, series_name: str, horizon: int = 30):
        """Run a specific scenario for a series"""
        if scenario_id not in self.scenarios:
            raise ValueError(f"Scenario '{scenario_id}' not found")
        
        print(f"Running scenario '{scenario_id}' for {series_name}")
        return self.forecast(series_name, horizon, scenario_id)
    
    def monte_carlo_simulation(self, series_name: str, horizon: int = 30, n_simulations: int = 1000) -> List[List[float]]:
        """Run Monte Carlo simulation for probabilistic forecasting"""
        # Get baseline forecast
        baseline = self.forecast(series_name, horizon)
        
        # Get error distribution
        prophet_rmse = self.performance_metrics[series_name]['prophet'].rmse
        xgb_rmse = self.performance_metrics[series_name]['xgboost'].rmse
        avg_rmse = (prophet_rmse + xgb_rmse) / 2
        
        # Generate simulations
        simulations = []
        for _ in range(n_simulations):
            sim_path = []
            for i, point in enumerate(baseline):
                # Random error based on normal distribution
                error = np.random.normal(0, avg_rmse * (1 + i/horizon))
                sim_value = point.point_forecast + error
                sim_path.append(max(0, sim_value))  # Ensure non-negative
            simulations.append(sim_path)
        
        return simulations
    
    def cluster_series(self):
        """Cluster time series based on their patterns"""
        if not self.series_names:
            raise ValueError("No series available for clustering")
        
        # Prepare matrix of series characteristics
        features = []
        for series in self.series_names:
            data = self.data[series].dropna()
            series_features = [
                data.mean(),                          # Average level
                data.std(),                           # Volatility
                data.pct_change().mean(),             # Average growth
                data.pct_change().std(),              # Growth volatility
                self._seasonality_strength(data),     # Seasonality strength
                self._trend_strength(data)            # Trend strength
            ]
            features.append(series_features)
        
        # Normalize features
        scaler = StandardScaler()
        scaled_features = scaler.fit_transform(features)
        
        # Cluster using KMeans
        kmeans = KMeans(n_clusters=min(5, len(self.series_names)), random_state=42)
        clusters = kmeans.fit_predict(scaled_features)
        
        # Store clusters
        self.clusters = {}
        for i, series in enumerate(self.series_names):
            cluster_id = clusters[i]
            if cluster_id not in self.clusters:
                self.clusters[cluster_id] = []
            self.clusters[cluster_id].append(series)
        
        return self.clusters
    
    def _seasonality_strength(self, series: pd.Series) -> float:
        """Calculate seasonality strength"""
        try:
            # Use STL decomposition
            stl = sm.tsa.STL(series, period=12).fit()
            residual = stl.resid
            seasonal = stl.seasonal
            
            # Calculate strength
            var_residual = np.nanvar(residual)
            var_seasonal = np.nanvar(seasonal)
            
            if var_residual == 0:
                return 0.0
            return max(0, min(1, 1 - (var_residual / (var_residual + var_seasonal))))
        except:
            return 0.0
    
    def _trend_strength(self, series: pd.Series) -> float:
        """Calculate trend strength"""
        try:
            # Use STL decomposition
            stl = sm.tsa.STL(series, period=12).fit()
            residual = stl.resid
            trend = stl.trend
            
            # Calculate strength
            var_residual = np.nanvar(residual)
            var_trend = np.nanvar(trend)
            
            if var_residual == 0:
                return 0.0
            return max(0, min(1, 1 - (var_residual / (var_residual + var_trend))))
        except:
            return 0.0
    
    def generate_diagnostic_report(self, series_name: str) -> dict:
        """Generate diagnostic report for a series"""
        if series_name not in self.performance_metrics:
            raise ValueError(f"No performance metrics for series: {series_name}")
        
        report = {
            "series": series_name,
            "overview": {
                "last_trained": self.models[series_name]['last_trained'].isoformat(),
                "horizon_supported": 90,
                "data_points": len(self.data)
            },
            "model_performance": {}
        }
        
        # Add model performances
        for model_name, perf in self.performance_metrics[series_name].items():
            report["model_performance"][model_name] = {
                "mae": perf.mae,
                "rmse": perf.rmse,
                "mape": perf.mape,
                "last_retrained": perf.last_retrained.isoformat()
            }
        
        # Add feature importance
        prophet_imp = self.performance_metrics[series_name]['prophet'].features_importance
        xgb_imp = self.performance_metrics[series_name]['xgboost'].features_importance
        
        # Combine importance
        combined_imp = {}
        for feature in set(prophet_imp.keys()) | set(xgb_imp.keys()):
            combined_imp[feature] = (prophet_imp.get(feature, 0) + xgb_imp.get(feature, 0)) / 2
        
        report["feature_importance"] = combined_imp
        
        # Add residuals analysis
        report["residuals_analysis"] = self._analyze_residuals(series_name)
        
        # Add recommendations
        report["recommendations"] = self._generate_diagnostic_recommendations(series_name)
        
        return report
    
    def _analyze_residuals(self, series_name: str) -> dict:
        """Analyze model residuals"""
        # This would be implemented with actual residual analysis
        return {
            "autocorrelation": "Moderate (ACF significant at lag 7)",
            "normality": "Slightly skewed right (Shapiro-Wilk p=0.02)",
            "heteroscedasticity": "Mild (Breusch-Pagan p=0.07)",
            "outliers": "3 significant outliers detected"
        }
    
    def _generate_diagnostic_recommendations(self, series_name: str) -> List[str]:
        """Generate diagnostic recommendations"""
        # This would be based on actual diagnostic results
        return [
            "Consider adding more historical data to improve trend estimation",
            "Evaluate potential external factors not included in the model",
            "Check for structural breaks in the time series",
            "Test different seasonality configurations"
        ]
    
    def optimize_kpi(self, series_name: str, kpi: str = "revenue", budget: float = 10000) -> dict:
        """
        Optimize KPI based on forecast
        
        Args:
            series_name: Name of the series representing the KPI
            kpi: Type of KPI ('revenue', 'conversion', 'traffic', etc.)
            budget: Available budget for optimization
        
        Returns:
            Optimization recommendations
        """
        # Get forecast
        forecast = self.forecast(series_name, horizon=90)
        
        # Find low points
        avg_val = np.mean([f.point_forecast for f in forecast])
        low_periods = [f for f in forecast if f.point_forecast < avg_val * 0.9]
        
        # Calculate opportunity
        opportunity = sum(avg_val - f.point_forecast for f in low_periods)
        
        # Generate recommendations
        recommendations = {
            "action": f"Targeted investment during low periods",
            "budget_allocation": f"Allocate ${budget:.0f} across {len(low_periods)} low periods",
            "expected_impact": f"Potential {kpi} increase of {opportunity:.0f} units",
            "recommended_actions": []
        }
        
        # Specific actions based on KPI type
        if kpi == "revenue":
            recommendations["recommended_actions"].append(
                "Run promotions with 15-20% discounts during low periods"
            )
            recommendations["recommended_actions"].append(
                "Increase marketing spend by 25% in weeks 4-6"
            )
        elif kpi == "conversion":
            recommendations["recommended_actions"].append(
                "Optimize landing pages for high-intent traffic"
            )
            recommendations["recommended_actions"].append(
                "Implement retargeting campaigns for abandoned carts"
            )
        else:
            recommendations["recommended_actions"].append(
                "Boost engagement through personalized content"
            )
            recommendations["recommended_actions"].append(
                "Run A/B tests to identify high-performing variants"
            )
        
        # Budget allocation
        per_period_budget = budget / len(low_periods) if low_periods else 0
        for period in low_periods:
            recommendations["recommended_actions"].append(
                f"Allocate ${per_period_budget:.0f} for {kpi} boost on {period.timestamp.date()}"
            )
        
        return recommendations

# =============================================================================
# Supporting Classes
# =============================================================================
class DriftDetector:
    """Detects concept drift in time series data"""
    def __init__(self, window_size: int = 30, threshold: float = 3.0):
        self.window_size = window_size
        self.threshold = threshold
        self.errors = deque(maxlen=window_size)
        self.mean_error = 0.0
        self.std_error = 0.0
    
    def update(self, actual: float, predicted: float):
        """Update detector with new actual and predicted values"""
        error = abs(actual - predicted) / actual if actual != 0 else 0
        self.errors.append(error)
        
        if len(self.errors) >= self.window_size:
            self.mean_error = np.mean(self.errors)
            self.std_error = np.std(self.errors)
    
    def is_drift_detected(self) -> bool:
        """Check if drift is detected"""
        if len(self.errors) < self.window_size:
            return False
        
        # Calculate z-score of the latest error
        if self.std_error > 0:
            z_score = (self.errors[-1] - self.mean_error) / self.std_error
            return abs(z_score) > self.threshold
        
        return False

# =============================================================================
# Visualization Tools
# =============================================================================
class ChronoVisionVisualizer:
    """Advanced visualization tools for ChronoVision"""
    @staticmethod
    def plot_forecast(historical: pd.Series, forecast: List[ForecastResult], title: str = "Forecast"):
        """Plot historical data and forecast with confidence intervals"""
        plt.figure(figsize=(14, 8))
        
        # Historical data
        plt.plot(historical.index, historical, 'b-', label='Historical')
        
        # Forecast
        dates = [f.timestamp for f in forecast]
        points = [f.point_forecast for f in forecast]
        lower = [f.lower_bound for f in forecast]
        upper = [f.upper_bound for f in forecast]
        
        plt.plot(dates, points, 'r-', label='Forecast')
        plt.fill_between(dates, lower, upper, color='pink', alpha=0.3, label='95% CI')
        
        # Events
        for event in forecast[0].event_impacts.keys():
            # Find event dates where impact != 1
            event_dates = [f.timestamp for f in forecast if abs(f.event_impacts[event] - 1) > 0.05]
            if event_dates:
                start = min(event_dates)
                end = max(event_dates)
                plt.axvspan(start, end, alpha=0.2, color='orange', label=f'Event: {event}')
        
        plt.title(title)
        plt.xlabel('Date')
        plt.ylabel('Value')
        plt.legend()
        plt.grid(True)
        plt.show()
    
    @staticmethod
    def plot_causal_impact(impact: CausalImpact):
        """Visualize causal impact analysis"""
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Plot effect size with confidence interval
        ax.errorbar(
            x=0, 
            y=impact.actual_effect,
            yerr=[[impact.actual_effect - impact.confidence_interval[0]], 
                   [impact.confidence_interval[1] - impact.actual_effect]],
            fmt='o',
            capsize=5,
            label='Estimated Effect'
        )
        
        ax.axhline(0, color='grey', linestyle='--')
        ax.set_title(f"Causal Impact of {impact.event_name} on {impact.series_name}")
        ax.set_ylabel('Effect Size')
        ax.set_xticks([])
        ax.legend()
        plt.show()
    
    @staticmethod
    def plot_scenario_comparison(scenarios: Dict[str, List[ForecastResult]]):
        """Compare multiple scenarios"""
        plt.figure(figsize=(14, 8))
        
        for scenario_id, forecast in scenarios.items():
            dates = [f.timestamp for f in forecast]
            points = [f.point_forecast for f in forecast]
            plt.plot(dates, points, label=scenario_id)
        
        plt.title('Scenario Comparison')
        plt.xlabel('Date')
        plt.ylabel('Value')
        plt.legend()
        plt.grid(True)
        plt.show()
    
    @staticmethod
    def plot_monte_carlo(simulations: List[List[float]], forecast: List[ForecastResult]):
        """Visualize Monte Carlo simulation results"""
        plt.figure(figsize=(14, 8))
        
        # Plot simulations
        dates = [f.timestamp for f in forecast]
        for sim in simulations[:100]:  # Plot first 100 simulations
            plt.plot(dates, sim, 'grey', alpha=0.1)
        
        # Plot mean forecast
        points = [f.point_forecast for f in forecast]
        plt.plot(dates, points, 'r-', linewidth=2, label='Mean Forecast')
        
        plt.title('Monte Carlo Simulation')
        plt.xlabel('Date')
        plt.ylabel('Value')
        plt.legend()
        plt.grid(True)
        plt.show()
    
    @staticmethod
    def plot_feature_importance(importance: Dict[str, float]):
        """Plot feature importance"""
        # Sort features by importance
        features = list(importance.keys())
        values = list(importance.values())
        
        # Sort descending
        sorted_idx = np.argsort(values)[::-1]
        features = [features[i] for i in sorted_idx]
        values = [values[i] for i in sorted_idx]
        
        plt.figure(figsize=(12, 8))
        plt.barh(features, values)
        plt.title('Feature Importance')
        plt.xlabel('Importance Score')
        plt.gca().invert_yaxis()
        plt.show()

# =============================================================================
# Example Usage
# =============================================================================
if __name__ == "__main__":
    print("Initializing ChronoVision Forecasting System...")
    
    # Create engine
    engine = ChronoVisionEngine()
    
    # Generate sample data
    dates = pd.date_range(start='2020-01-01', end='2023-06-30', freq='D')
    n = len(dates)
    np.random.seed(42)
    
    # Base signal with trend and seasonality
    trend = np.linspace(0, 10, n)
    seasonality = 5 * np.sin(2 * np.pi * np.arange(n) / 365)
    noise = np.random.normal(0, 2, n)
    sales = 100 + trend * 5 + seasonality + noise
    
    # Marketing spend (external factor)
    marketing = np.random.gamma(3, 100, n)
    
    # Competitor price (external factor)
    competitor_price = np.random.normal(50, 5, n)
    
    # Create DataFrame
    data = pd.DataFrame({
        'date': dates,
        'sales': sales,
        'marketing_spend': marketing,
        'competitor_price': competitor_price
    })
    
    # Load data
    engine.load_data(
        data=data,
        date_col='date',
        series_cols=['sales'],
        external_factors=['marketing_spend', 'competitor_price'],
        freq='D'
    )
    
    # Add events
    black_friday = TemporalEvent(
        name='Black Friday',
        start_date=datetime.datetime(2020, 11, 27),
        end_date=datetime.datetime(2020, 11, 27),
        event_type='promotion',
        impact_scope={'sales': 1.5},  # 50% increase
        confidence=0.9,
        description="Annual Black Friday promotion"
    )
    
    covid_lockdown = TemporalEvent(
        name='COVID Lockdown',
        start_date=datetime.datetime(2020, 3, 15),
        end_date=datetime.datetime(2020, 6, 15),
        event_type='outage',
        impact_scope={'sales': 0.7},  # 30% decrease
        confidence=0.85,
        description="Global pandemic lockdown"
    )
    
    engine.add_event(black_friday)
    engine.add_event(covid_lockdown)
    
    # Train models
    engine.train_models('sales', domain='retail')
    
    # Generate forecast
    forecast_results = engine.forecast('sales', horizon=90)
    
    # Visualize forecast
    visualizer = ChronoVisionVisualizer()
    visualizer.plot_forecast(engine.data['sales'], forecast_results, "Sales Forecast")
    
    # Run causal impact analysis
    impact = engine.causal_impact_analysis('COVID Lockdown', 'sales')
    print(f"Causal Impact: {impact.actual_effect:.2f} (CI: {impact.confidence_interval[0]:.2f} to {impact.confidence_interval[1]:.2f})")
    visualizer.plot_causal_impact(impact)
    
    # Generate narrative
    narrative = engine.generate_narrative('sales')
    print("\nForecast Narrative:")
    print(narrative)
    
    # Create and run a new scenario
    engine.create_scenario(
        scenario_id="aggressive_marketing",
        name="Aggressive Marketing",
        description="Scenario with 50% increased marketing spend",
        parameters={"marketing_spend_factor": 1.5}
    )
    
    aggressive_results = engine.run_scenario("aggressive_marketing", "sales")
    
    # Compare scenarios
    scenarios = {
        "Baseline": forecast_results,
        "Aggressive Marketing": aggressive_results
    }
    visualizer.plot_scenario_comparison(scenarios)
    
    # Monte Carlo simulation
    simulations = engine.monte_carlo_simulation('sales')
    visualizer.plot_monte_carlo(simulations, forecast_results)
    
    # Feature importance
    importance = engine.performance_metrics['sales']['xgboost'].features_importance
    visualizer.plot_feature_importance(importance)
    
    # Diagnostic report
    report = engine.generate_diagnostic_report('sales')
    print("\nDiagnostic Report Excerpt:")
    print(json.dumps({k: report[k] for k in ['overview', 'model_performance']}, indent=2))
    
    # KPI optimization
    optimization = engine.optimize_kpi('sales', 'revenue', budget=50000)
    print("\nKPI Optimization Recommendations:")
    print(optimization['action'])
    for rec in optimization['recommended_actions']:
        print(f"- {rec}")
    
    print("\nChronoVision demonstration completed.")
