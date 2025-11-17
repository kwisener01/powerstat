"""
Data analysis module for DataScope Analytics.

This module provides the DataAnalyzer class which encapsulates various
statistical and manufacturing analysis methods.
"""

import logging
from typing import Dict, Optional, Any, List
import pandas as pd
import numpy as np
from scipy import stats
import config

# Configure logging
logging.basicConfig(level=getattr(logging, config.LOG_LEVEL))
logger = logging.getLogger(__name__)


class DataAnalyzer:
    """AI-powered data analysis team simulator"""

    def __init__(self, df: pd.DataFrame):
        """
        Initialize the DataAnalyzer with a DataFrame.

        Args:
            df: Input DataFrame to analyze
        """
        self.df = df
        self.numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
        self.categorical_columns = df.select_dtypes(include=['object']).columns.tolist()
        logger.info(f"DataAnalyzer initialized with {len(df)} rows, "
                   f"{len(self.numeric_columns)} numeric cols, "
                   f"{len(self.categorical_columns)} categorical cols")

    def basic_stats_analysis(self) -> Dict[str, Any]:
        """
        Perform basic statistical analysis.

        Returns:
            Dictionary containing shape, missing values, data types, and summaries
        """
        try:
            analysis = {
                'shape': self.df.shape,
                'missing_values': self.df.isnull().sum().to_dict(),
                'data_types': self.df.dtypes.to_dict(),
                'numeric_summary': self.df.describe() if self.numeric_columns else None,
                'categorical_summary': {}
            }

            for col in self.categorical_columns:
                analysis['categorical_summary'][col] = {
                    'unique_count': self.df[col].nunique(),
                    'top_values': self.df[col].value_counts().head().to_dict()
                }

            logger.info("Basic statistics analysis completed successfully")
            return analysis

        except Exception as e:
            logger.error(f"Error in basic_stats_analysis: {str(e)}")
            raise

    def correlation_analysis(self) -> Optional[Dict[str, Any]]:
        """
        Perform correlation analysis on numeric columns.

        Returns:
            Dictionary with correlation matrix and strong correlations, or None if insufficient data
        """
        if len(self.numeric_columns) < 2:
            logger.warning("Correlation analysis requires at least 2 numeric columns")
            return None

        try:
            corr_matrix = self.df[self.numeric_columns].corr()

            # Find strongest correlations
            corr_pairs = []
            for i in range(len(corr_matrix.columns)):
                for j in range(i+1, len(corr_matrix.columns)):
                    corr_val = corr_matrix.iloc[i, j]
                    if abs(corr_val) > config.CORRELATION_THRESHOLD:
                        corr_pairs.append({
                            'var1': corr_matrix.columns[i],
                            'var2': corr_matrix.columns[j],
                            'correlation': corr_val
                        })

            logger.info(f"Correlation analysis found {len(corr_pairs)} strong correlations")
            return {
                'matrix': corr_matrix,
                'strong_correlations': sorted(corr_pairs, key=lambda x: abs(x['correlation']), reverse=True)
            }

        except Exception as e:
            logger.error(f"Error in correlation_analysis: {str(e)}")
            return None

    def outlier_analysis(self) -> Dict[str, Dict[str, Any]]:
        """
        Detect outliers using IQR method.

        Returns:
            Dictionary mapping column names to outlier information
        """
        outliers = {}

        try:
            for col in self.numeric_columns:
                Q1 = self.df[col].quantile(0.25)
                Q3 = self.df[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - config.OUTLIER_IQR_MULTIPLIER * IQR
                upper_bound = Q3 + config.OUTLIER_IQR_MULTIPLIER * IQR

                outlier_mask = (self.df[col] < lower_bound) | (self.df[col] > upper_bound)
                outlier_count = outlier_mask.sum()

                if outlier_count > 0:
                    outliers[col] = {
                        'count': outlier_count,
                        'percentage': (outlier_count / len(self.df)) * 100,
                        'values': self.df[outlier_mask][col].tolist()
                    }

            logger.info(f"Outlier analysis completed. Found outliers in {len(outliers)} columns")
            return outliers

        except Exception as e:
            logger.error(f"Error in outlier_analysis: {str(e)}")
            return {}

    def trend_analysis(self, date_col: Optional[str] = None,
                      value_col: Optional[str] = None) -> Optional[Dict[str, Any]]:
        """
        Perform time series trend analysis.

        Args:
            date_col: Name of the date/time column
            value_col: Name of the value column to analyze

        Returns:
            Dictionary with trend statistics or None if analysis fails
        """
        if date_col is None or value_col is None:
            logger.warning("Trend analysis requires both date_col and value_col")
            return None

        try:
            df_trend = self.df.copy()
            df_trend[date_col] = pd.to_datetime(df_trend[date_col])
            df_trend = df_trend.sort_values(date_col)

            # Calculate trend using linear regression
            x = np.arange(len(df_trend))
            slope, intercept, r_value, p_value, std_err = stats.linregress(x, df_trend[value_col])

            result = {
                'slope': slope,
                'r_squared': r_value**2,
                'p_value': p_value,
                'trend_direction': 'Increasing' if slope > 0 else 'Decreasing',
                'data': df_trend[[date_col, value_col]]
            }

            logger.info(f"Trend analysis completed. Direction: {result['trend_direction']}, "
                       f"R²: {result['r_squared']:.3f}")
            return result

        except ValueError as e:
            logger.error(f"Value error in trend_analysis: {str(e)} - Check column data types")
            return None
        except KeyError as e:
            logger.error(f"Column not found in trend_analysis: {str(e)}")
            return None
        except Exception as e:
            logger.error(f"Unexpected error in trend_analysis: {str(e)}")
            return None

    def manufacturing_time_analysis(self, timestamp_col: str,
                                   part_id_col: Optional[str] = None) -> Optional[Dict[str, Any]]:
        """
        Analyze manufacturing timestamps and production metrics.

        Args:
            timestamp_col: Name of the timestamp column
            part_id_col: Optional name of the part ID column

        Returns:
            Dictionary with comprehensive manufacturing metrics or None if analysis fails
        """
        try:
            df_mfg = self.df.copy()

            # Parse timestamps
            try:
                df_mfg[timestamp_col] = pd.to_datetime(df_mfg[timestamp_col])
            except Exception as e:
                logger.error(f"Failed to parse timestamps in column '{timestamp_col}': {str(e)}")
                return None

            df_mfg = df_mfg.sort_values(timestamp_col)

            # Calculate time differences between consecutive parts
            df_mfg['time_diff_seconds'] = df_mfg[timestamp_col].diff().dt.total_seconds()
            df_mfg['time_diff_minutes'] = df_mfg['time_diff_seconds'] / 60
            df_mfg['time_diff_hours'] = df_mfg['time_diff_minutes'] / 60

            # Remove first row (NaN) and filter out unrealistic values
            time_diffs = df_mfg['time_diff_seconds'].dropna()
            max_seconds = config.MAX_CYCLE_TIME_HOURS * 3600
            time_diffs = time_diffs[time_diffs <= max_seconds]

            if len(time_diffs) == 0:
                logger.warning("No valid time differences found after filtering")
                return None

            # Calculate production rate metrics
            mean_cycle_time = time_diffs.mean()
            median_cycle_time = time_diffs.median()
            std_cycle_time = time_diffs.std()
            min_cycle_time = time_diffs.min()
            max_cycle_time = time_diffs.max()

            # Production rate (parts per hour)
            parts_per_hour = 3600 / mean_cycle_time if mean_cycle_time > 0 else 0

            # Detect production stoppages
            stoppage_threshold = mean_cycle_time + config.STOPPAGE_SIGMA_THRESHOLD * std_cycle_time
            stoppages = time_diffs[time_diffs > stoppage_threshold]

            # Calculate OEE-related metrics
            total_production_time = (df_mfg[timestamp_col].max() -
                                    df_mfg[timestamp_col].min()).total_seconds()
            ideal_cycle_time = time_diffs.quantile(config.IDEAL_PERFORMANCE_QUANTILE)
            theoretical_max_parts = total_production_time / ideal_cycle_time if ideal_cycle_time > 0 else 0
            actual_parts = len(df_mfg) - 1
            performance_efficiency = (actual_parts / theoretical_max_parts * 100) if theoretical_max_parts > 0 else 0

            # Shift analysis
            df_mfg['hour'] = df_mfg[timestamp_col].dt.hour
            df_mfg['shift'] = df_mfg['hour'].apply(lambda x:
                'Day Shift (6-14)' if 6 <= x < 14 else
                'Evening Shift (14-22)' if 14 <= x < 22 else
                'Night Shift (22-6)')

            shift_stats = df_mfg.groupby('shift')['time_diff_seconds'].agg(['count', 'mean', 'std']).round(2)

            # Part-specific analysis if part_id_col is provided
            part_analysis = None
            if part_id_col and part_id_col in df_mfg.columns:
                part_analysis = df_mfg.groupby(part_id_col)['time_diff_seconds'].agg([
                    'count', 'mean', 'std', 'min', 'max'
                ]).round(2)
                part_analysis.columns = ['Count', 'Avg_Cycle_Time', 'Std_Cycle_Time',
                                        'Min_Cycle_Time', 'Max_Cycle_Time']

            result = {
                'cycle_time_stats': {
                    'mean_seconds': mean_cycle_time,
                    'median_seconds': median_cycle_time,
                    'std_seconds': std_cycle_time,
                    'min_seconds': min_cycle_time,
                    'max_seconds': max_cycle_time,
                    'mean_minutes': mean_cycle_time / 60,
                    'median_minutes': median_cycle_time / 60
                },
                'production_metrics': {
                    'parts_per_hour': parts_per_hour,
                    'parts_per_day': parts_per_hour * 24,
                    'total_parts': actual_parts,
                    'performance_efficiency': performance_efficiency
                },
                'stoppage_analysis': {
                    'stoppage_count': len(stoppages),
                    'total_stoppage_time': stoppages.sum(),
                    'avg_stoppage_duration': stoppages.mean() if len(stoppages) > 0 else 0,
                    'stoppage_threshold_seconds': stoppage_threshold
                },
                'shift_analysis': shift_stats,
                'part_analysis': part_analysis,
                'time_series_data': df_mfg[[timestamp_col, 'time_diff_seconds',
                                           'time_diff_minutes', 'hour', 'shift']],
                'raw_time_diffs': time_diffs
            }

            logger.info(f"Manufacturing analysis completed. Parts/hour: {parts_per_hour:.1f}, "
                       f"Efficiency: {performance_efficiency:.1f}%")
            return result

        except KeyError as e:
            logger.error(f"Column not found in manufacturing_time_analysis: {str(e)}")
            return None
        except Exception as e:
            logger.error(f"Unexpected error in manufacturing_time_analysis: {str(e)}")
            return None
