"""
Chart building module for DataScope Analytics.

This module provides functions to create various charts and visualizations
using Plotly and other visualization libraries.
"""

import logging
from typing import Optional, Dict, Any
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st
import config

# Configure logging
logging.basicConfig(level=getattr(logging, config.LOG_LEVEL))
logger = logging.getLogger(__name__)


def create_pareto_chart(df: pd.DataFrame, category_col: str,
                       value_col: Optional[str] = None) -> Optional[go.Figure]:
    """
    Create a Pareto Chart for analyzing the vital few vs trivial many.

    Args:
        df: Input DataFrame
        category_col: Column name for categories
        value_col: Optional column name for values (uses frequency if None)

    Returns:
        Plotly Figure object or None if creation fails
    """
    try:
        if value_col is None:
            # Count frequency if no value column specified
            pareto_data = df[category_col].value_counts().reset_index()
            pareto_data.columns = [category_col, 'Count']
            value_col = 'Count'
        else:
            pareto_data = df.groupby(category_col)[value_col].sum().reset_index()
            pareto_data = pareto_data.sort_values(value_col, ascending=False)

        # Calculate cumulative percentage
        pareto_data['Cumulative'] = pareto_data[value_col].cumsum()
        pareto_data['Cumulative_Percent'] = (pareto_data['Cumulative'] /
                                             pareto_data[value_col].sum()) * 100

        # Create subplot with secondary y-axis
        fig = make_subplots(specs=[[{"secondary_y": True}]])

        # Add bar chart
        fig.add_trace(
            go.Bar(x=pareto_data[category_col], y=pareto_data[value_col], name=value_col),
            secondary_y=False,
        )

        # Add line chart for cumulative percentage
        fig.add_trace(
            go.Scatter(x=pareto_data[category_col], y=pareto_data['Cumulative_Percent'],
                      mode='lines+markers', name='Cumulative %', line=dict(color='red')),
            secondary_y=True,
        )

        # Update layout
        fig.update_xaxes(title_text=category_col)
        fig.update_yaxes(title_text=value_col, secondary_y=False)
        fig.update_yaxes(title_text="Cumulative Percentage", secondary_y=True, range=[0, 100])
        fig.update_layout(title_text="Pareto Chart", height=config.DEFAULT_CHART_HEIGHT)

        logger.info(f"Pareto chart created successfully for {category_col}")
        return fig

    except KeyError as e:
        logger.error(f"Column not found in create_pareto_chart: {str(e)}")
        st.error(f"Error: Column '{e}' not found in the dataset")
        return None
    except Exception as e:
        logger.error(f"Unexpected error in create_pareto_chart: {str(e)}")
        st.error(f"Could not create Pareto chart: {str(e)}")
        return None


def create_run_chart(df: pd.DataFrame, date_col: str,
                    value_col: str) -> Optional[go.Figure]:
    """
    Create a Run Chart with control limits for process monitoring.

    Args:
        df: Input DataFrame
        date_col: Column name for date/time values
        value_col: Column name for values to plot

    Returns:
        Plotly Figure object or None if creation fails
    """
    try:
        df_run = df.copy()

        # Parse dates
        try:
            df_run[date_col] = pd.to_datetime(df_run[date_col])
        except Exception as e:
            logger.error(f"Failed to parse dates in column '{date_col}': {str(e)}")
            st.error(f"Could not parse dates in '{date_col}'. Please ensure it contains valid date/time values.")
            return None

        df_run = df_run.sort_values(date_col)

        # Calculate control limits
        mean_val = df_run[value_col].mean()
        std_val = df_run[value_col].std()
        ucl = mean_val + config.CONTROL_LIMIT_SIGMA * std_val
        lcl = mean_val - config.CONTROL_LIMIT_SIGMA * std_val

        fig = go.Figure()

        # Add main data line
        fig.add_trace(go.Scatter(
            x=df_run[date_col], y=df_run[value_col],
            mode='lines+markers',
            name=value_col,
            line=dict(color='blue')
        ))

        # Add control limits
        fig.add_hline(y=mean_val, line_dash="dash", line_color="green",
                     annotation_text="Mean")
        fig.add_hline(y=ucl, line_dash="dash", line_color="red",
                     annotation_text="UCL")
        fig.add_hline(y=lcl, line_dash="dash", line_color="red",
                     annotation_text="LCL")

        fig.update_layout(
            title="Run Chart with Control Limits",
            xaxis_title=date_col,
            yaxis_title=value_col,
            height=config.DEFAULT_CHART_HEIGHT
        )

        logger.info(f"Run chart created successfully for {value_col} over {date_col}")
        return fig

    except KeyError as e:
        logger.error(f"Column not found in create_run_chart: {str(e)}")
        st.error(f"Error: Column '{e}' not found in the dataset")
        return None
    except Exception as e:
        logger.error(f"Unexpected error in create_run_chart: {str(e)}")
        st.error(f"Could not create run chart: {str(e)}")
        return None


def create_manufacturing_dashboard(mfg_analysis: Dict[str, Any]) -> Optional[go.Figure]:
    """
    Create a comprehensive manufacturing performance dashboard.

    Args:
        mfg_analysis: Dictionary containing manufacturing analysis results

    Returns:
        Plotly Figure object with subplots or None if creation fails
    """
    if not mfg_analysis:
        logger.warning("No manufacturing analysis data provided")
        return None

    try:
        # Create subplots
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Cycle Time Distribution', 'Production Rate Over Time',
                           'Shift Performance', 'Control Chart'),
            specs=[[{"type": "histogram"}, {"type": "scatter"}],
                   [{"type": "bar"}, {"type": "scatter"}]]
        )

        # 1. Cycle Time Distribution
        fig.add_trace(
            go.Histogram(x=mfg_analysis['raw_time_diffs'], name='Cycle Time',
                        nbinsx=config.HISTOGRAM_BINS, showlegend=False),
            row=1, col=1
        )

        # 2. Production Rate Over Time (using time series data)
        ts_data = mfg_analysis['time_series_data'].dropna()
        if len(ts_data) > 1:
            # Calculate rolling production rate
            ts_data['rolling_rate'] = 3600 / ts_data['time_diff_seconds'].rolling(
                window=config.ROLLING_WINDOW_SIZE, min_periods=1).mean()

            fig.add_trace(
                go.Scatter(x=ts_data.index, y=ts_data['rolling_rate'],
                          mode='lines', name='Parts/Hour', showlegend=False),
                row=1, col=2
            )

        # 3. Shift Performance
        shift_data = mfg_analysis['shift_analysis']
        if not shift_data.empty:
            fig.add_trace(
                go.Bar(x=shift_data.index, y=shift_data['mean'],
                       name='Avg Cycle Time', showlegend=False,
                       error_y=dict(type='data', array=shift_data['std'])),
                row=2, col=1
            )

        # 4. Control Chart for Cycle Times
        time_diffs = mfg_analysis['raw_time_diffs']
        mean_time = time_diffs.mean()
        std_time = time_diffs.std()
        ucl = mean_time + config.CONTROL_LIMIT_SIGMA * std_time
        lcl = max(0, mean_time - config.CONTROL_LIMIT_SIGMA * std_time)

        fig.add_trace(
            go.Scatter(x=list(range(len(time_diffs))), y=time_diffs,
                      mode='markers', name='Cycle Time', showlegend=False),
            row=2, col=2
        )

        # Add control limits
        fig.add_hline(y=mean_time, line_dash="dash", line_color="green", row=2, col=2)
        fig.add_hline(y=ucl, line_dash="dash", line_color="red", row=2, col=2)
        fig.add_hline(y=lcl, line_dash="dash", line_color="red", row=2, col=2)

        fig.update_layout(height=config.DASHBOARD_CHART_HEIGHT,
                         title_text="Manufacturing Performance Dashboard")

        logger.info("Manufacturing dashboard created successfully")
        return fig

    except KeyError as e:
        logger.error(f"Missing data in create_manufacturing_dashboard: {str(e)}")
        st.error(f"Error: Missing required data '{e}' in manufacturing analysis")
        return None
    except Exception as e:
        logger.error(f"Unexpected error in create_manufacturing_dashboard: {str(e)}")
        st.error(f"Could not create manufacturing dashboard: {str(e)}")
        return None


def generate_sample_sales_data(seed: int = 42) -> pd.DataFrame:
    """
    Generate sample sales data for demonstration purposes.

    Args:
        seed: Random seed for reproducibility

    Returns:
        DataFrame with sample sales data
    """
    try:
        np.random.seed(seed)
        dates = pd.date_range('2023-01-01', periods=365, freq='D')
        regions = ['North', 'South', 'East', 'West']
        products = ['Product A', 'Product B', 'Product C', 'Product D', 'Product E']

        sample_data = []
        for date in dates:
            for _ in range(np.random.randint(1, 5)):
                sample_data.append({
                    'Date': date,
                    'Region': np.random.choice(regions),
                    'Product': np.random.choice(products),
                    'Sales': np.random.normal(1000, 200),
                    'Quantity': np.random.randint(1, 50),
                    'Cost': np.random.normal(600, 100)
                })

        logger.info("Sample sales data generated successfully")
        return pd.DataFrame(sample_data)

    except Exception as e:
        logger.error(f"Error generating sample sales data: {str(e)}")
        raise


def generate_sample_manufacturing_data() -> pd.DataFrame:
    """
    Generate realistic sample manufacturing timestamp data.

    Returns:
        DataFrame with sample manufacturing data
    """
    try:
        start_time = pd.Timestamp(config.SAMPLE_DATA_START_DATE)

        # Simulate production with realistic cycle times and stoppages
        timestamps = []
        current_time = start_time

        for i in range(config.SAMPLE_DATA_ROWS):
            # Base cycle time varies by part type
            part_type = np.random.choice(config.SAMPLE_PART_TYPES)
            base_cycle = config.SAMPLE_BASE_CYCLE_TIMES[part_type]

            # Add variation and shift effects (2-shift system: 6:30-18:30 and 18:30-6:30)
            hour = current_time.hour
            minute = current_time.minute
            decimal_hour = hour + minute / 60.0

            # Night shift (18:30-6:30) is typically slower than day shift
            shift_multiplier = 1.0
            if config.SHIFT_DAY_START <= decimal_hour < config.SHIFT_NIGHT_START:
                # Day shift (6:30 AM - 6:30 PM) - baseline performance
                shift_multiplier = 1.0
            else:
                # Night shift (6:30 PM - 6:30 AM) - slightly slower
                shift_multiplier = 1.10

            # Random variation
            cycle_time = np.random.normal(base_cycle * shift_multiplier, base_cycle * 0.1)
            cycle_time = max(cycle_time, base_cycle * 0.5)  # Minimum time

            # Occasional stoppages
            if np.random.random() < config.SAMPLE_STOPPAGE_PROBABILITY:
                cycle_time += np.random.exponential(config.SAMPLE_AVG_STOPPAGE_DURATION)

            current_time += pd.Timedelta(seconds=cycle_time)

            timestamps.append({
                'timestamp': current_time,
                'part_id': part_type,
                'operator': np.random.choice(config.SAMPLE_OPERATORS),
                'machine': f'M{np.random.randint(1, config.SAMPLE_MACHINES + 1)}',
                'quality_score': np.random.normal(95, 3)
            })

        logger.info("Sample manufacturing data generated successfully")
        return pd.DataFrame(timestamps)

    except Exception as e:
        logger.error(f"Error generating sample manufacturing data: {str(e)}")
        raise
