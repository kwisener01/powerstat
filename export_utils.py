"""
Export utilities for DataScope Analytics.

This module provides functions to export analysis results, charts, and reports
in various formats (CSV, Excel, PDF, images).
"""

import logging
from typing import Optional, Dict, Any
from io import BytesIO
import pandas as pd
import plotly.graph_objects as go
from datetime import datetime
import config

# Configure logging
logging.basicConfig(level=getattr(logging, config.LOG_LEVEL))
logger = logging.getLogger(__name__)


def export_dataframe_to_csv(df: pd.DataFrame, filename: Optional[str] = None) -> bytes:
    """
    Export DataFrame to CSV format.

    Args:
        df: DataFrame to export
        filename: Optional filename (for logging purposes)

    Returns:
        CSV data as bytes
    """
    try:
        csv_buffer = BytesIO()
        df.to_csv(csv_buffer, index=False)
        csv_data = csv_buffer.getvalue()

        logger.info(f"DataFrame exported to CSV: {filename or 'unnamed'}")
        return csv_data

    except Exception as e:
        logger.error(f"Error exporting DataFrame to CSV: {str(e)}")
        raise


def export_dataframe_to_excel(df: pd.DataFrame, sheet_name: str = "Data",
                              filename: Optional[str] = None) -> bytes:
    """
    Export DataFrame to Excel format.

    Args:
        df: DataFrame to export
        sheet_name: Name of the Excel sheet
        filename: Optional filename (for logging purposes)

    Returns:
        Excel data as bytes
    """
    try:
        excel_buffer = BytesIO()
        with pd.ExcelWriter(excel_buffer, engine='openpyxl') as writer:
            df.to_excel(writer, sheet_name=sheet_name, index=False)

        excel_data = excel_buffer.getvalue()
        logger.info(f"DataFrame exported to Excel: {filename or 'unnamed'}")
        return excel_data

    except Exception as e:
        logger.error(f"Error exporting DataFrame to Excel: {str(e)}")
        raise


def export_analysis_report(basic_stats: Dict[str, Any],
                          corr_analysis: Optional[Dict[str, Any]] = None,
                          outliers: Optional[Dict[str, Any]] = None) -> bytes:
    """
    Export comprehensive analysis report to Excel with multiple sheets.

    Args:
        basic_stats: Basic statistics analysis results
        corr_analysis: Optional correlation analysis results
        outliers: Optional outlier analysis results

    Returns:
        Excel data as bytes containing multiple sheets
    """
    try:
        excel_buffer = BytesIO()

        with pd.ExcelWriter(excel_buffer, engine='openpyxl') as writer:
            # Summary sheet
            summary_data = {
                'Metric': ['Total Rows', 'Total Columns', 'Numeric Columns', 'Categorical Columns'],
                'Value': [
                    basic_stats['shape'][0],
                    basic_stats['shape'][1],
                    len([col for col, dtype in basic_stats['data_types'].items()
                         if pd.api.types.is_numeric_dtype(dtype)]),
                    len([col for col, dtype in basic_stats['data_types'].items()
                         if dtype == 'object'])
                ]
            }
            pd.DataFrame(summary_data).to_excel(writer, sheet_name='Summary', index=False)

            # Missing values sheet
            if basic_stats.get('missing_values'):
                missing_df = pd.DataFrame(
                    [(k, v) for k, v in basic_stats['missing_values'].items() if v > 0],
                    columns=['Column', 'Missing Count']
                )
                if not missing_df.empty:
                    missing_df.to_excel(writer, sheet_name='Missing Values', index=False)

            # Descriptive statistics sheet
            if basic_stats.get('numeric_summary') is not None:
                basic_stats['numeric_summary'].to_excel(writer, sheet_name='Descriptive Stats')

            # Correlation sheet
            if corr_analysis and corr_analysis.get('matrix') is not None:
                corr_analysis['matrix'].to_excel(writer, sheet_name='Correlation Matrix')

                if corr_analysis.get('strong_correlations'):
                    strong_corr_df = pd.DataFrame(corr_analysis['strong_correlations'])
                    strong_corr_df.to_excel(writer, sheet_name='Strong Correlations', index=False)

            # Outliers sheet
            if outliers:
                outlier_summary = []
                for col, info in outliers.items():
                    outlier_summary.append({
                        'Column': col,
                        'Outlier Count': info['count'],
                        'Percentage': f"{info['percentage']:.2f}%"
                    })
                if outlier_summary:
                    pd.DataFrame(outlier_summary).to_excel(writer, sheet_name='Outliers', index=False)

        excel_data = excel_buffer.getvalue()
        logger.info("Analysis report exported to Excel")
        return excel_data

    except Exception as e:
        logger.error(f"Error exporting analysis report: {str(e)}")
        raise


def export_manufacturing_report(mfg_analysis: Dict[str, Any]) -> bytes:
    """
    Export manufacturing analysis report to Excel with multiple sheets.

    Args:
        mfg_analysis: Manufacturing analysis results

    Returns:
        Excel data as bytes containing multiple sheets
    """
    try:
        excel_buffer = BytesIO()

        with pd.ExcelWriter(excel_buffer, engine='openpyxl') as writer:
            # KPIs sheet
            kpi_data = {
                'KPI': [
                    'Average Cycle Time (seconds)',
                    'Average Cycle Time (minutes)',
                    'Median Cycle Time (seconds)',
                    'Cycle Time Std Dev (seconds)',
                    'Min Cycle Time (seconds)',
                    'Max Cycle Time (seconds)',
                    'Parts per Hour',
                    'Parts per Day',
                    'Total Parts Produced',
                    'Performance Efficiency (%)',
                    'Production Stoppages',
                    'Total Stoppage Time (hours)',
                    'Avg Stoppage Duration (minutes)'
                ],
                'Value': [
                    f"{mfg_analysis['cycle_time_stats']['mean_seconds']:.2f}",
                    f"{mfg_analysis['cycle_time_stats']['mean_minutes']:.2f}",
                    f"{mfg_analysis['cycle_time_stats']['median_seconds']:.2f}",
                    f"{mfg_analysis['cycle_time_stats']['std_seconds']:.2f}",
                    f"{mfg_analysis['cycle_time_stats']['min_seconds']:.2f}",
                    f"{mfg_analysis['cycle_time_stats']['max_seconds']:.2f}",
                    f"{mfg_analysis['production_metrics']['parts_per_hour']:.2f}",
                    f"{mfg_analysis['production_metrics']['parts_per_day']:.2f}",
                    mfg_analysis['production_metrics']['total_parts'],
                    f"{mfg_analysis['production_metrics']['performance_efficiency']:.2f}",
                    mfg_analysis['stoppage_analysis']['stoppage_count'],
                    f"{mfg_analysis['stoppage_analysis']['total_stoppage_time']/3600:.2f}",
                    f"{mfg_analysis['stoppage_analysis']['avg_stoppage_duration']/60:.2f}"
                ]
            }
            pd.DataFrame(kpi_data).to_excel(writer, sheet_name='KPIs', index=False)

            # Shift analysis sheet
            if not mfg_analysis['shift_analysis'].empty:
                shift_data = mfg_analysis['shift_analysis'].copy()
                shift_data['Avg_Minutes'] = shift_data['mean'] / 60
                shift_data['Parts_per_Hour'] = 3600 / shift_data['mean']
                shift_data.to_excel(writer, sheet_name='Shift Analysis')

            # Part analysis sheet (if available)
            if mfg_analysis.get('part_analysis') is not None:
                part_data = mfg_analysis['part_analysis'].copy()
                part_data['Avg_Minutes'] = part_data['Avg_Cycle_Time'] / 60
                part_data['Parts_per_Hour'] = 3600 / part_data['Avg_Cycle_Time']
                part_data.to_excel(writer, sheet_name='Part Analysis')

            # Time series data sheet (sample)
            ts_data = mfg_analysis['time_series_data'].dropna()
            if len(ts_data) > 0:
                # Export up to 1000 rows
                ts_data.head(1000).to_excel(writer, sheet_name='Time Series Data', index=False)

        excel_data = excel_buffer.getvalue()
        logger.info("Manufacturing report exported to Excel")
        return excel_data

    except Exception as e:
        logger.error(f"Error exporting manufacturing report: {str(e)}")
        raise


def get_timestamp_filename(prefix: str, extension: str) -> str:
    """
    Generate a filename with timestamp.

    Args:
        prefix: Filename prefix
        extension: File extension (without dot)

    Returns:
        Filename string with timestamp
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"{prefix}_{timestamp}.{extension}"


def export_chart_as_html(fig: go.Figure, filename: Optional[str] = None) -> bytes:
    """
    Export Plotly chart as standalone HTML file.

    Args:
        fig: Plotly Figure object
        filename: Optional filename (for logging purposes)

    Returns:
        HTML data as bytes
    """
    try:
        html_buffer = BytesIO()
        html_string = fig.to_html(include_plotlyjs='cdn')
        html_buffer.write(html_string.encode('utf-8'))
        html_data = html_buffer.getvalue()

        logger.info(f"Chart exported to HTML: {filename or 'unnamed'}")
        return html_data

    except Exception as e:
        logger.error(f"Error exporting chart to HTML: {str(e)}")
        raise
