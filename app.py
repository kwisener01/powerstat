"""
DataScope Analytics - AI-Powered Business Intelligence Platform

A Streamlit application for comprehensive data analysis including statistical analysis,
visualizations, Pareto charts, run charts, and manufacturing analytics.
"""

import logging
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

# Import custom modules
import config
from data_analyzer import DataAnalyzer
from chart_builders import (
    create_pareto_chart,
    create_run_chart,
    create_manufacturing_dashboard,
    generate_sample_sales_data,
    generate_sample_manufacturing_data
)
from export_utils import (
    export_dataframe_to_csv,
    export_dataframe_to_excel,
    export_analysis_report,
    export_manufacturing_report,
    get_timestamp_filename,
    export_chart_as_html
)

# Configure logging
logging.basicConfig(level=getattr(logging, config.LOG_LEVEL))
logger = logging.getLogger(__name__)

# Configure page
st.set_page_config(
    page_title=config.APP_TITLE,
    page_icon=config.APP_ICON,
    layout=config.PAGE_LAYOUT,
    initial_sidebar_state=config.SIDEBAR_STATE
)

# Apply custom CSS
st.markdown(config.MAIN_HEADER_STYLE, unsafe_allow_html=True)


def calculate_run_chart_insights(df: pd.DataFrame, date_col: str, value_col: str) -> dict:
    """
    Calculate process control insights for run charts.

    Args:
        df: Input DataFrame
        date_col: Name of the date column
        value_col: Name of the value column

    Returns:
        Dictionary with insights or None if calculation fails
    """
    try:
        df_temp = df.copy()
        df_temp[date_col] = pd.to_datetime(df_temp[date_col])
        df_temp = df_temp.sort_values(date_col)

        mean_val = df_temp[value_col].mean()
        std_val = df_temp[value_col].std()
        ucl = mean_val + config.CONTROL_LIMIT_SIGMA * std_val
        lcl = mean_val - config.CONTROL_LIMIT_SIGMA * std_val

        out_of_control = ((df_temp[value_col] > ucl) | (df_temp[value_col] < lcl)).sum()

        return {
            'mean': mean_val,
            'std': std_val,
            'ucl': ucl,
            'lcl': lcl,
            'out_of_control_count': out_of_control
        }
    except (ValueError, KeyError) as e:
        logger.error(f"Error calculating run chart insights: {str(e)}")
        st.error(f"Could not calculate control statistics: {str(e)}")
        return None
    except Exception as e:
        logger.error(f"Unexpected error in run chart insights: {str(e)}")
        st.error(f"Unexpected error calculating control statistics")
        return None


def generate_manufacturing_recommendations(mfg_analysis: dict) -> list:
    """
    Generate AI-powered manufacturing recommendations based on analysis.

    Args:
        mfg_analysis: Manufacturing analysis results

    Returns:
        List of recommendation strings
    """
    recommendations = []

    try:
        # Performance efficiency
        perf_eff = mfg_analysis['production_metrics']['performance_efficiency']
        if perf_eff < config.PERFORMANCE_GOOD:
            recommendations.append(
                f"🔴 Performance efficiency is below {config.PERFORMANCE_GOOD}%. "
                "Consider investigating bottlenecks."
            )
        elif perf_eff < config.PERFORMANCE_EXCELLENT:
            recommendations.append(
                "🟡 Performance efficiency could be improved. "
                "Look for optimization opportunities."
            )
        else:
            recommendations.append("🟢 Good performance efficiency. Maintain current processes.")

        # Cycle time variability
        cv = (mfg_analysis['cycle_time_stats']['std_seconds'] /
              mfg_analysis['cycle_time_stats']['mean_seconds'])
        if cv > config.CV_MODERATE:
            recommendations.append(
                "🔴 High cycle time variability detected. Investigate process consistency."
            )
        elif cv > config.CV_LOW:
            recommendations.append(
                "🟡 Moderate cycle time variability. Consider process standardization."
            )
        else:
            recommendations.append("🟢 Low cycle time variability. Process appears stable.")

        # Stoppages
        stoppage_ratio = (mfg_analysis['stoppage_analysis']['stoppage_count'] /
                         len(mfg_analysis['raw_time_diffs']))
        if stoppage_ratio > config.STOPPAGE_PERCENTAGE_THRESHOLD:
            recommendations.append(
                "🔴 High number of production stoppages. Review maintenance schedules."
            )

        return recommendations

    except (KeyError, ZeroDivisionError) as e:
        logger.error(f"Error generating recommendations: {str(e)}")
        return ["⚠️ Could not generate recommendations due to incomplete data"]


def main():
    """Main application entry point"""
    st.markdown(f'<h1 class="main-header">{config.APP_ICON} {config.APP_TITLE}</h1>',
                unsafe_allow_html=True)
    st.markdown("### AI-Powered Business Intelligence & Data Analysis Platform")

    # Sidebar
    st.sidebar.header("🚀 Data Upload & Settings")

    # File upload
    uploaded_file = st.sidebar.file_uploader(
        "Upload your data file",
        type=config.SUPPORTED_FILE_TYPES,
        help="Upload CSV or Excel files for analysis"
    )

    if uploaded_file is not None:
        try:
            # Load data
            if uploaded_file.name.endswith('.csv'):
                df = pd.read_csv(uploaded_file)
            else:
                df = pd.read_excel(uploaded_file)

            st.sidebar.success(f"✅ File loaded: {uploaded_file.name}")
            st.sidebar.info(f"Shape: {df.shape[0]} rows × {df.shape[1]} columns")

            # Initialize analyzer
            analyzer = DataAnalyzer(df)

            # Main content tabs
            tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
                "📋 Data Overview",
                "📊 Statistical Analysis",
                "📈 Visualizations",
                "🎯 Pareto Analysis",
                "📉 Run Charts",
                "🏭 Manufacturing Analytics"
            ])

            with tab1:
                st.markdown('<div class="analysis-section">', unsafe_allow_html=True)
                st.header("🤖 AI Data Overview Team")

                # Basic statistics
                basic_stats = analyzer.basic_stats_analysis()

                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Total Rows", basic_stats['shape'][0])
                with col2:
                    st.metric("Total Columns", basic_stats['shape'][1])
                with col3:
                    st.metric("Numeric Columns", len(analyzer.numeric_columns))
                with col4:
                    st.metric("Categorical Columns", len(analyzer.categorical_columns))

                # Data preview
                st.subheader("📖 Data Preview")
                st.dataframe(df.head(config.MAX_PREVIEW_ROWS), use_container_width=True)

                # Export data
                st.subheader("💾 Export Data")
                col1, col2 = st.columns(2)
                with col1:
                    csv_data = export_dataframe_to_csv(df)
                    st.download_button(
                        label="📥 Download as CSV",
                        data=csv_data,
                        file_name=get_timestamp_filename("data_export", "csv"),
                        mime="text/csv",
                        help="Export the full dataset as CSV"
                    )
                with col2:
                    excel_data = export_dataframe_to_excel(df, sheet_name="Data")
                    st.download_button(
                        label="📥 Download as Excel",
                        data=excel_data,
                        file_name=get_timestamp_filename("data_export", "xlsx"),
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                        help="Export the full dataset as Excel"
                    )

                # Missing values
                if any(basic_stats['missing_values'].values()):
                    st.subheader("⚠️ Missing Values Analysis")
                    missing_df = pd.DataFrame(list(basic_stats['missing_values'].items()),
                                            columns=['Column', 'Missing Count'])
                    missing_df = missing_df[missing_df['Missing Count'] > 0]
                    st.bar_chart(missing_df.set_index('Column'))

                st.markdown('</div>', unsafe_allow_html=True)

            with tab2:
                st.header("🧮 AI Statistical Analysis Team")

                # Descriptive statistics
                if analyzer.numeric_columns:
                    st.subheader("📈 Descriptive Statistics")
                    st.dataframe(basic_stats['numeric_summary'], use_container_width=True)

                # Correlation analysis
                corr_analysis = analyzer.correlation_analysis()
                if corr_analysis:
                    st.subheader("🔗 Correlation Analysis")

                    # Correlation heatmap
                    fig = px.imshow(
                        corr_analysis['matrix'],
                        title="Correlation Matrix",
                        color_continuous_scale='RdBu_r',
                        aspect="auto"
                    )
                    st.plotly_chart(fig, use_container_width=True)

                    # Strong correlations
                    if corr_analysis['strong_correlations']:
                        st.subheader(f"💪 Strong Correlations (|r| > {config.CORRELATION_THRESHOLD})")
                        for corr in corr_analysis['strong_correlations'][:5]:
                            st.write(f"**{corr['var1']}** ↔ **{corr['var2']}**: "
                                   f"{corr['correlation']:.3f}")

                # Outlier analysis
                outliers = analyzer.outlier_analysis()
                if outliers:
                    st.subheader("🎯 Outlier Detection")
                    for col, info in outliers.items():
                        st.write(f"**{col}**: {info['count']} outliers "
                               f"({info['percentage']:.1f}%)")

                # Export analysis report
                st.subheader("💾 Export Analysis Report")
                report_data = export_analysis_report(basic_stats, corr_analysis, outliers)
                st.download_button(
                    label="📥 Download Complete Analysis Report (Excel)",
                    data=report_data,
                    file_name=get_timestamp_filename("analysis_report", "xlsx"),
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    help="Export comprehensive analysis with all statistics, correlations, and outliers"
                )

            with tab3:
                st.header("📊 AI Visualization Team")

                if analyzer.numeric_columns:
                    # Distribution plots
                    st.subheader("📊 Distribution Analysis")
                    selected_numeric = st.selectbox("Select numeric column",
                                                   analyzer.numeric_columns)

                    col1, col2 = st.columns(2)
                    with col1:
                        fig = px.histogram(df, x=selected_numeric,
                                         title=f"Histogram of {selected_numeric}")
                        st.plotly_chart(fig, use_container_width=True)

                    with col2:
                        fig = px.box(df, y=selected_numeric,
                                   title=f"Box Plot of {selected_numeric}")
                        st.plotly_chart(fig, use_container_width=True)

                if len(analyzer.numeric_columns) >= 2:
                    # Scatter plots
                    st.subheader("🔍 Relationship Analysis")
                    col1, col2 = st.columns(2)
                    with col1:
                        x_var = st.selectbox("X-axis variable", analyzer.numeric_columns,
                                           key="x_scatter")
                    with col2:
                        y_var = st.selectbox("Y-axis variable", analyzer.numeric_columns,
                                           key="y_scatter")

                    if analyzer.categorical_columns:
                        color_var = st.selectbox("Color by (optional)",
                                               [None] + analyzer.categorical_columns)
                        fig = px.scatter(df, x=x_var, y=y_var, color=color_var,
                                       title=f"{y_var} vs {x_var}")
                    else:
                        fig = px.scatter(df, x=x_var, y=y_var,
                                       title=f"{y_var} vs {x_var}")

                    st.plotly_chart(fig, use_container_width=True)

            with tab4:
                st.header("🎯 Pareto Analysis")
                st.markdown("*Identify the vital few from the trivial many*")

                if analyzer.categorical_columns:
                    category_col = st.selectbox("Select category column",
                                              analyzer.categorical_columns)

                    # Option to use frequency or another column for values
                    analysis_type = st.radio("Analysis type:",
                                           ["Frequency Count", "Sum of Values"])

                    if analysis_type == "Sum of Values" and analyzer.numeric_columns:
                        value_col = st.selectbox("Select value column", analyzer.numeric_columns)
                    else:
                        value_col = None

                    if st.button("Generate Pareto Chart"):
                        fig = create_pareto_chart(df, category_col, value_col)
                        if fig:
                            st.plotly_chart(fig, use_container_width=True)

                            # 80/20 insights
                            st.subheader("🎯 80/20 Rule Insights")
                            if value_col is None:
                                pareto_data = df[category_col].value_counts()
                            else:
                                pareto_data = df.groupby(category_col)[value_col].sum().sort_values(
                                    ascending=False)

                            cumsum = pareto_data.cumsum()
                            total = pareto_data.sum()
                            top_20_count = max(1, int(len(pareto_data) * 0.2))
                            top_20_contribution = (cumsum.iloc[top_20_count-1] / total) * 100

                            st.info(f"Top 20% of categories contribute "
                                  f"{top_20_contribution:.1f}% of total value")
                else:
                    st.warning("No categorical columns found for Pareto analysis")

            with tab5:
                st.header("📉 Run Chart Analysis")
                st.markdown("*Monitor process performance over time*")

                # Date column selection
                potential_date_cols = [col for col in df.columns
                                     if 'date' in col.lower() or 'time' in col.lower()]
                if not potential_date_cols:
                    potential_date_cols = df.columns.tolist()

                date_col = st.selectbox("Select date/time column", potential_date_cols)

                if analyzer.numeric_columns:
                    value_col = st.selectbox("Select value column", analyzer.numeric_columns)

                    if st.button("Generate Run Chart"):
                        fig = create_run_chart(df, date_col, value_col)
                        if fig:
                            st.plotly_chart(fig, use_container_width=True)

                            # Control chart insights
                            st.subheader("📊 Process Control Insights")
                            insights = calculate_run_chart_insights(df, date_col, value_col)

                            if insights:
                                col1, col2, col3 = st.columns(3)
                                with col1:
                                    st.metric("Process Mean", f"{insights['mean']:.2f}")
                                with col2:
                                    st.metric("Standard Deviation", f"{insights['std']:.2f}")
                                with col3:
                                    st.metric("Out of Control Points",
                                            insights['out_of_control_count'])

                                if insights['out_of_control_count'] > 0:
                                    st.warning(
                                        f"⚠️ {insights['out_of_control_count']} points are "
                                        "outside control limits - investigate special causes!"
                                    )
                                else:
                                    st.success("✅ Process appears to be in statistical control")
                else:
                    st.warning("No numeric columns found for run chart analysis")

            with tab6:
                st.header("🏭 Manufacturing Time Analytics")
                st.markdown("*Analyze production timestamps and inter-arrival times*")

                # Timestamp column selection
                potential_timestamp_cols = [
                    col for col in df.columns
                    if any(keyword in col.lower()
                          for keyword in ['time', 'date', 'stamp', 'created', 'produced'])
                ]
                if not potential_timestamp_cols:
                    potential_timestamp_cols = df.columns.tolist()

                timestamp_col = st.selectbox("Select timestamp column",
                                           potential_timestamp_cols,
                                           help="Column containing production timestamps")

                # Optional part ID column
                part_id_col = st.selectbox("Select part/product ID column (optional)",
                                         [None] + df.columns.tolist(),
                                         help="Column identifying different parts or products")

                if st.button("🔍 Analyze Manufacturing Times"):
                    with st.spinner("Analyzing production timestamps..."):
                        mfg_analysis = analyzer.manufacturing_time_analysis(timestamp_col,
                                                                           part_id_col)

                        if mfg_analysis:
                            # Key Performance Indicators
                            st.subheader("⚡ Production KPIs")
                            col1, col2, col3, col4 = st.columns(4)

                            with col1:
                                st.metric("Avg Cycle Time",
                                        f"{mfg_analysis['cycle_time_stats']['mean_minutes']:.1f} min")
                            with col2:
                                st.metric("Parts per Hour",
                                        f"{mfg_analysis['production_metrics']['parts_per_hour']:.1f}")
                            with col3:
                                st.metric("Production Stoppages",
                                        mfg_analysis['stoppage_analysis']['stoppage_count'])
                            with col4:
                                st.metric("Performance Efficiency",
                                        f"{mfg_analysis['production_metrics']['performance_efficiency']:.1f}%")

                            # Cycle Time Statistics
                            st.subheader("⏱️ Cycle Time Analysis")
                            col1, col2 = st.columns(2)

                            with col1:
                                st.write("**Cycle Time Statistics:**")
                                stats_data = {
                                    "Metric": ["Mean", "Median", "Std Dev", "Min", "Max"],
                                    "Seconds": [
                                        f"{mfg_analysis['cycle_time_stats']['mean_seconds']:.1f}",
                                        f"{mfg_analysis['cycle_time_stats']['median_seconds']:.1f}",
                                        f"{mfg_analysis['cycle_time_stats']['std_seconds']:.1f}",
                                        f"{mfg_analysis['cycle_time_stats']['min_seconds']:.1f}",
                                        f"{mfg_analysis['cycle_time_stats']['max_seconds']:.1f}"
                                    ],
                                    "Minutes": [
                                        f"{mfg_analysis['cycle_time_stats']['mean_minutes']:.2f}",
                                        f"{mfg_analysis['cycle_time_stats']['median_minutes']:.2f}",
                                        f"{mfg_analysis['cycle_time_stats']['std_seconds']/60:.2f}",
                                        f"{mfg_analysis['cycle_time_stats']['min_seconds']/60:.2f}",
                                        f"{mfg_analysis['cycle_time_stats']['max_seconds']/60:.2f}"
                                    ]
                                }
                                st.dataframe(pd.DataFrame(stats_data), use_container_width=True)

                            with col2:
                                # Cycle time distribution
                                fig = px.histogram(x=mfg_analysis['raw_time_diffs'],
                                                 title="Cycle Time Distribution",
                                                 labels={'x': 'Cycle Time (seconds)', 'y': 'Frequency'})
                                st.plotly_chart(fig, use_container_width=True)

                            # Stoppage Analysis
                            if mfg_analysis['stoppage_analysis']['stoppage_count'] > 0:
                                st.subheader("🛑 Production Stoppage Analysis")
                                col1, col2, col3 = st.columns(3)

                                with col1:
                                    st.metric("Total Stoppage Time",
                                            f"{mfg_analysis['stoppage_analysis']['total_stoppage_time']/3600:.1f} hours")
                                with col2:
                                    st.metric("Avg Stoppage Duration",
                                            f"{mfg_analysis['stoppage_analysis']['avg_stoppage_duration']/60:.1f} min")
                                with col3:
                                    st.metric("Stoppage Threshold",
                                            f"{mfg_analysis['stoppage_analysis']['stoppage_threshold_seconds']/60:.1f} min")

                                st.warning(f"⚠️ Detected {mfg_analysis['stoppage_analysis']['stoppage_count']} production stoppages")
                            else:
                                st.success("✅ No significant production stoppages detected")

                            # Shift Analysis
                            st.subheader("🕐 Shift Performance Analysis")
                            shift_data = mfg_analysis['shift_analysis']
                            if not shift_data.empty:
                                # Add calculated columns
                                shift_data['Avg_Minutes'] = shift_data['mean'] / 60
                                shift_data['Parts_per_Hour'] = 3600 / shift_data['mean']

                                st.dataframe(shift_data.round(2), use_container_width=True)

                                # Shift comparison chart
                                fig = px.bar(x=shift_data.index, y=shift_data['Parts_per_Hour'],
                                           title="Production Rate by Shift",
                                           labels={'x': 'Shift', 'y': 'Parts per Hour'})
                                st.plotly_chart(fig, use_container_width=True)

                            # Part-specific analysis
                            if mfg_analysis['part_analysis'] is not None:
                                st.subheader("🔧 Part-Specific Analysis")
                                part_data = mfg_analysis['part_analysis']
                                part_data['Avg_Minutes'] = part_data['Avg_Cycle_Time'] / 60
                                part_data['Parts_per_Hour'] = 3600 / part_data['Avg_Cycle_Time']

                                st.dataframe(part_data.round(2), use_container_width=True)

                                # Part performance comparison
                                fig = px.bar(x=part_data.index, y=part_data['Parts_per_Hour'],
                                           title="Production Rate by Part Type",
                                           labels={'x': 'Part ID', 'y': 'Parts per Hour'})
                                st.plotly_chart(fig, use_container_width=True)

                            # Manufacturing Dashboard
                            st.subheader("📊 Manufacturing Performance Dashboard")
                            dashboard_fig = create_manufacturing_dashboard(mfg_analysis)
                            if dashboard_fig:
                                st.plotly_chart(dashboard_fig, use_container_width=True)

                            # Time Series Analysis
                            st.subheader("📈 Production Timeline")
                            ts_data = mfg_analysis['time_series_data'].dropna()
                            if len(ts_data) > 1:
                                # Production timeline with cycle times
                                fig = px.line(ts_data, x=ts_data.index, y='time_diff_seconds',
                                            title="Cycle Time Over Production Sequence",
                                            labels={'x': 'Production Sequence', 'y': 'Cycle Time (seconds)'})
                                st.plotly_chart(fig, use_container_width=True)

                                # Hourly production pattern
                                hourly_stats = ts_data.groupby('hour')['time_diff_seconds'].agg(
                                    ['mean', 'count']).reset_index()
                                hourly_stats['parts_per_hour'] = 3600 / hourly_stats['mean']

                                fig = px.line(hourly_stats, x='hour', y='parts_per_hour',
                                            title="Production Rate by Hour of Day",
                                            labels={'hour': 'Hour of Day', 'parts_per_hour': 'Parts per Hour'})
                                st.plotly_chart(fig, use_container_width=True)

                            # Recommendations
                            st.subheader("💡 AI Recommendations")
                            recommendations = generate_manufacturing_recommendations(mfg_analysis)
                            for rec in recommendations:
                                st.write(f"• {rec}")

                            # Export manufacturing report
                            st.subheader("💾 Export Manufacturing Report")
                            mfg_report_data = export_manufacturing_report(mfg_analysis)
                            st.download_button(
                                label="📥 Download Manufacturing Analysis Report (Excel)",
                                data=mfg_report_data,
                                file_name=get_timestamp_filename("manufacturing_report", "xlsx"),
                                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                                help="Export comprehensive manufacturing analysis with KPIs, shift data, and part analysis"
                            )

                        else:
                            st.error("❌ Could not analyze manufacturing times. "
                                   "Please check your timestamp column format.")

                # Sample manufacturing data generator
                st.subheader("🎲 Generate Sample Manufacturing Data")
                if st.button("Create Sample Production Data"):
                    sample_mfg_df = generate_sample_manufacturing_data()
                    st.session_state['sample_mfg_data'] = sample_mfg_df

                    st.success("✅ Sample manufacturing data generated!")
                    st.info("💡 Use 'timestamp' as your timestamp column and 'part_id' for part analysis")
                    st.dataframe(sample_mfg_df.head(config.MAX_PREVIEW_ROWS),
                               use_container_width=True)

        except Exception as e:
            logger.error(f"Error loading or processing file: {str(e)}")
            st.error(f"Error loading file: {str(e)}")

    else:
        # Welcome screen
        st.markdown("""
        ## 🚀 Welcome to DataScope Analytics

        Your AI-powered business intelligence platform that brings together a team of specialized AI analysts:

        ### 🤖 Our AI Analysis Team:
        - **📊 Statistical Analyst**: Descriptive statistics, distributions, and data quality
        - **🔗 Correlation Specialist**: Relationship analysis and feature interactions
        - **🎯 Outlier Detective**: Anomaly detection and data validation
        - **📈 Trend Analyst**: Time series analysis and forecasting
        - **🎨 Visualization Expert**: Interactive charts and dashboards
        - **🏭 Manufacturing Time Analyst**: Production timestamps and cycle time analysis

        ### 📈 Advanced Analytics Features:
        - **Pareto Charts**: 80/20 analysis to identify key drivers
        - **Run Charts**: Process control and performance monitoring
        - **Manufacturing Analytics**: Inter-arrival times, cycle time analysis, shift performance
        - **Production KPIs**: OEE metrics, stoppage detection, performance efficiency
        - **Interactive Dashboards**: Drill-down capabilities
        - **Statistical Modeling**: Correlation, regression, clustering

        ### 🔧 Supported Formats:
        - CSV files
        - Excel files (.xlsx, .xls)
        - Automatic data type detection
        - Missing value handling

        **👈 Upload your data file using the sidebar to get started!**
        """)

        # Sample data generator
        st.subheader("🎲 Try with Sample Data")
        if st.button("Generate Sample Sales Data"):
            sample_df = generate_sample_sales_data()
            st.session_state['sample_data'] = sample_df

            st.success("Sample data generated! Use the analysis tabs above to explore.")
            st.dataframe(sample_df.head(), use_container_width=True)


if __name__ == "__main__":
    main()
