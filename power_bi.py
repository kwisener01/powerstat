import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import io
import base64

# Configure page
st.set_page_config(
    page_title="DataScope Analytics",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        color: #1f77b4;
        margin-bottom: 2rem;
    }
    .metric-container {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 10px;
        margin: 0.5rem 0;
    }
    .analysis-section {
        border-left: 4px solid #1f77b4;
        padding-left: 1rem;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

class DataAnalyzer:
    """AI-powered data analysis team simulator"""
    
    def __init__(self, df):
        self.df = df
        self.numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
        self.categorical_columns = df.select_dtypes(include=['object']).columns.tolist()
    
    def basic_stats_analysis(self):
        """Statistical Analyst AI"""
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
        
        return analysis
    
    def correlation_analysis(self):
        """Correlation Specialist AI"""
        if len(self.numeric_columns) < 2:
            return None
        
        corr_matrix = self.df[self.numeric_columns].corr()
        
        # Find strongest correlations
        corr_pairs = []
        for i in range(len(corr_matrix.columns)):
            for j in range(i+1, len(corr_matrix.columns)):
                corr_val = corr_matrix.iloc[i, j]
                if abs(corr_val) > 0.5:  # Only strong correlations
                    corr_pairs.append({
                        'var1': corr_matrix.columns[i],
                        'var2': corr_matrix.columns[j],
                        'correlation': corr_val
                    })
        
        return {
            'matrix': corr_matrix,
            'strong_correlations': sorted(corr_pairs, key=lambda x: abs(x['correlation']), reverse=True)
        }
    
    def outlier_analysis(self):
        """Outlier Detection AI"""
        outliers = {}
        
        for col in self.numeric_columns:
            Q1 = self.df[col].quantile(0.25)
            Q3 = self.df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            outlier_mask = (self.df[col] < lower_bound) | (self.df[col] > upper_bound)
            outlier_count = outlier_mask.sum()
            
            if outlier_count > 0:
                outliers[col] = {
                    'count': outlier_count,
                    'percentage': (outlier_count / len(self.df)) * 100,
                    'values': self.df[outlier_mask][col].tolist()
                }
        
        return outliers
    
    def trend_analysis(self, date_col=None, value_col=None):
        """Time Series Analyst AI"""
        if date_col is None or value_col is None:
            return None
        
        try:
            df_trend = self.df.copy()
            df_trend[date_col] = pd.to_datetime(df_trend[date_col])
            df_trend = df_trend.sort_values(date_col)
            
            # Calculate trend
            x = np.arange(len(df_trend))
            slope, intercept, r_value, p_value, std_err = stats.linregress(x, df_trend[value_col])
            
            return {
                'slope': slope,
                'r_squared': r_value**2,
                'p_value': p_value,
                'trend_direction': 'Increasing' if slope > 0 else 'Decreasing',
                'data': df_trend[[date_col, value_col]]
            }
        except:
            return None
    
    def manufacturing_time_analysis(self, timestamp_col, part_id_col=None):
        """Manufacturing Time Analyst AI - Specialized for production timestamps"""
        try:
            df_mfg = self.df.copy()
            df_mfg[timestamp_col] = pd.to_datetime(df_mfg[timestamp_col])
            df_mfg = df_mfg.sort_values(timestamp_col)
            
            # Calculate time differences between consecutive parts
            df_mfg['time_diff_seconds'] = df_mfg[timestamp_col].diff().dt.total_seconds()
            df_mfg['time_diff_minutes'] = df_mfg['time_diff_seconds'] / 60
            df_mfg['time_diff_hours'] = df_mfg['time_diff_minutes'] / 60
            
            # Remove first row (NaN) and filter out unrealistic values (> 24 hours)
            time_diffs = df_mfg['time_diff_seconds'].dropna()
            time_diffs = time_diffs[time_diffs <= 86400]  # 24 hours max
            
            if len(time_diffs) == 0:
                return None
            
            # Calculate production rate metrics
            mean_cycle_time = time_diffs.mean()
            median_cycle_time = time_diffs.median()
            std_cycle_time = time_diffs.std()
            min_cycle_time = time_diffs.min()
            max_cycle_time = time_diffs.max()
            
            # Production rate (parts per hour)
            parts_per_hour = 3600 / mean_cycle_time if mean_cycle_time > 0 else 0
            
            # Detect production stoppages (time gaps > 3 standard deviations)
            stoppage_threshold = mean_cycle_time + 3 * std_cycle_time
            stoppages = time_diffs[time_diffs > stoppage_threshold]
            
            # Calculate OEE-related metrics
            total_production_time = (df_mfg[timestamp_col].max() - df_mfg[timestamp_col].min()).total_seconds()
            ideal_cycle_time = time_diffs.quantile(0.10)  # Best 10% performance
            theoretical_max_parts = total_production_time / ideal_cycle_time if ideal_cycle_time > 0 else 0
            actual_parts = len(df_mfg) - 1  # Subtract 1 for the first NaN
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
                part_analysis.columns = ['Count', 'Avg_Cycle_Time', 'Std_Cycle_Time', 'Min_Cycle_Time', 'Max_Cycle_Time']
            
            return {
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
                'time_series_data': df_mfg[[timestamp_col, 'time_diff_seconds', 'time_diff_minutes', 'hour', 'shift']],
                'raw_time_diffs': time_diffs
            }
        except Exception as e:
            return None

def create_pareto_chart(df, category_col, value_col=None):
    """Create Pareto Chart"""
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
    pareto_data['Cumulative_Percent'] = (pareto_data['Cumulative'] / pareto_data[value_col].sum()) * 100
    
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
    fig.update_layout(title_text="Pareto Chart", height=500)
    
    return fig

def create_manufacturing_dashboard(mfg_analysis):
    """Create manufacturing-specific visualizations"""
    if not mfg_analysis:
        return None
    
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
                    nbinsx=30, showlegend=False),
        row=1, col=1
    )
    
    # 2. Production Rate Over Time (using time series data)
    ts_data = mfg_analysis['time_series_data'].dropna()
    if len(ts_data) > 1:
        # Calculate rolling production rate
        ts_data['rolling_rate'] = 3600 / ts_data['time_diff_seconds'].rolling(window=10, min_periods=1).mean()
        
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
    ucl = mean_time + 3 * std_time
    lcl = max(0, mean_time - 3 * std_time)
    
    fig.add_trace(
        go.Scatter(x=list(range(len(time_diffs))), y=time_diffs, 
                  mode='markers', name='Cycle Time', showlegend=False),
        row=2, col=2
    )
    
    # Add control limits
    fig.add_hline(y=mean_time, line_dash="dash", line_color="green", row=2, col=2)
    fig.add_hline(y=ucl, line_dash="dash", line_color="red", row=2, col=2)
    fig.add_hline(y=lcl, line_dash="dash", line_color="red", row=2, col=2)
    
    fig.update_layout(height=800, title_text="Manufacturing Performance Dashboard")
    
    return fig
    """Create Run Chart with control limits"""
    try:
        df_run = df.copy()
        df_run[date_col] = pd.to_datetime(df_run[date_col])
        df_run = df_run.sort_values(date_col)
        
        # Calculate control limits
        mean_val = df_run[value_col].mean()
        std_val = df_run[value_col].std()
        ucl = mean_val + 3 * std_val
        lcl = mean_val - 3 * std_val
        
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
            height=500
        )
        
        return fig
    except:
        st.error("Could not create run chart. Please check your date and value columns.")
        return None

def main():
    st.markdown('<h1 class="main-header">📊 DataScope Analytics</h1>', unsafe_allow_html=True)
    st.markdown("### AI-Powered Business Intelligence & Data Analysis Platform")
    
    # Sidebar
    st.sidebar.header("🚀 Data Upload & Settings")
    
    # File upload
    uploaded_file = st.sidebar.file_uploader(
        "Upload your data file",
        type=['csv', 'xlsx', 'xls'],
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
                st.dataframe(df.head(10), use_container_width=True)
                
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
                        st.subheader("💪 Strong Correlations (|r| > 0.5)")
                        for corr in corr_analysis['strong_correlations'][:5]:
                            st.write(f"**{corr['var1']}** ↔ **{corr['var2']}**: {corr['correlation']:.3f}")
                
                # Outlier analysis
                outliers = analyzer.outlier_analysis()
                if outliers:
                    st.subheader("🎯 Outlier Detection")
                    for col, info in outliers.items():
                        st.write(f"**{col}**: {info['count']} outliers ({info['percentage']:.1f}%)")
            
            with tab3:
                st.header("📊 AI Visualization Team")
                
                if analyzer.numeric_columns:
                    # Distribution plots
                    st.subheader("📊 Distribution Analysis")
                    selected_numeric = st.selectbox("Select numeric column", analyzer.numeric_columns)
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        fig = px.histogram(df, x=selected_numeric, title=f"Histogram of {selected_numeric}")
                        st.plotly_chart(fig, use_container_width=True)
                    
                    with col2:
                        fig = px.box(df, y=selected_numeric, title=f"Box Plot of {selected_numeric}")
                        st.plotly_chart(fig, use_container_width=True)
                
                if len(analyzer.numeric_columns) >= 2:
                    # Scatter plots
                    st.subheader("🔍 Relationship Analysis")
                    col1, col2 = st.columns(2)
                    with col1:
                        x_var = st.selectbox("X-axis variable", analyzer.numeric_columns, key="x_scatter")
                    with col2:
                        y_var = st.selectbox("Y-axis variable", analyzer.numeric_columns, key="y_scatter")
                    
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
                    category_col = st.selectbox("Select category column", analyzer.categorical_columns)
                    
                    # Option to use frequency or another column for values
                    analysis_type = st.radio("Analysis type:", ["Frequency Count", "Sum of Values"])
                    
                    if analysis_type == "Sum of Values" and analyzer.numeric_columns:
                        value_col = st.selectbox("Select value column", analyzer.numeric_columns)
                    else:
                        value_col = None
                    
                    if st.button("Generate Pareto Chart"):
                        fig = create_pareto_chart(df, category_col, value_col)
                        st.plotly_chart(fig, use_container_width=True)
                        
                        # 80/20 insights
                        st.subheader("🎯 80/20 Rule Insights")
                        if value_col is None:
                            pareto_data = df[category_col].value_counts()
                        else:
                            pareto_data = df.groupby(category_col)[value_col].sum().sort_values(ascending=False)
                        
                        cumsum = pareto_data.cumsum()
                        total = pareto_data.sum()
                        top_20_count = max(1, int(len(pareto_data) * 0.2))
                        top_20_contribution = (cumsum.iloc[top_20_count-1] / total) * 100
                        
                        st.info(f"Top 20% of categories contribute {top_20_contribution:.1f}% of total value")
                else:
                    st.warning("No categorical columns found for Pareto analysis")
            
            with tab5:
                st.header("📉 Run Chart Analysis")
                st.markdown("*Monitor process performance over time*")
                
                # Date column selection
                potential_date_cols = [col for col in df.columns if 'date' in col.lower() or 'time' in col.lower()]
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
                            try:
                                df_temp = df.copy()
                                df_temp[date_col] = pd.to_datetime(df_temp[date_col])
                                df_temp = df_temp.sort_values(date_col)
                                
                                mean_val = df_temp[value_col].mean()
                                std_val = df_temp[value_col].std()
                                ucl = mean_val + 3 * std_val
                                lcl = mean_val - 3 * std_val
                                
                                out_of_control = ((df_temp[value_col] > ucl) | (df_temp[value_col] < lcl)).sum()
                                
                                col1, col2, col3 = st.columns(3)
                                with col1:
                                    st.metric("Process Mean", f"{mean_val:.2f}")
                                with col2:
                                    st.metric("Standard Deviation", f"{std_val:.2f}")
                                with col3:
                                    st.metric("Out of Control Points", out_of_control)
                                
                                if out_of_control > 0:
                                    st.warning(f"⚠️ {out_of_control} points are outside control limits - investigate special causes!")
                                else:
                                    st.success("✅ Process appears to be in statistical control")
                            except:
                                st.error("Could not calculate control statistics")
                else:
                    st.warning("No numeric columns found for run chart analysis")
            
            with tab6:
                st.header("🏭 Manufacturing Time Analytics")
                st.markdown("*Analyze production timestamps and inter-arrival times*")
                
                # Timestamp column selection
                potential_timestamp_cols = [col for col in df.columns if any(keyword in col.lower() 
                                          for keyword in ['time', 'date', 'stamp', 'created', 'produced'])]
                if not potential_timestamp_cols:
                    potential_timestamp_cols = df.columns.tolist()
                
                timestamp_col = st.selectbox("Select timestamp column", potential_timestamp_cols, 
                                           help="Column containing production timestamps")
                
                # Optional part ID column
                part_id_col = st.selectbox("Select part/product ID column (optional)", 
                                         [None] + df.columns.tolist(),
                                         help="Column identifying different parts or products")
                
                if st.button("🔍 Analyze Manufacturing Times"):
                    with st.spinner("Analyzing production timestamps..."):
                        mfg_analysis = analyzer.manufacturing_time_analysis(timestamp_col, part_id_col)
                        
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
                                hourly_stats = ts_data.groupby('hour')['time_diff_seconds'].agg(['mean', 'count']).reset_index()
                                hourly_stats['parts_per_hour'] = 3600 / hourly_stats['mean']
                                
                                fig = px.line(hourly_stats, x='hour', y='parts_per_hour',
                                            title="Production Rate by Hour of Day",
                                            labels={'hour': 'Hour of Day', 'parts_per_hour': 'Parts per Hour'})
                                st.plotly_chart(fig, use_container_width=True)
                            
                            # Recommendations
                            st.subheader("💡 AI Recommendations")
                            recommendations = []
                            
                            # Performance efficiency
                            if mfg_analysis['production_metrics']['performance_efficiency'] < 70:
                                recommendations.append("🔴 Performance efficiency is below 70%. Consider investigating bottlenecks.")
                            elif mfg_analysis['production_metrics']['performance_efficiency'] < 85:
                                recommendations.append("🟡 Performance efficiency could be improved. Look for optimization opportunities.")
                            else:
                                recommendations.append("🟢 Good performance efficiency. Maintain current processes.")
                            
                            # Cycle time variability
                            cv = mfg_analysis['cycle_time_stats']['std_seconds'] / mfg_analysis['cycle_time_stats']['mean_seconds']
                            if cv > 0.3:
                                recommendations.append("🔴 High cycle time variability detected. Investigate process consistency.")
                            elif cv > 0.15:
                                recommendations.append("🟡 Moderate cycle time variability. Consider process standardization.")
                            else:
                                recommendations.append("🟢 Low cycle time variability. Process appears stable.")
                            
                            # Stoppages
                            if mfg_analysis['stoppage_analysis']['stoppage_count'] > len(mfg_analysis['raw_time_diffs']) * 0.05:
                                recommendations.append("🔴 High number of production stoppages. Review maintenance schedules.")
                            
                            for rec in recommendations:
                                st.write(f"• {rec}")
                        
                        else:
                            st.error("❌ Could not analyze manufacturing times. Please check your timestamp column format.")
                
                # Sample manufacturing data generator
                st.subheader("🎲 Generate Sample Manufacturing Data")
                if st.button("Create Sample Production Data"):
                    # Generate realistic manufacturing timestamp data
                    start_time = pd.Timestamp('2024-01-01 06:00:00')
                    
                    # Simulate production with realistic cycle times and stoppages
                    timestamps = []
                    current_time = start_time
                    part_types = ['A001', 'A002', 'B001', 'B002', 'C001']
                    operators = ['Op1', 'Op2', 'Op3', 'Op4']
                    
                    for i in range(1000):
                        # Base cycle time varies by part type
                        part_type = np.random.choice(part_types)
                        base_cycle = {'A001': 45, 'A002': 52, 'B001': 38, 'B002': 41, 'C001': 55}[part_type]
                        
                        # Add variation and shift effects
                        hour = current_time.hour
                        shift_multiplier = 1.0
                        if 22 <= hour or hour < 6:  # Night shift slower
                            shift_multiplier = 1.15
                        elif 14 <= hour < 22:  # Evening shift
                            shift_multiplier = 1.05
                        
                        # Random variation
                        cycle_time = np.random.normal(base_cycle * shift_multiplier, base_cycle * 0.1)
                        cycle_time = max(cycle_time, base_cycle * 0.5)  # Minimum time
                        
                        # Occasional stoppages
                        if np.random.random() < 0.02:  # 2% chance of stoppage
                            cycle_time += np.random.exponential(300)  # Average 5 min stoppage
                        
                        current_time += pd.Timedelta(seconds=cycle_time)
                        
                        timestamps.append({
                            'timestamp': current_time,
                            'part_id': part_type,
                            'operator': np.random.choice(operators),
                            'machine': f'M{np.random.randint(1, 6)}',
                            'quality_score': np.random.normal(95, 3)
                        })
                    
                    sample_mfg_df = pd.DataFrame(timestamps)
                    st.session_state['sample_mfg_data'] = sample_mfg_df
                    
                    st.success("✅ Sample manufacturing data generated!")
                    st.info("💡 Use 'timestamp' as your timestamp column and 'part_id' for part analysis")
                    st.dataframe(sample_mfg_df.head(10), use_container_width=True)
        
        except Exception as e:
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
            # Create sample data
            np.random.seed(42)
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
            
            sample_df = pd.DataFrame(sample_data)
            st.session_state['sample_data'] = sample_df
            
            st.success("Sample data generated! Use the analysis tabs above to explore.")
            st.dataframe(sample_df.head(), use_container_width=True)

if __name__ == "__main__":
    main()