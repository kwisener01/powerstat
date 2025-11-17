"""
Configuration settings for DataScope Analytics application.
"""

# Application Settings
APP_TITLE = "DataScope Analytics"
APP_ICON = "📊"
PAGE_LAYOUT = "wide"
SIDEBAR_STATE = "expanded"

# Styling
MAIN_HEADER_STYLE = """
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
"""

# Analysis Thresholds
CORRELATION_THRESHOLD = 0.5  # Minimum correlation to report as "strong"
OUTLIER_IQR_MULTIPLIER = 1.5  # Standard IQR multiplier for outlier detection
STOPPAGE_SIGMA_THRESHOLD = 3  # Sigma multiplier for detecting production stoppages

# Shift Definitions (hours)
SHIFT_DEFINITIONS = {
    'Day Shift (6-14)': (6, 14),
    'Evening Shift (14-22)': (14, 22),
    'Night Shift (22-6)': (22, 6)
}

# Control Chart Settings
CONTROL_LIMIT_SIGMA = 3  # Standard deviation multiplier for control limits

# Manufacturing Analytics
MAX_CYCLE_TIME_HOURS = 24  # Maximum reasonable cycle time (filter outliers)
IDEAL_PERFORMANCE_QUANTILE = 0.10  # Use best 10% for ideal cycle time calculation

# Performance Efficiency Thresholds
PERFORMANCE_EXCELLENT = 85  # Percent
PERFORMANCE_GOOD = 70  # Percent

# Cycle Time Variability Thresholds (Coefficient of Variation)
CV_LOW = 0.15  # Good process stability
CV_MODERATE = 0.30  # Acceptable variability

# Stoppage Detection
STOPPAGE_PERCENTAGE_THRESHOLD = 0.05  # 5% of production runs

# Chart Settings
DEFAULT_CHART_HEIGHT = 500
DASHBOARD_CHART_HEIGHT = 800
HISTOGRAM_BINS = 30
ROLLING_WINDOW_SIZE = 10  # For rolling averages in charts

# File Upload Settings
SUPPORTED_FILE_TYPES = ['csv', 'xlsx', 'xls']
MAX_PREVIEW_ROWS = 10

# Sample Data Settings
SAMPLE_DATA_START_DATE = '2024-01-01 06:00:00'
SAMPLE_DATA_ROWS = 1000
SAMPLE_PART_TYPES = ['A001', 'A002', 'B001', 'B002', 'C001']
SAMPLE_OPERATORS = ['Op1', 'Op2', 'Op3', 'Op4']
SAMPLE_MACHINES = 5
SAMPLE_BASE_CYCLE_TIMES = {
    'A001': 45,
    'A002': 52,
    'B001': 38,
    'B002': 41,
    'C001': 55
}
SAMPLE_STOPPAGE_PROBABILITY = 0.02  # 2% chance
SAMPLE_AVG_STOPPAGE_DURATION = 300  # seconds (5 minutes)

# Logging
LOG_LEVEL = "INFO"
