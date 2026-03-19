"""
config.py — Central configuration for the Air Quality Spark pipeline.
Edit the values below to match your environment.
"""

from datetime import date, timedelta

# ── Spark ─────────────────────────────────────────────────────────────────────
SPARK_MASTER   = 'local[*]'          # 'local[*]' for local mode, or spark://host:7077
SPARK_APP_NAME = 'AirQualitySpark'

# ── Data ──────────────────────────────────────────────────────────────────────
DATA_DIR          = 'data/'           # directory containing JSON sensor files
COUNTRY_CODE_FILE = 'data/country_codes.json'
OUTPUT_DIR        = 'output/'         # directory for HTML maps and charts

# ── Analysis dates ────────────────────────────────────────────────────────────
# The pipeline analyses AQI improvement between yesterday and today.
# Defaults to the last day in the generated dataset.
DATE_TODAY      = str(date.today())
DATE_YESTERDAY  = str(date.today() - timedelta(days=1))

# ── AQI thresholds ────────────────────────────────────────────────────────────
# AQI is computed from PM10 (P1) and PM2.5 (P2) using the European CAQI scale.
# Score 1 = excellent, 10 = very bad
PM10_THRESHOLDS  = [16, 33, 50, 58, 66, 75, 83, 91, 100]
PM25_THRESHOLDS  = [11, 23, 35, 41, 47, 53, 58, 64, 70]
GOOD_AQI_LIMIT   = 3    # AQI ≤ 3 is considered "good air quality"

# ── Clustering ────────────────────────────────────────────────────────────────
K_CLUSTERS  = 100   # number of geographic clusters for K-Means
K_SEED      = 42    # reproducibility seed
TOP_N       = 50    # top N clusters to display

# ── Data generator (synthetic data) ──────────────────────────────────────────
N_SENSORS          = 600    # number of simulated sensors worldwide
N_DAYS             = 22     # number of days to simulate
READINGS_PER_DAY   = 3      # readings per sensor per day
