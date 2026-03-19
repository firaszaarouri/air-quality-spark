"""
pipeline/ingestion.py
─────────────────────────────────────────────────────────────────────────────
Spark data ingestion and preprocessing pipeline.

Steps:
  1. Read all JSON sensor files from the data directory (lazy, single read)
  2. Select only required columns via Spark SQL
  3. Clean data (remove duplicates, filter invalid values)
  4. Convert PM10/PM2.5 to AQI (1–10 scale, European CAQI)
  5. Join with country codes to get alpha3 codes and english names
─────────────────────────────────────────────────────────────────────────────
"""

import sys
from pyspark.sql import SparkSession, DataFrame
from pyspark.sql import functions as F
from pyspark.sql.types import IntegerType

import config


def create_spark_session() -> SparkSession:
    """Create and return a configured SparkSession."""
    spark = (
        SparkSession.builder
        .master(config.SPARK_MASTER)
        .appName(config.SPARK_APP_NAME)
        .config("spark.sql.shuffle.partitions", "50")       # optimised for medium datasets
        .config("spark.driver.memory", "2g")
        .config("spark.ui.showConsoleProgress", "false")    # cleaner console output
        .getOrCreate()
    )
    spark.sparkContext.setLogLevel("ERROR")
    return spark


def load_raw_data(spark: SparkSession, data_path: str) -> DataFrame:
    """
    Load all JSON sensor files from the data directory.

    The JSON files are read only ONCE and registered as a temp view —
    all subsequent queries run against this in-memory representation.

    Args:
        spark:     active SparkSession
        data_path: directory containing JSON files (one per day)

    Returns:
        sensors_df: raw DataFrame with full JSON schema
    """
    print(f"[Ingestion] Loading data from: {data_path}")

    try:
        raw_df = spark.read.json(data_path)
    except Exception as e:
        print(f"[Error] Could not read data from {data_path}: {e}")
        sys.exit(1)

    raw_df.createOrReplaceTempView("sensors_raw")
    print(f"[Ingestion] Raw records loaded: {raw_df.count():,}")
    return raw_df


def extract_columns(spark: SparkSession) -> DataFrame:
    """
    Extract only the required columns from the raw sensor data using Spark SQL.

    Filters to rows where the first value_type is P1 (PM10) and second is P2 (PM2.5).
    This WHERE clause is evaluated first, which minimises the data carried forward.

    Returns:
        DataFrame with columns: country, latitude, longitude, sensor_id, date, P1, P2
    """
    df = spark.sql("""
        SELECT
            location.country                            AS country,
            DOUBLE(location.latitude)                   AS latitude,
            DOUBLE(location.longitude)                  AS longitude,
            sensor.id                                   AS sensor_id,
            CAST(timestamp AS DATE)                     AS date,
            DOUBLE(sensordatavalues.value[0])           AS P1,
            DOUBLE(sensordatavalues.value[1])           AS P2
        FROM sensors_raw
        WHERE sensordatavalues.value_type[0] = 'P1'
          AND sensordatavalues.value_type[1] = 'P2'
    """)

    # Rename for clarity
    df = (df
          .withColumnRenamed("location.latitude",  "latitude")
          .withColumnRenamed("location.longitude", "longitude"))

    return df


def clean_data(df: DataFrame) -> DataFrame:
    """
    Apply data quality rules:
      - Remove duplicate readings (same sensor, same timestamp)
      - Remove rows with null coordinates or null PM values
      - Filter out physically impossible PM values (PM10 < 0 or > 1000)

    Returns:
        Cleaned DataFrame
    """
    df = df.dropDuplicates(['sensor_id', 'date', 'latitude', 'longitude'])
    df = df.dropna(subset=['latitude', 'longitude', 'P1', 'P2', 'date', 'country'])
    df = df.filter((F.col('P1') > 0) & (F.col('P1') < 1000))
    df = df.filter((F.col('P2') > 0) & (F.col('P2') < 500))
    return df


def convert_pm_to_aqi(df: DataFrame) -> DataFrame:
    """
    Convert PM10 (P1) and PM2.5 (P2) to AQI using the European CAQI scale.

    Scale: 1 (excellent) → 10 (very bad)

    PM10 thresholds (µg/m³):  16, 33, 50, 58, 66, 75, 83, 91, 100, >100
    PM2.5 thresholds (µg/m³): 11, 23, 35, 41, 47, 53, 58, 64, 70,  >70

    AQI is the maximum of AQI_PM10 and AQI_PM25 (worst pollutant wins).
    """
    p1 = F.col("P1")
    p2 = F.col("P2")

    def pm_to_score(col, thresholds):
        expr = F.when(col <= thresholds[0], 1)
        for i, t in enumerate(thresholds[1:], 2):
            expr = expr.when(col <= t, i)
        return expr.otherwise(10)

    aqi_p1 = pm_to_score(p1, config.PM10_THRESHOLDS)
    aqi_p2 = pm_to_score(p2, config.PM25_THRESHOLDS)

    df = df.withColumn("AQI", F.greatest(aqi_p1, aqi_p2).cast(IntegerType()))
    df = df.drop("P1", "P2")
    return df


def join_country_codes(spark: SparkSession, df: DataFrame,
                        country_file: str) -> DataFrame:
    """
    Join the sensor DataFrame with a country code lookup table.

    Converts alpha-2 codes (e.g. "FR") to:
      - alpha-3 codes  (e.g. "FRA") — required for Folium choropleth maps
      - English names  (e.g. "France") — used in bar charts

    Args:
        spark:        active SparkSession
        df:           sensor DataFrame with 'country' column (alpha2)
        country_file: path to country codes JSON file

    Returns:
        DataFrame with country column replaced by alpha3 code,
        and new column 'country_name'
    """
    try:
        codes_df = spark.read.json(country_file, multiLine=True)
    except Exception:
        print("[Warning] Country code file not found — skipping join")
        return df.withColumn("country_name", F.col("country"))

    codes_df = codes_df.select(
        F.col("alpha2Code").alias("alpha2"),
        F.col("alpha3Code").alias("alpha3"),
        F.col("englishShortName").alias("country_name")
    )

    df = (df
          .join(codes_df, df.country == codes_df.alpha2, how="left")
          .drop("country", "alpha2")
          .withColumnRenamed("alpha3", "country"))

    return df


def build_sensor_dataframe(spark: SparkSession) -> DataFrame:
    """
    Full ingestion pipeline — call this from main.py.

    Returns a clean, AQI-enriched DataFrame ready for analysis:
        country, country_name, latitude, longitude, sensor_id, date, AQI
    """
    load_raw_data(spark, config.DATA_DIR)
    df = extract_columns(spark)
    df = clean_data(df)
    df = convert_pm_to_aqi(df)
    df = join_country_codes(spark, df, config.COUNTRY_CODE_FILE)

    # Cache — this DataFrame is used by all 3 analyses
    df.cache()
    print(f"[Ingestion] Clean records: {df.count():,}")
    print(f"[Ingestion] Date range: {df.agg(F.min('date'), F.max('date')).collect()[0]}")
    return df
