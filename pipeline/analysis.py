"""
pipeline/analysis.py
─────────────────────────────────────────────────────────────────────────────
The three core analyses of the Air Quality Spark pipeline.

Analysis 1 — Top 10 countries by AQI improvement (last 24h)
Analysis 2 — Top 50 geographic clusters by AQI improvement (K-Means)
Analysis 3 — Longest consecutive streak of good air quality per cluster

All functions operate on PySpark DataFrames using lazy evaluation.
No data is collected to the driver until .toPandas() or .collect() is called.
─────────────────────────────────────────────────────────────────────────────
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import date, timedelta

from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F
from pyspark.sql import Window
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.clustering import KMeans, KMeansModel

import config


# ══════════════════════════════════════════════════════════════════════════════
#  ANALYSIS 1 — Top 10 countries by AQI improvement over the last 24h
# ══════════════════════════════════════════════════════════════════════════════

def compute_aqi_improvement_by_country(df: DataFrame,
                                        date_today: str,
                                        date_yesterday: str) -> DataFrame:
    """
    Rank countries by their average AQI improvement between yesterday and today.

    Method:
      1. Filter to the two most recent days
      2. Compute average AQI per (country, day)
      3. Negate today's AQI — so summing gives (yesterday - today)
         → positive value = improvement (AQI decreased = air got better)
      4. Group by country, sum, and sort descending

    Args:
        df:             clean sensor DataFrame
        date_today:     'YYYY-MM-DD' string for today
        date_yesterday: 'YYYY-MM-DD' string for yesterday

    Returns:
        DataFrame: country, country_name, AQI_improvement — sorted best→worst
    """
    print(f"\n[Analysis 1] Computing country AQI improvement: {date_yesterday} → {date_today}")

    # Step 1: keep only the 2 days of interest
    two_days = df.filter(
        (F.col("date") == date_yesterday) | (F.col("date") == date_today)
    )

    # Step 2: average AQI per country per day
    avg_by_day = (two_days
                  .groupBy("country", "country_name", "date")
                  .agg(F.avg("AQI").alias("avg_AQI")))

    # Step 3: negate today's AQI so sum = improvement
    signed = avg_by_day.withColumn(
        "avg_AQI",
        F.when(F.col("date") == date_today, -F.col("avg_AQI"))
         .otherwise(F.col("avg_AQI"))
    )

    # Step 4: group by country, sum → improvement score, sort
    improvement = (signed
                   .groupBy("country", "country_name")
                   .agg(F.sum("avg_AQI").alias("AQI_improvement"))
                   .orderBy(F.col("AQI_improvement").desc()))

    return improvement


def worst_aqi_today(df: DataFrame, date_today: str) -> DataFrame:
    """
    Return countries ranked by worst average AQI today (for choropleth map).

    Args:
        df:         clean sensor DataFrame
        date_today: 'YYYY-MM-DD' string

    Returns:
        DataFrame: country, avg_AQI — sorted worst→best
    """
    return (df
            .filter(F.col("date") == date_today)
            .groupBy("country")
            .agg(F.avg("AQI").alias("avg_AQI"))
            .orderBy(F.col("avg_AQI").desc()))


def plot_top10_countries(improvement_df: DataFrame,
                          output_dir: str = 'output/') -> pd.DataFrame:
    """
    Plot horizontal bar chart of top 10 countries by AQI improvement.

    Returns the full improvement pandas DataFrame for downstream use.
    """
    pdf = improvement_df.toPandas()
    top10 = pdf.head(10)

    fig, ax = plt.subplots(figsize=(10, 6))
    colors = ['#2ecc71' if v >= 0 else '#e74c3c' for v in top10['AQI_improvement']]
    bars = ax.barh(top10['country_name'], top10['AQI_improvement'], color=colors)
    ax.invert_yaxis()
    ax.set_xlabel('AQI Improvement (positive = better air)')
    ax.set_title('Top 10 Countries — AQI Improvement in Last 24h', fontsize=14, fontweight='bold')
    ax.axvline(x=0, color='grey', linewidth=0.8, linestyle='--')

    # Value labels
    for bar, val in zip(bars, top10['AQI_improvement']):
        ax.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height() / 2,
                f'{val:+.2f}', va='center', fontsize=9)

    plt.tight_layout()
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, 'top10_countries.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[Analysis 1] Chart saved: {output_dir}top10_countries.png")
    return pdf


# ══════════════════════════════════════════════════════════════════════════════
#  ANALYSIS 2 — Top 50 geographic clusters by AQI improvement
# ══════════════════════════════════════════════════════════════════════════════

def train_or_load_kmeans(df: DataFrame, model_path: str = 'kmeans_model') -> KMeansModel:
    """
    Train a K-Means model on sensor GPS coordinates, or load a saved one.

    Training uses one unique (lat, lon) per sensor to avoid bias from
    sensors with more readings. The model is saved to disk after training
    so subsequent runs don't need to retrain.

    Args:
        df:         clean sensor DataFrame
        model_path: path to save/load the trained model

    Returns:
        Trained or loaded KMeansModel
    """
    assembler = VectorAssembler(
        inputCols=['latitude', 'longitude'],
        outputCol='geo_coordinates'
    )

    if os.path.exists(model_path):
        print(f"[Analysis 2] Loading existing K-Means model from {model_path}")
        return KMeansModel.load(model_path)

    print(f"[Analysis 2] Training K-Means with K={config.K_CLUSTERS} clusters...")

    # One point per unique sensor location — efficient training
    training_data = (df
                     .dropDuplicates(['latitude', 'longitude'])
                     .select('latitude', 'longitude'))
    training_data = assembler.transform(training_data)

    kmeans = (KMeans()
              .setK(config.K_CLUSTERS)
              .setSeed(config.K_SEED)
              .setFeaturesCol('geo_coordinates')
              .setPredictionCol('cluster'))

    model = kmeans.fit(training_data)
    model.save(model_path)
    print(f"[Analysis 2] Model trained and saved to {model_path}")
    print(f"[Analysis 2] Training cost (inertia): {model.summary.trainingCost:.2f}")
    return model, assembler


def assign_clusters(df: DataFrame, model: KMeansModel,
                     assembler: VectorAssembler) -> DataFrame:
    """Apply the trained K-Means model to assign a cluster to every sensor reading."""
    df = assembler.transform(df)
    df = model.transform(df)
    return df.drop('geo_coordinates')


def compute_aqi_improvement_by_cluster(df: DataFrame,
                                        model: KMeansModel,
                                        date_today: str,
                                        date_yesterday: str) -> DataFrame:
    """
    Rank geographic clusters by AQI improvement (same method as countries).

    Also adds cluster centroid coordinates for map rendering.

    Returns:
        DataFrame: cluster, AQI_improvement, centroid_lat, centroid_lon
    """
    print(f"\n[Analysis 2] Computing cluster AQI improvement: {date_yesterday} → {date_today}")

    two_days = df.filter(
        (F.col("date") == date_yesterday) | (F.col("date") == date_today)
    )

    avg_by_day = (two_days
                  .groupBy("cluster", "date")
                  .agg(F.avg("AQI").alias("avg_AQI")))

    signed = avg_by_day.withColumn(
        "avg_AQI",
        F.when(F.col("date") == date_today, -F.col("avg_AQI"))
         .otherwise(F.col("avg_AQI"))
    )

    improvement = (signed
                   .groupBy("cluster")
                   .agg(F.sum("avg_AQI").alias("AQI_improvement"))
                   .orderBy(F.col("AQI_improvement").desc()))

    # Collect to pandas first, then join with centroid coords in pandas
    # (avoids spark.createDataFrame which crashes on Python 3.14)
    improvement_pdf = improvement.toPandas()

    centers = model.clusterCenters()
    centers_df = pd.DataFrame(
        [(i, float(c[0]), float(c[1])) for i, c in enumerate(centers)],
        columns=['cluster', 'centroid_lat', 'centroid_lon']
    )

    improvement_pdf = improvement_pdf.merge(centers_df, on='cluster', how='left')
    improvement_pdf = improvement_pdf.sort_values('AQI_improvement', ascending=False)

    return improvement_pdf


def plot_top50_clusters(improvement_df, output_dir: str = 'output/') -> pd.DataFrame:
    """Plot horizontal bar chart of top 50 clusters by AQI improvement.
    Accepts either a pandas DataFrame or a Spark DataFrame."""
    if hasattr(improvement_df, 'toPandas'):
        pdf = improvement_df.toPandas()
    else:
        pdf = improvement_df
    top50 = pdf.head(config.TOP_N)

    fig, ax = plt.subplots(figsize=(12, 18))
    colors = ['#2ecc71' if v >= 0 else '#e74c3c' for v in top50['AQI_improvement']]
    ax.barh(top50['cluster'].astype(str), top50['AQI_improvement'], color=colors)
    ax.invert_yaxis()
    ax.set_xlabel('AQI Improvement (positive = better air)')
    ax.set_title(f'Top {config.TOP_N} Geographic Clusters — AQI Improvement Last 24h',
                 fontsize=14, fontweight='bold')
    ax.axvline(x=0, color='grey', linewidth=0.8, linestyle='--')

    plt.tight_layout()
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, 'top50_clusters.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[Analysis 2] Chart saved: {output_dir}top50_clusters.png")
    return pdf


def elbow_method(df: DataFrame, k_range: range = range(5, 51, 5),
                  output_dir: str = 'output/'):
    """
    Run the Elbow Method to find the optimal number of clusters.

    Trains K-Means for each K in k_range and plots the inertia curve.
    This is expensive — only run once to determine K, then set config.K_CLUSTERS.
    """
    print(f"[Analysis 2] Running Elbow Method for K in {list(k_range)}...")

    assembler = VectorAssembler(
        inputCols=['latitude', 'longitude'],
        outputCol='geo_coordinates'
    )
    training_data = (df
                     .dropDuplicates(['latitude', 'longitude'])
                     .select('latitude', 'longitude'))
    training_data = assembler.transform(training_data)

    costs = []
    for k in k_range:
        model = KMeans().setK(k).setSeed(42).setFeaturesCol('geo_coordinates').fit(training_data)
        costs.append((k, model.summary.trainingCost))
        print(f"  K={k:3d} → inertia={model.summary.trainingCost:.0f}")

    k_vals, cost_vals = zip(*costs)
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(k_vals, cost_vals, 'bo-', linewidth=2, markersize=6)
    ax.set_xlabel('Number of Clusters (K)')
    ax.set_ylabel('Inertia (within-cluster sum of squares)')
    ax.set_title('Elbow Method — Optimal K Selection', fontsize=13, fontweight='bold')
    ax.grid(True, alpha=0.3)

    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, 'elbow_curve.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[Analysis 2] Elbow curve saved: {output_dir}elbow_curve.png")


# ══════════════════════════════════════════════════════════════════════════════
#  ANALYSIS 3 — Longest consecutive streak of good air quality per cluster
# ══════════════════════════════════════════════════════════════════════════════

def compute_streaks(df: DataFrame) -> DataFrame:
    """
    Compute the longest consecutive streak of good air quality days per cluster.

    Good air quality = average AQI ≤ config.GOOD_AQI_LIMIT (default: 3)

    Algorithm (Spark Window-based, O(n log n)):
    ─────────────────────────────────────────
    Step 1: Average AQI per (cluster, date) → add LowIndex column (1=good, 0=bad)

    Step 2: Two windows:
      w1 = row_number() partitioned by cluster, ordered by date
      w2 = row_number() partitioned by (cluster, LowIndex), ordered by date
      grp = w1 - w2  → unique identifier for each consecutive streak

    Step 3: Window w3 partitioned by (cluster, LowIndex, grp):
      streak = row_number() over w3 if LowIndex==1 else 0
      Only keep the last row of each streak (transition detection with lead())

    Step 4: Max streak per cluster → final ranking

    Returns:
        DataFrame: cluster, centroid_lat, centroid_lon, max_streak,
                   streak_distribution (list of streak lengths)
    """
    print("\n[Analysis 3] Computing good air quality streaks per cluster...")

    # Step 1: daily average AQI + LowIndex flag
    daily = (df
             .groupBy("cluster", "date")
             .agg(F.avg("AQI").alias("avg_AQI"))
             .orderBy("cluster", "date")
             .withColumn("LowIndex",
                         F.when(F.col("avg_AQI") <= config.GOOD_AQI_LIMIT, 1)
                          .otherwise(0)))

    # Step 2: streak group identifier using dual-window trick
    w1 = Window.partitionBy("cluster").orderBy("date")
    w2 = Window.partitionBy("cluster", "LowIndex").orderBy("date")

    with_grp = (daily
                .withColumn("next_LowIndex", F.lead("LowIndex", 1).over(w1))
                .withColumn("grp", F.row_number().over(w1) - F.row_number().over(w2)))

    # Step 3: streak counter
    w3 = Window.partitionBy("cluster", "LowIndex", "grp").orderBy("date")

    with_streak = with_grp.withColumn(
        "streak",
        F.when(
            (F.col("LowIndex") == 0) | (F.col("LowIndex") == F.col("next_LowIndex")),
            0
        ).otherwise(F.row_number().over(w3))
    )

    # Step 4: max streak per cluster
    max_streaks = (with_streak
                   .groupBy("cluster")
                   .agg(F.max("streak").alias("max_streak"))
                   .orderBy(F.col("max_streak").desc()))

    return with_streak, max_streaks


def streak_distribution(with_streak: DataFrame) -> pd.DataFrame:
    """
    Compute the full distribution of streak lengths across all clusters.

    Returns a pandas DataFrame suitable for histogram plotting.
    """
    dist = (with_streak
            .filter(F.col("streak") > 0)
            .groupBy("streak")
            .agg(F.count("streak").alias("count"))
            .orderBy("streak"))
    return dist.toPandas()


def plot_streak_distribution(dist_pdf: pd.DataFrame, output_dir: str = 'output/'):
    """Plot global histogram of good air quality streak lengths."""
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(dist_pdf['streak'], dist_pdf['count'],
           color='#3498db', edgecolor='white', linewidth=0.5)
    ax.set_xlabel('Streak Length (days)')
    ax.set_ylabel('Frequency')
    ax.set_title('Distribution of Good Air Quality Streaks — All Clusters',
                 fontsize=13, fontweight='bold')
    ax.grid(axis='y', alpha=0.3)

    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, 'streak_distribution.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[Analysis 3] Streak distribution chart saved: {output_dir}streak_distribution.png")


def plot_streak_per_cluster(with_streak: DataFrame, cluster_id: int,
                              output_dir: str = 'output/'):
    """Plot streak histogram for a single cluster."""
    pdf = (with_streak
           .filter((F.col("cluster") == cluster_id) & (F.col("streak") > 0))
           .select("streak")
           .toPandas())

    if pdf.empty:
        print(f"[Analysis 3] No good air days found for cluster {cluster_id}")
        return

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(pdf['streak'], bins=range(0, int(pdf['streak'].max()) + 2),
            color='#2ecc71', edgecolor='white')
    ax.set_xlabel('Streak Length (days)')
    ax.set_ylabel('Frequency')
    ax.set_title(f'Good Air Quality Streak Distribution — Cluster {cluster_id}',
                 fontsize=12, fontweight='bold')
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, f'streak_cluster_{cluster_id}.png'),
                dpi=150, bbox_inches='tight')
    plt.close()
