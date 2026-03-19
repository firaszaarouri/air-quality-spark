"""
main.py
─────────────────────────────────────────────────────────────────────────────
Air Quality Spark Pipeline — Entry Point

Orchestrates the full pipeline:
  0. (Optional) Generate synthetic data
  1. Ingest and preprocess sensor JSON data
  2. Analysis 1 — Top 10 countries by AQI improvement (last 24h)
  3. Analysis 2 — Top 50 geographic clusters by AQI improvement (K-Means)
  4. Analysis 3 — Longest consecutive streak of good air quality
  5. Save all charts and interactive HTML maps to output/

Usage:
    python main.py                    # full pipeline
    python main.py --generate-data    # generate synthetic data first
    python main.py --elbow            # run elbow method to tune K
    python main.py --no-maps          # skip map generation (faster)

All outputs saved to output/ directory.
─────────────────────────────────────────────────────────────────────────────
"""

import argparse
import os
import sys
import time
from datetime import date, timedelta

import config

# ── Argument parsing ──────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(
        description='Air Quality Analysis with PySpark',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument('--generate-data', action='store_true',
                        help='Generate synthetic sensor data before running')
    parser.add_argument('--elbow', action='store_true',
                        help='Run Elbow Method to find optimal K (slow)')
    parser.add_argument('--no-maps', action='store_true',
                        help='Skip interactive map generation')
    parser.add_argument('--retrain', action='store_true',
                        help='Force retrain K-Means model (delete existing)')
    parser.add_argument('--days', type=int, default=config.N_DAYS,
                        help=f'Days of data to generate (default: {config.N_DAYS})')
    return parser.parse_args()


# ── Pipeline ──────────────────────────────────────────────────────────────────

def run(args):
    start_time = time.time()

    print("╔══════════════════════════════════════════════════════════╗")
    print("║         Air Quality Analysis — PySpark Pipeline          ║")
    print("╚══════════════════════════════════════════════════════════╝\n")

    # ── Step 0: Generate data ─────────────────────────────────────────────────
    if args.generate_data:
        print("═" * 60)
        print("Step 0 — Generating synthetic sensor data")
        print("═" * 60)
        from data_generator import generate_dataset, generate_country_codes
        config.N_DAYS = args.days
        generate_country_codes(config.DATA_DIR)
        generate_dataset(config.DATA_DIR, args.days)

    if not os.path.isdir(config.DATA_DIR) or not os.listdir(config.DATA_DIR):
        print("[Error] No data found. Run with --generate-data first.")
        sys.exit(1)

    # ── Step 1: Ingest data ───────────────────────────────────────────────────
    print("\n" + "═" * 60)
    print("Step 1 — Data Ingestion & Preprocessing")
    print("═" * 60)

    from pipeline.ingestion import create_spark_session, build_sensor_dataframe
    spark = create_spark_session()
    df    = build_sensor_dataframe(spark)

    # Determine analysis dates from actual data
    date_range = df.agg({"date": "max"}).collect()[0][0]
    date_today     = str(date_range)
    date_yesterday = str(date_range - timedelta(days=1))
    print(f"\n[Main] Analysis window: {date_yesterday} → {date_today}")

    # ── Step 2: Analysis 1 ────────────────────────────────────────────────────
    print("\n" + "═" * 60)
    print("Analysis 1 — Top 10 Countries by AQI Improvement")
    print("═" * 60)

    from pipeline.analysis import (
        compute_aqi_improvement_by_country, worst_aqi_today, plot_top10_countries
    )
    from pipeline.analysis import (
        train_or_load_kmeans, assign_clusters,
        compute_aqi_improvement_by_cluster, plot_top50_clusters, elbow_method,
        compute_streaks, streak_distribution, plot_streak_distribution,
        plot_streak_per_cluster
    )

    improvement_countries = compute_aqi_improvement_by_country(df, date_today, date_yesterday)
    worst_today           = worst_aqi_today(df, date_today)

    print("\nTop 10 countries — AQI improvement:")
    improvement_countries.show(10, truncate=False)

    pdf_countries = plot_top10_countries(improvement_countries, config.OUTPUT_DIR)

    # ── Step 3: Analysis 2 ────────────────────────────────────────────────────
    print("\n" + "═" * 60)
    print("Analysis 2 — Top 50 Geographic Clusters by AQI Improvement")
    print("═" * 60)

    # Elbow method (optional — slow)
    if args.elbow:
        elbow_method(df, k_range=range(10, 101, 10), output_dir=config.OUTPUT_DIR)

    # Force retrain if requested
    model_path = 'kmeans_model'
    if args.retrain and os.path.exists(model_path):
        import shutil
        shutil.rmtree(model_path)
        print(f"[Main] Existing model deleted — retraining")

    result = train_or_load_kmeans(df, model_path)
    if isinstance(result, tuple):
        model, assembler = result
    else:
        # Model was loaded — recreate assembler
        from pyspark.ml.feature import VectorAssembler
        model = result
        assembler = VectorAssembler(
            inputCols=['latitude', 'longitude'],
            outputCol='geo_coordinates'
        )

    # Assign clusters to full dataset
    df_clustered = assign_clusters(df, model, assembler)
    df_clustered.cache()

    improvement_clusters = compute_aqi_improvement_by_cluster(
        df_clustered, model, date_today, date_yesterday
    )

    print(f"\nTop {config.TOP_N} clusters — AQI improvement:")
    print(improvement_clusters.head(10).to_string(index=False))

    pdf_clusters = plot_top50_clusters(improvement_clusters, config.OUTPUT_DIR)

    # ── Step 4: Analysis 3 ────────────────────────────────────────────────────
    print("\n" + "═" * 60)
    print("Analysis 3 — Longest Good Air Quality Streaks")
    print("═" * 60)

    with_streak, max_streaks = compute_streaks(df_clustered)

    print("\nTop 10 clusters — longest good air quality streak:")
    max_streaks.show(10)

    dist_pdf = streak_distribution(with_streak)
    plot_streak_distribution(dist_pdf, config.OUTPUT_DIR)

    # Detailed plot for the best cluster
    best_cluster = int(max_streaks.first()['cluster'])
    plot_streak_per_cluster(with_streak, best_cluster, config.OUTPUT_DIR)

    # Streak count by cluster for map
    streak_count_pdf = (with_streak
                        .filter(with_streak.streak > 0)
                        .groupBy("cluster", "streak")
                        .agg({"streak": "count"})
                        .toPandas())

    # ── Step 5: Maps ──────────────────────────────────────────────────────────
    if not args.no_maps:
        print("\n" + "═" * 60)
        print("Step 5 — Generating Interactive Maps")
        print("═" * 60)

        from pipeline.visualisation import (
            map_country_aqi, map_cluster_improvement,
            map_cluster_zones, map_streak_popups
        )

        worst_pdf = worst_today.toPandas()
        top50_pdf = pdf_clusters.head(config.TOP_N)

        # Sensor sample for cluster zone map
        sensor_sample = (df_clustered
                         .select('latitude', 'longitude', 'cluster')
                         .dropDuplicates(['latitude', 'longitude'])
                         .toPandas())

        map_country_aqi(worst_pdf, config.OUTPUT_DIR)
        map_cluster_improvement(top50_pdf, config.OUTPUT_DIR)
        map_cluster_zones(sensor_sample, model.clusterCenters(), config.OUTPUT_DIR)
        map_streak_popups(streak_count_pdf, model.clusterCenters(), config.OUTPUT_DIR)

    # ── Summary ───────────────────────────────────────────────────────────────
    elapsed = time.time() - start_time

    print("\n╔══════════════════════════════════════════════════════════╗")
    print("║  ✅  Pipeline Complete                                    ║")
    print("╠══════════════════════════════════════════════════════════╣")
    print(f"║  Total time       : {elapsed:.1f}s{'':>37}║")
    print(f"║  Analysis date    : {date_today}{'':>38}║")
    print(f"║  Output directory : {config.OUTPUT_DIR}{'':>40}║")
    print("╠══════════════════════════════════════════════════════════╣")
    print("║  Charts generated :                                      ║")
    print("║    top10_countries.png                                   ║")
    print("║    top50_clusters.png                                    ║")
    print("║    streak_distribution.png                               ║")
    if not args.no_maps:
        print("║  Maps generated :                                        ║")
        print("║    map_country_aqi.html                                  ║")
        print("║    map_cluster_improvement.html                          ║")
        print("║    map_cluster_zones.html                                ║")
        print("║    map_streak_popups.html                                ║")
    print("╚══════════════════════════════════════════════════════════╝")

    spark.stop()


if __name__ == '__main__':
    args = parse_args()
    run(args)
