from src import etl, features, models, report


def main():
    print("Step 1: fetching ACN sessions...")
    etl.fetch_acn()

    print("Step 2: ingesting raw sessions into Postgres...")
    etl.ingest_acn()

    print("Step 3: cleaning sessions...")
    etl.clean_sessions()

    print("Step 4: building station-hourly features...")
    features.main()

    print("Step 5: training XGBoost models...")
    models.train_xgb()

    print("Step 6: training KMeans behavior clusters...")
    models.train_kmeans()

    print("Step 7: running batch predictions...")
    models.batch_predict()

    print("Step 8: generating report...")
    report.main()

    print("Pipeline complete.")


if __name__ == "__main__":
    main()
