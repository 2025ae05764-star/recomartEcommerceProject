# Databricks notebook source
# DBTITLE 1,Overview
# Task 2 – Step 2: Data Ingestion in Databricks (BATCH ONLY)
# ================================================
# RecoMart Data Management Pipeline
# 
# **UPDATED ARCHITECTURE:**
# This notebook now handles BATCH ingestion only for:
#   - Purchase History (CSV)
#   - Product Catalog (JSON)
#   - External Signals (JSON)
#
# User Interaction Logs are handled by STREAMING ingestion in:
#   - task2_streaming_ingestion notebook
#
# CLUSTER REQUIREMENTS:
#   Runtime : Databricks Runtime 13.x or above (includes Spark 3.4+, Delta Lake)
#   Node type: Standard (e.g. i3.xlarge or ds3_v2 for cost efficiency)
#
# HOW TO USE:
#   1. This runs on schedule every 2 hours for batch data
#   2. Streaming ingestion runs continuously for user interactions
#   3. All data flows into Bronze layer Delta tables

# COMMAND ----------

# %md
# ## RecoMart – Data Ingestion (Bronze Layer)
# Reads raw files from S3 and loads them as Delta tables for downstream processing.

# COMMAND ----------

# DBTITLE 1,Configuration & Imports
# ── Cell 1: Configuration & Imports ──────────────────────────────────────────
import logging
from datetime import datetime
import uuid
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import (
    StructType, StructField, StringType, DoubleType,
    IntegerType, BooleanType, TimestampType, ArrayType
)

# ── CONFIG — update these values ──────────────────────────────────────────────
VOLUME_BASE_PATH = "/Volumes/workspace/recomart_source_data/raw/data"  # Volume path for source data
DELTA_DB         = "recomart_bronze"              # Databricks database name
TODAY            = datetime.now()

# Delta table output paths
DELTA_TABLES = {
    "clickstream":  f"{DELTA_DB}.user_interaction_logs",
    "transactions": f"{DELTA_DB}.purchase_history",
    "catalog":      f"{DELTA_DB}.product_catalog",
    "external":     f"{DELTA_DB}.external_signals",
}

print(f"Config loaded. Ingestion date: {TODAY.strftime('%Y-%m-%d %H:%M:%S')}")
print(f"Source path: {VOLUME_BASE_PATH}")

# COMMAND ----------

# ── Cell 2: Spark Session & Database Setup ─────────────────────────────────────
# In Databricks, SparkSession is pre-created as `spark`.
# The line below makes it explicit and adds S3 access config.

#spark.conf.set("spark.sql.adaptive.enabled", "true")
#spark.conf.set("spark.databricks.delta.optimizeWrite.enabled", "true")

# AWS credential config (skip if using IAM instance profile — recommended)
#spark.conf.set("fs.s3a.access.key", dbutils.secrets.get("recomart", "key"))
#spark.conf.set("fs.s3a.secret.key", dbutils.secrets.get("recomart", "secretkey"))

# Create Bronze database if it doesn't exist
spark.sql(f"CREATE DATABASE IF NOT EXISTS {DELTA_DB}")
print(f"Database ready: {DELTA_DB}")

# COMMAND ----------

# ── Cell 3: Schema Definitions (BATCH ONLY) ────────────────────────────────────────────────
# Explicit schemas enforce data contracts at ingestion time.
# If a file doesn't match, Spark raises an error immediately.
#
# NOTE: User interaction logs use streaming ingestion with Auto Loader schema inference
#       (see task2_streaming_ingestion)

schema_transactions = StructType([
    StructField("order_id",        StringType(),  True),
    StructField("user_id",         StringType(),  False),
    StructField("item_id",         StringType(),  False),
    StructField("quantity",        IntegerType(), True),
    StructField("unit_price",      DoubleType(),  True),
    StructField("total_price",     DoubleType(),  True),
    StructField("purchase_date",   StringType(),  True),
    StructField("payment_method",  StringType(),  True),
    StructField("gender",          StringType(),  True),
    StructField("age_group",       StringType(),  True),
])

# For JSON sources we use multiLine + inferred schema, but validate key columns
print("Schemas defined for batch ingestion (purchase_history, product_catalog, external_signals).")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Create Logging Table

# COMMAND ----------

# DBTITLE 1,Logging Infrastructure
# ══════════════════════════════════════════════════════════════════════════════
# Logging Infrastructure
# ══════════════════════════════════════════════════════════════════════════════

import uuid
from datetime import datetime
from pyspark.sql.types import StructType, StructField, StringType, IntegerType, TimestampType, LongType

# Create logging table if it doesn't exist
spark.sql("""
    CREATE TABLE IF NOT EXISTS recomart_bronze.ingestion_logs (
        log_id STRING,
        run_id STRING,
        job_run_id BIGINT,
        notebook_name STRING,
        job_type STRING,
        status STRING,
        start_time TIMESTAMP,
        end_time TIMESTAMP,
        duration_seconds INT,
        records_processed INT,
        tables_updated STRING,
        error_message STRING,
        error_type STRING
    )
    USING DELTA
    COMMENT 'Execution logs for batch and streaming ingestion pipelines'
""")

class IngestionLogger:
    """Logger for tracking ingestion pipeline execution"""
    
    def __init__(self, notebook_name: str, job_type: str, job_run_id: int = None):
        self.notebook_name = notebook_name
        self.job_type = job_type
        self.run_id = str(uuid.uuid4())[:8]
        self.start_time = datetime.now()
        self.records_processed = 0
        self.tables_updated = []
        
        # Accept job_run_id as parameter (allows parent task to pass it)
        # Fall back to widget parameter if not provided
        if job_run_id is not None:
            self.job_run_id = job_run_id
        else:
            # Try to get from notebook widget (can be passed from job)
            try:
                self.job_run_id = int(dbutils.widgets.get("job_run_id"))
            except:
                self.job_run_id = None  # Running interactively without job context
        
        print(f"\n{'='*80}")
        print(f"INGESTION RUN STARTED")
        print(f"{'='*80}")
        print(f"  Run ID: {self.run_id}")
        if self.job_run_id:
            print(f"  Job Run ID: {self.job_run_id}")
        else:
            print(f"  Job Run ID: NULL (interactive mode)")
        print(f"  Notebook: {self.notebook_name}")
        print(f"  Job Type: {self.job_type}")
        print(f"  Start Time: {self.start_time.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"{'='*80}\n")
        
        # Log start
        self._write_log('RUNNING', None, None, None, None)
    
    def add_records(self, count: int):
        """Increment the record count"""
        self.records_processed += count
    
    def add_table(self, table_name: str):
        """Add a table to the updated list"""
        self.tables_updated.append(table_name)
    
    def log_success(self):
        """Log successful completion"""
        end_time = datetime.now()
        duration = int((end_time - self.start_time).total_seconds())
        
        print(f"\n{'='*80}")
        print(f"✅ INGESTION COMPLETED SUCCESSFULLY")
        print(f"{'='*80}")
        print(f"  Run ID: {self.run_id}")
        if self.job_run_id:
            print(f"  Job Run ID: {self.job_run_id}")
        print(f"  Duration: {duration} seconds")
        print(f"  Records Processed: {self.records_processed:,}")
        print(f"  Tables Updated: {', '.join(self.tables_updated)}")
        print(f"  End Time: {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"{'='*80}\n")
        
        self._write_log('SUCCESS', end_time, duration, None, None)
    
    def log_failure(self, error: Exception):
        """Log failed execution"""
        end_time = datetime.now()
        duration = int((end_time - self.start_time).total_seconds())
        error_type = type(error).__name__
        error_message = str(error)
        
        print(f"\n{'='*80}")
        print(f"❌ INGESTION FAILED")
        print(f"{'='*80}")
        print(f"  Run ID: {self.run_id}")
        if self.job_run_id:
            print(f"  Job Run ID: {self.job_run_id}")
        print(f"  Duration: {duration} seconds")
        print(f"  Records Processed: {self.records_processed:,}")
        print(f"  Error Type: {error_type}")
        print(f"  Error Message: {error_message}")
        print(f"  End Time: {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"{'='*80}\n")
        
        self._write_log('FAILED', end_time, duration, error_message, error_type)
    
    def _write_log(self, status: str, end_time, duration, error_message, error_type):
        """Write log entry to Delta table"""
        log_schema = StructType([
            StructField("log_id", StringType(), False),
            StructField("run_id", StringType(), False),
            StructField("job_run_id", LongType(), True),
            StructField("notebook_name", StringType(), False),
            StructField("job_type", StringType(), False),
            StructField("status", StringType(), False),
            StructField("start_time", TimestampType(), False),
            StructField("end_time", TimestampType(), True),
            StructField("duration_seconds", IntegerType(), True),
            StructField("records_processed", IntegerType(), True),
            StructField("tables_updated", StringType(), True),
            StructField("error_message", StringType(), True),
            StructField("error_type", StringType(), True)
        ])
        
        log_data = [(
            str(uuid.uuid4()),
            self.run_id,
            self.job_run_id,
            self.notebook_name,
            self.job_type,
            status,
            self.start_time,
            end_time,
            duration,
            self.records_processed if self.records_processed > 0 else None,
            ','.join(self.tables_updated) if self.tables_updated else None,
            error_message,
            error_type
        )]
        
        log_df = spark.createDataFrame(log_data, log_schema)
        log_df.write.format("delta").mode("append").saveAsTable("recomart_bronze.ingestion_logs")

print("✓ Logging infrastructure initialized")
print("  Log table: recomart_bronze.ingestion_logs")
print("  Supports: job_run_id parameter or widget")

# COMMAND ----------

# DBTITLE 1,Ingestion Helper Functions
# ── Cell 4: Ingestion Helper Functions ────────────────────────────────────────

def get_latest_file(base_path: str, file_prefix: str, file_extension: str) -> str:
    """
    Find the most recent timestamped file matching the pattern.
    
    Args:
        base_path: Directory containing the files
        file_prefix: File name prefix (e.g., 'user_interaction_logs')
        file_extension: File extension (e.g., 'csv' or 'json')
    
    Returns:
        Full path to the latest file
    """
    try:
        files = dbutils.fs.ls(base_path)
        
        # Filter files by prefix and extension
        matching_files = [
            (f.path, f.modificationTime) 
            for f in files 
            if f.name.startswith(file_prefix) and f.name.endswith(file_extension)
        ]
        
        if not matching_files:
            raise FileNotFoundError(f"No files found matching {file_prefix}*.{file_extension}")
        
        # Sort by modification time (newest first) and return the latest
        matching_files.sort(key=lambda x: x[1], reverse=True)
        latest_file = matching_files[0][0]
        
        print(f"  Latest file: {latest_file.split('/')[-1]}")
        return latest_file
        
    except Exception as e:
        print(f"  [ERROR] Failed to find latest file for {file_prefix}: {e}")
        raise

def ingest_csv(source_key: str, base_path: str, file_prefix: str, schema, table_name: str):
    """
    Read the latest CSV file from volume path, add ingestion metadata, and save as Delta table.
    
    Args:
        source_key : short name for logging (e.g. 'clickstream')
        base_path  : base directory path
        file_prefix: file name prefix to match
        schema     : PySpark StructType for the CSV
        table_name : target Delta table (database.table)
    
    Returns:
        PySpark DataFrame
    """
    print(f"\n[{source_key.upper()}] Finding latest CSV file...")
    run_id = str(uuid.uuid4())
    pipeline_name = source_key+"_ingestion"
    start_time = datetime.now()
    
    try:
        # Get the latest file
        file_path = get_latest_file(base_path, file_prefix, "csv")
        
        df = (
            spark.read
                 .option("header", "true")
                 .option("inferSchema", "false")     # use explicit schema
                 .option("mode", "PERMISSIVE")        # bad rows → _corrupt_record col
                 .option("columnNameOfCorruptRecord", "_corrupt_record")
                 .schema(schema)
                 .csv(file_path)
        )

        # ── Add ingestion metadata ─────────────────────────────────────────────
        df = (
            df.withColumn("_ingested_at",  F.lit(TODAY.isoformat()))
              .withColumn("_source_path",  F.lit(file_path))
              .withColumn("_source_type",  F.lit("csv"))
              .withColumn("_batch_id",     F.lit(TODAY.strftime("%Y%m%d%H%M%S")))
        )

        row_count     = df.count()
        corrupt_count = (df.filter(F.col("_corrupt_record").isNotNull()).count()
                         if "_corrupt_record" in df.columns else 0)

        print(f"  Rows read       : {row_count:,}")
        print(f"  Corrupt rows    : {corrupt_count:,}")
        print(f"  Columns         : {len(df.columns)}")

        # ── Write to Delta ─────────────────────────────────────────────────────
        print(f"  Writing to Delta: {table_name}")
        (
            df.write
              .format("delta")
              .mode("append")  # Changed to append to accumulate historical data
              .option("mergeSchema", "true")
              .saveAsTable(table_name)
        )
        print(f"  [OK] {table_name} written successfully.")
        status = "SUCCESS"
        record_count = df.count()
        error_message = None
        return df

    except Exception as e:
        print(f"  [ERROR] Failed to ingest {source_key}: {e}")
        status = "FAILED"
        record_count = 0
        error_message = str(e)
        raise
    finally:
        end_time = datetime.now()

def ingest_json(source_key: str, base_path: str, file_prefix: str, data_field: str, table_name: str):
    """
    Read the latest multi-record JSON file from volume path.
    The actual records are expected under response['data'] array.
    
    Args:
        source_key : short name
        base_path  : base directory path
        file_prefix: file name prefix to match
        data_field : top-level key that holds the array (e.g. 'data')
        table_name : target Delta table
    
    Returns:
        PySpark DataFrame (exploded records)
    """
    print(f"\n[{source_key.upper()}] Finding latest JSON file...")
    
    try:
        # Get the latest file
        file_path = get_latest_file(base_path, file_prefix, "json")
        
        # Read full JSON, then explode the nested array
        raw = (
            spark.read
                 .option("multiLine", "true")
                 .option("mode", "PERMISSIVE")
                 .json(file_path)
        )

        # Explode the records array into individual rows
        df = raw.select(F.explode(F.col(data_field)).alias("record"))
        df = df.select("record.*")

        # Add ingestion metadata
        df = (
            df.withColumn("_ingested_at", F.lit(TODAY.isoformat()))
              .withColumn("_source_path", F.lit(file_path))
              .withColumn("_source_type", F.lit("json_api"))
              .withColumn("_batch_id",    F.lit(TODAY.strftime("%Y%m%d%H%M%S")))
        )

        row_count = df.count()
        print(f"  Records read    : {row_count:,}")
        print(f"  Columns         : {len(df.columns)}")

        # Write to Delta
        print(f"  Writing to Delta: {table_name}")
        (
            df.write
              .format("delta")
              .mode("append")  # Changed to append to accumulate historical data
              .option("mergeSchema", "true")
              .saveAsTable(table_name)
        )
        print(f"  [OK] {table_name} written successfully.")
        return df

    except Exception as e:
        print(f"  [ERROR] Failed to ingest {source_key}: {e}")
        raise

# COMMAND ----------

# DBTITLE 1,Run Ingestion
# ── Cell 5: Run Batch Ingestion (Purchase, Catalog, Signals ONLY) ────────────────────────

from datetime import datetime
TODAY = datetime.now()

# Try to get job_run_id from notebook parameter (if running as part of job)
try:
    job_run_id = int(dbutils.widgets.get("job_run_id"))
except:
    job_run_id = None

# Initialize logger
logger = IngestionLogger(notebook_name="task2_databricks_ingestion", job_type="BATCH", job_run_id=job_run_id)

try:
    print("=" * 60)
    print("RecoMart BATCH Data Ingestion — Bronze Layer")
    print(f"Run time: {TODAY.strftime('%Y-%m-%d %H:%M:%S')}")
    print("\nNOTE: User interaction logs are handled by streaming ingestion")
    print("=" * 60)

    # Source 1: Purchase History (CSV) - find latest timestamped file
    df_transactions = ingest_csv(
        "transactions",
        VOLUME_BASE_PATH,
        "purchase_history",
        schema_transactions,
        DELTA_TABLES["transactions"]
    )
    records_trans = df_transactions.count()
    logger.add_records(records_trans)
    logger.add_table("purchase_history")
    print(f"  ✓ Ingested {records_trans:,} purchase records")

    # Source 2: Product Catalog (JSON API response) - find latest timestamped file
    df_catalog = ingest_json(
        "catalog",
        VOLUME_BASE_PATH,
        "product_catalog",
        data_field="data",
        table_name=DELTA_TABLES["catalog"]
    )
    records_catalog = df_catalog.count()
    logger.add_records(records_catalog)
    logger.add_table("product_catalog")
    print(f"  ✓ Ingested {records_catalog:,} product records")

    # Source 3: External Signals (JSON API response) - find latest timestamped file
    df_external = ingest_json(
        "external",
        VOLUME_BASE_PATH,
        "external_signals",
        data_field="data",
        table_name=DELTA_TABLES["external"]
    )
    records_external = df_external.count()
    logger.add_records(records_external)
    logger.add_table("external_signals")
    print(f"  ✓ Ingested {records_external:,} external signal records")

    print("\n" + "=" * 60)
    print("✓ Batch ingestion complete (3 tables)")
    print("=" * 60)
    
    # Log success
    logger.log_success()
    
except Exception as e:
    # Log failure with error details
    logger.log_failure(e)
    print(f"\n❌ ERROR: {str(e)}")
    print("\nFull stack trace:")
    import traceback
    traceback.print_exc()
    raise  # Re-raise to stop notebook execution

# COMMAND ----------

# DBTITLE 1,Quick Validation Checks
# ── Cell 6: Quick Validation Checks (BATCH TABLES ONLY) ───────────────────────────────────
print("\n" + "=" * 60)
print("BATCH INGESTION VALIDATION CHECKS")
print("=" * 60)

checks = [
    ("purchase_history",      df_transactions, ["order_id", "user_id", "item_id", "unit_price"]),
    ("product_catalog",       df_catalog,      ["item_id", "name", "category", "price"]),
    ("external_signals",      df_external,     ["item_id", "signal_id", "trend_score", "social_mentions"]),
]

for name, df, required_cols in checks:
    print(f"\n[{name}]")
    missing = [c for c in required_cols if c not in df.columns]
    null_counts = {c: df.filter(F.col(c).isNull()).count() for c in required_cols if c in df.columns}
    print(f"  Row count         : {df.count():,}")
    print(f"  Missing key cols  : {missing if missing else 'None'}")
    for col, cnt in null_counts.items():
        status = "WARN" if cnt > 0 else "OK"
        print(f"  Nulls in {col:25s}: {cnt:>5} [{status}]")

# COMMAND ----------

# DBTITLE 1,Preview Ingested Data
# ── Cell 7: Preview Ingested Data (BATCH ONLY) ─────────────────────────────────────────
print("\n── Purchase History (first 5 rows) ──")
df_transactions.select(
    "order_id","user_id","item_id","quantity","unit_price","payment_method"
).show(5, truncate=False)

print("── Product Catalog (first 5 rows) ──")
df_catalog.select(
    "item_id","name","category","brand","price","stock"
).show(5, truncate=False)

print("── External Signals (first 5 rows) ──")
df_external.select(
    "item_id","signal_id","trend_score","social_mentions","timestamp"
).show(5, truncate=False)

# COMMAND ----------

# DBTITLE 1,Ingestion Summary
# ── Cell 8: Ingestion Summary (BATCH ONLY) ────────────────────────────────────────────
print("\n" + "=" * 60)
print("BATCH INGESTION COMPLETE — Summary")
print("=" * 60)
summary = [
    ("purchase_history",  spark.table(DELTA_TABLES["transactions"]).count()),
    ("product_catalog",   spark.table(DELTA_TABLES["catalog"]).count()),
    ("external_signals",  spark.table(DELTA_TABLES["external"]).count()),
]

for table, count in summary:
    print(f"  {DELTA_DB}.{table:25s}: {count:>8,} rows")

print("\n" + "=" * 60)
print("✓ All BATCH tables updated successfully")
print("\nNOTE: User interaction logs use STREAMING ingestion")
print("      Check task2_streaming_ingestion for real-time events")
print("=" * 60)

# COMMAND ----------

# ── Cell 9: Verify Delta Tables ───────────────────────────────────────────────
# Run this cell after ingestion to confirm tables are queryable
display(spark.sql(f"SHOW TABLES IN {DELTA_DB}"))

# COMMAND ----------

# DBTITLE 1,View Ingestion Logs
# ══════════════════════════════════════════════════════════════════════════════
# View Ingestion Execution Logs
# ══════════════════════════════════════════════════════════════════════════════

print("\n" + "="*80)
print("INGESTION EXECUTION LOGS")
print("="*80)

# Query logs from the logging table
logs_df = spark.table("recomart_bronze.ingestion_logs").orderBy("start_time", ascending=False).limit(20)

print(f"\nShowing last 20 ingestion runs:\n")
display(logs_df.select(
    "run_id",
    "job_run_id",
    "notebook_name",
    "job_type",
    "status",
    "start_time",
    "duration_seconds",
    "records_processed",
    "tables_updated",
    "error_type"
))

print("\nLegend:")
print("  run_id: Internal short UUID for this execution")
print("  job_run_id: Databricks job run ID (NULL if interactive)")
print("\nTo find logs for a specific job run:")
print("  SELECT * FROM recomart_bronze.ingestion_logs WHERE job_run_id = <your_job_run_id>")
print("="*80)
