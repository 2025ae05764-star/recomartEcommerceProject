"""
Task 7 – Feature Store Implementation
=======================================
RecoMart Data Management Pipeline

Architecture:
  ┌─────────────────────────────────────────────────┐
  │                  Feature Store                   │
  │                                                  │
  │  Registry → Offline Store → Online Store → API  │
  └─────────────────────────────────────────────────┘

Components built:
  1. FeatureRegistry    – central catalogue of all features (metadata)
  2. OfflineStore       – SQLite warehouse (used for training)
  3. OnlineStore        – dict-based in-memory cache (low-latency inference)
  4. FeatureServer      – retrieval API (get_user_features, get_item_features,
                          get_training_dataset, materialize_online)
  5. FeaturePipeline    – end-to-end orchestration
"""

import os, json, sqlite3, time, hashlib
from datetime import datetime
from typing  import List, Dict, Optional, Any
import pandas as pd
import numpy  as np

FEAT_DIR  = "/home/claude/features"
STORE_DIR = "/home/claude/feature_store"
os.makedirs(STORE_DIR, exist_ok=True)

# ══════════════════════════════════════════════════════════════════════════════
# 1. FEATURE REGISTRY
#    Central catalogue: what every feature is, where it lives, its version.
# ══════════════════════════════════════════════════════════════════════════════

class FeatureRegistry:
    """
    Stores metadata for every feature:
      - name, group (user/item/user_item), source table, primary key
      - dtype, feature_type (continuous / categorical / binary)
      - version, description, tags
      - lineage (which raw table it was derived from)
    """

    def __init__(self, registry_path: str):
        self.path = registry_path
        self.conn = sqlite3.connect(registry_path)
        self._init_schema()

    def _init_schema(self):
        self.conn.executescript("""
            CREATE TABLE IF NOT EXISTS feature_definitions (
                feature_id      TEXT PRIMARY KEY,
                feature_name    TEXT NOT NULL,
                feature_group   TEXT NOT NULL,
                table_name      TEXT NOT NULL,
                primary_key     TEXT NOT NULL,
                dtype           TEXT,
                feature_type    TEXT,
                description     TEXT,
                source_table    TEXT,
                version         TEXT DEFAULT 'v1.0',
                is_active       INTEGER DEFAULT 1,
                created_at      TEXT,
                updated_at      TEXT,
                tags            TEXT
            );

            CREATE TABLE IF NOT EXISTS feature_versions (
                version_id      TEXT PRIMARY KEY,
                feature_name    TEXT NOT NULL,
                version         TEXT NOT NULL,
                checksum        TEXT,
                row_count       INTEGER,
                registered_at   TEXT,
                notes           TEXT
            );

            CREATE TABLE IF NOT EXISTS feature_sets (
                set_name        TEXT PRIMARY KEY,
                description     TEXT,
                feature_names   TEXT,
                created_at      TEXT
            );
        """)
        self.conn.commit()

    def register_feature(self, name: str, group: str, table: str,
                          pk: str, dtype: str, ftype: str,
                          description: str, source: str,
                          version: str = "v1.0", tags: List[str] = None):
        fid = hashlib.md5(f"{name}_{version}".encode()).hexdigest()[:12]
        now = datetime.now().isoformat()
        self.conn.execute("""
            INSERT OR REPLACE INTO feature_definitions
            (feature_id, feature_name, feature_group, table_name, primary_key,
             dtype, feature_type, description, source_table, version, created_at,
             updated_at, tags)
            VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)
        """, (fid, name, group, table, pk, dtype, ftype, description,
              source, version, now, now, json.dumps(tags or [])))
        self.conn.commit()

    def register_feature_set(self, set_name: str, description: str,
                              feature_names: List[str]):
        self.conn.execute("""
            INSERT OR REPLACE INTO feature_sets (set_name, description,
                feature_names, created_at)
            VALUES (?,?,?,?)
        """, (set_name, description, json.dumps(feature_names),
               datetime.now().isoformat()))
        self.conn.commit()

    def get_feature(self, name: str) -> Optional[Dict]:
        cur = self.conn.execute(
            "SELECT * FROM feature_definitions WHERE feature_name=? AND is_active=1",
            (name,))
        row = cur.fetchone()
        if row:
            cols = [d[0] for d in cur.description]
            return dict(zip(cols, row))
        return None

    def list_features(self, group: str = None) -> pd.DataFrame:
        q = "SELECT * FROM feature_definitions WHERE is_active=1"
        params = ()
        if group:
            q += " AND feature_group=?"
            params = (group,)
        return pd.read_sql(q, self.conn, params=params)

    def record_version(self, feature_name: str, version: str,
                        row_count: int, checksum: str, notes: str = ""):
        vid = hashlib.md5(f"{feature_name}_{version}_{datetime.now().isoformat()}".encode()).hexdigest()[:12]
        self.conn.execute("""
            INSERT OR REPLACE INTO feature_versions
            (version_id, feature_name, version, checksum, row_count, registered_at, notes)
            VALUES (?,?,?,?,?,?,?)
        """, (vid, feature_name, version, checksum, row_count,
               datetime.now().isoformat(), notes))
        self.conn.commit()

    def close(self):
        self.conn.close()


# ══════════════════════════════════════════════════════════════════════════════
# 2. OFFLINE STORE
#    Historical feature warehouse used to build training datasets.
#    Backed by SQLite (production: Hive / Delta Lake / BigQuery).
# ══════════════════════════════════════════════════════════════════════════════

class OfflineStore:
    """
    Persists feature tables for training.
    Supports point-in-time correct joins (time-travel queries).
    """

    def __init__(self, db_path: str):
        self.db_path = db_path
        self.conn    = sqlite3.connect(db_path)

    def write_features(self, df: pd.DataFrame, table: str,
                        pk: str, version: str = "v1.0"):
        """Write a feature DataFrame; add metadata columns."""
        df = df.copy()
        df["_version"]    = version
        df["_written_at"] = datetime.now().isoformat()
        df.to_sql(table, self.conn, if_exists="replace", index=False)

        # Create index on pk
        try:
            self.conn.execute(f"CREATE INDEX IF NOT EXISTS idx_{table}_{pk} "
                              f"ON {table}({pk})")
        except Exception:
            pass
        self.conn.commit()

        checksum = hashlib.md5(pd.util.hash_pandas_object(df).values.tobytes()).hexdigest()
        return len(df), checksum

    def read_features(self, table: str, keys: List[str] = None,
                       pk: str = None, columns: List[str] = None) -> pd.DataFrame:
        """Read features by primary key values. If keys=None, return all."""
        cols = ", ".join(columns) if columns else "*"
        if keys and pk:
            placeholders = ",".join(["?"] * len(keys))
            q = f"SELECT {cols} FROM {table} WHERE {pk} IN ({placeholders})"
            return pd.read_sql(q, self.conn, params=keys)
        return pd.read_sql(f"SELECT {cols} FROM {table}", self.conn)

    def get_training_dataset(self, user_features: List[str],
                              item_features: List[str],
                              label_col: str = "ui_has_purchased",
                              limit: int = None) -> pd.DataFrame:
        """
        Build a training dataset by joining:
          user_item_features (labels)
          LEFT JOIN user_features
          LEFT JOIN item_features
        This is the canonical 'point-in-time' offline join.
        """
        u_cols = ", ".join([f"u.{c}" for c in user_features])
        i_cols = ", ".join([f"i.{c}" for c in item_features])
        lim    = f"LIMIT {limit}" if limit else ""

        q = f"""
        SELECT
            ui.user_id,
            ui.item_id,
            ui.ui_interaction_score,
            ui.ui_interaction_score_norm,
            ui.{label_col},
            {u_cols},
            {i_cols}
        FROM user_item_features ui
        LEFT JOIN user_features u ON ui.user_id = u.user_id
        LEFT JOIN item_features i ON ui.item_id = i.item_id
        {lim}
        """
        return pd.read_sql(q, self.conn)

    def list_tables(self) -> List[str]:
        cur = self.conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table'")
        return [r[0] for r in cur.fetchall()]

    def close(self):
        self.conn.close()


# ══════════════════════════════════════════════════════════════════════════════
# 3. ONLINE STORE
#    Low-latency key-value cache for real-time feature serving.
#    Production: Redis / DynamoDB / Cassandra.
#    Here: in-memory Python dict with TTL simulation.
# ══════════════════════════════════════════════════════════════════════════════

class OnlineStore:
    """
    Key-value store for real-time feature retrieval.
    Key   : f"{namespace}:{entity_id}"   e.g. "user:USER_0001"
    Value : dict of {feature_name: value}
    TTL   : simulated per-key expiry (seconds)
    """

    def __init__(self, default_ttl: int = 3600):
        self._store: Dict[str, Dict] = {}
        self._expiry: Dict[str, float] = {}
        self.default_ttl = default_ttl
        self.hits   = 0
        self.misses = 0

    def set(self, namespace: str, entity_id: str,
             features: Dict[str, Any], ttl: int = None):
        key = f"{namespace}:{entity_id}"
        self._store[key]  = features
        self._expiry[key] = time.time() + (ttl or self.default_ttl)

    def get(self, namespace: str, entity_id: str) -> Optional[Dict]:
        key = f"{namespace}:{entity_id}"
        if key in self._store:
            if time.time() < self._expiry[key]:
                self.hits += 1
                return self._store[key]
            else:
                del self._store[key]
                del self._expiry[key]
        self.misses += 1
        return None

    def delete(self, namespace: str, entity_id: str):
        key = f"{namespace}:{entity_id}"
        self._store.pop(key, None)
        self._expiry.pop(key, None)

    def bulk_set(self, namespace: str, records: pd.DataFrame,
                  id_col: str, ttl: int = None):
        """Materialize an entire DataFrame into the online store."""
        feature_cols = [c for c in records.columns
                        if c != id_col and not c.startswith("_")]
        for _, row in records.iterrows():
            fdict = {c: (None if pd.isna(row[c]) else row[c])
                     for c in feature_cols}
            self.set(namespace, str(row[id_col]), fdict, ttl)

    def stats(self) -> Dict:
        total = self.hits + self.misses
        return {
            "total_keys": len(self._store),
            "cache_hits":  self.hits,
            "cache_misses":self.misses,
            "hit_rate":    round(self.hits / total, 3) if total else 0.0,
        }


# ══════════════════════════════════════════════════════════════════════════════
# 4. FEATURE SERVER
#    Unified retrieval API — used by both training and inference.
# ══════════════════════════════════════════════════════════════════════════════

class FeatureServer:
    """
    Public API for the feature store.

    Training workflow:
        ds = server.get_training_dataset(user_feats, item_feats)

    Inference workflow:
        user_feats = server.get_user_features("USER_0001", keys)
        item_feats = server.get_item_features("ITEM_0023", keys)
        combined   = server.get_serving_vector("USER_0001", "ITEM_0023")
    """

    # Default feature lists used for training & serving
    DEFAULT_USER_FEATURES = [
        "user_total_events", "user_purchase_count", "user_purchase_rate",
        "user_avg_spend",    "user_total_spend",    "user_distinct_items",
        "user_recency_days", "user_tenure_days",    "user_activity_score_norm",
        "user_preferred_device_enc", "user_preferred_category_enc",
    ]
    DEFAULT_ITEM_FEATURES = [
        "item_total_interactions", "item_purchase_count",
        "item_conversion_rate",    "item_avg_rating",
        "item_popularity_score",   "item_avg_sentiment",
        "item_price_log",          "item_stock_norm",
        "item_category_enc",       "item_brand_enc",
        "item_trend_score_norm",   "item_return_rate",
    ]

    def __init__(self, offline: OfflineStore, online: OnlineStore,
                  registry: FeatureRegistry):
        self.offline  = offline
        self.online   = online
        self.registry = registry

    # ── Online retrieval ─────────────────────────────────────────────────────

    def get_user_features(self, user_id: str,
                           feature_keys: List[str] = None) -> Optional[Dict]:
        """Fetch user features from online store. Falls back to offline."""
        fdict = self.online.get("user", user_id)
        if fdict is None:
            # Fallback: read from offline store
            df = self.offline.read_features(
                "user_features", keys=[user_id], pk="user_id")
            if df.empty:
                return None
            fdict = df.iloc[0].to_dict()
            self.online.set("user", user_id, fdict)

        if feature_keys:
            return {k: fdict[k] for k in feature_keys if k in fdict}
        return fdict

    def get_item_features(self, item_id: str,
                           feature_keys: List[str] = None) -> Optional[Dict]:
        """Fetch item features from online store. Falls back to offline."""
        fdict = self.online.get("item", item_id)
        if fdict is None:
            df = self.offline.read_features(
                "item_features", keys=[item_id], pk="item_id")
            if df.empty:
                return None
            fdict = df.iloc[0].to_dict()
            self.online.set("item", item_id, fdict)

        if feature_keys:
            return {k: fdict[k] for k in feature_keys if k in fdict}
        return fdict

    def get_serving_vector(self, user_id: str, item_id: str) -> Optional[Dict]:
        """
        Return a single flat feature vector for a (user, item) pair.
        Used at inference time to score a candidate item.
        """
        uf = self.get_user_features(user_id, self.DEFAULT_USER_FEATURES)
        itf = self.get_item_features(item_id, self.DEFAULT_ITEM_FEATURES)
        if uf is None or itf is None:
            return None
        return {**{f"u_{k}": v for k, v in uf.items()},
                **{f"i_{k}": v for k, v in itf.items()},
                "user_id": user_id, "item_id": item_id}

    # ── Offline retrieval ────────────────────────────────────────────────────

    def get_training_dataset(self,
                              user_features: List[str] = None,
                              item_features: List[str] = None,
                              limit: int = None) -> pd.DataFrame:
        """Build the joined training dataset from offline store."""
        return self.offline.get_training_dataset(
            user_features or self.DEFAULT_USER_FEATURES,
            item_features or self.DEFAULT_ITEM_FEATURES,
            limit=limit)

    # ── Materialisation ──────────────────────────────────────────────────────

    def materialize_online(self, batch_size: int = 100):
        """
        Populate the online store from offline tables.
        In production this runs on a schedule (e.g. hourly Airflow DAG).
        """
        print("  Materialising user features →  online store ...")
        uf = self.offline.read_features("user_features")
        self.online.bulk_set("user", uf, id_col="user_id")
        print(f"    {len(uf):,} users loaded")

        print("  Materialising item features →  online store ...")
        itf = self.offline.read_features("item_features")
        self.online.bulk_set("item", itf, id_col="item_id")
        print(f"    {len(itf):,} items loaded")

        return {"users_loaded": len(uf), "items_loaded": len(itf)}


# ══════════════════════════════════════════════════════════════════════════════
# 5. FEATURE PIPELINE
#    Orchestrates: load → register → write offline → materialise online
# ══════════════════════════════════════════════════════════════════════════════

class FeaturePipeline:

    def __init__(self, feature_dir: str, store_dir: str):
        self.feature_dir = feature_dir
        self.store_dir   = store_dir

        reg_path  = f"{store_dir}/feature_registry.db"
        off_path  = f"{store_dir}/offline_store.db"

        self.registry = FeatureRegistry(reg_path)
        self.offline  = OfflineStore(off_path)
        self.online   = OnlineStore(default_ttl=3600)
        self.server   = FeatureServer(self.offline, self.online, self.registry)

    def run(self):
        print("=" * 60)
        print("RecoMart Feature Pipeline — Running")
        print("=" * 60)

        # ── Step 1: Load feature tables from Task 6 ──────────────────────────
        print("\n[1/4] Loading feature tables ...")
        uf  = pd.read_csv(f"{self.feature_dir}/user_features.csv")
        itf = pd.read_csv(f"{self.feature_dir}/item_features.csv")
        uif = pd.read_csv(f"{self.feature_dir}/user_item_features.csv")
        co  = pd.read_csv(f"{self.feature_dir}/co_purchase_pairs.csv")
        print(f"      user_features      : {len(uf):>5} rows × {uf.shape[1]} cols")
        print(f"      item_features      : {len(itf):>5} rows × {itf.shape[1]} cols")
        print(f"      user_item_features : {len(uif):>5} rows × {uif.shape[1]} cols")
        print(f"      co_purchase_pairs  : {len(co):>5} rows")

        # ── Step 2: Register features in registry ────────────────────────────
        print("\n[2/4] Registering features in registry ...")
        meta = pd.read_csv(f"{self.feature_dir}/feature_metadata.csv")

        SOURCE_MAP = {
            "clickstream": "user_interaction_logs (Task 2)",
            "transactions": "purchase_history (Task 2)",
            "catalog": "product_catalog (Task 2)",
            "external": "external_signals (Task 2)",
            "derived": "derived (Task 6)",
        }
        for _, row in meta.iterrows():
            self.registry.register_feature(
                name        = row["feature_name"],
                group       = row["group"],
                table       = row["table"],
                pk          = row["primary_key"],
                dtype       = row["dtype"],
                ftype       = row["feature_type"],
                description = row["description"],
                source      = SOURCE_MAP.get(row.get("source","derived"), "derived"),
                version     = row.get("version", "v1.0"),
                tags        = [],
            )

        # Register feature sets (logical groupings for reuse)
        self.registry.register_feature_set(
            "user_basic",
            "Core user engagement features",
            ["user_total_events", "user_purchase_count", "user_purchase_rate",
             "user_recency_days", "user_activity_score_norm"])

        self.registry.register_feature_set(
            "item_popularity",
            "Item engagement and quality features",
            ["item_total_interactions", "item_conversion_rate",
             "item_avg_rating", "item_popularity_score", "item_avg_sentiment"])

        self.registry.register_feature_set(
            "training_default",
            "Full feature set used for model training",
            self.server.DEFAULT_USER_FEATURES + self.server.DEFAULT_ITEM_FEATURES)

        total_registered = len(meta)
        print(f"      {total_registered} features registered across 3 groups")

        # ── Step 3: Write to offline store ───────────────────────────────────
        print("\n[3/4] Writing to offline store ...")

        n, cs = self.offline.write_features(uf,  "user_features",      "user_id")
        self.registry.record_version("user_features", "v1.0", n, cs)
        print(f"      user_features      written  ({n} rows, checksum: {cs[:8]}...)")

        n, cs = self.offline.write_features(itf, "item_features",      "item_id")
        self.registry.record_version("item_features", "v1.0", n, cs)
        print(f"      item_features      written  ({n} rows, checksum: {cs[:8]}...)")

        n, cs = self.offline.write_features(uif, "user_item_features", "user_id")
        self.registry.record_version("user_item_features", "v1.0", n, cs)
        print(f"      user_item_features written  ({n} rows, checksum: {cs[:8]}...)")

        n, cs = self.offline.write_features(co,  "co_purchase_pairs",  "item_a")
        print(f"      co_purchase_pairs  written  ({n} rows, checksum: {cs[:8]}...)")

        # ── Step 4: Materialise to online store ──────────────────────────────
        print("\n[4/4] Materialising to online store ...")
        mat = self.server.materialize_online()
        print(f"      Online store stats: {self.online.stats()}")

        print("\n" + "=" * 60)
        print("Feature Pipeline Complete")
        print("=" * 60)
        return self.server

    def close(self):
        self.registry.close()
        self.offline.close()


# ══════════════════════════════════════════════════════════════════════════════
# 6. DEMO — validate everything works end-to-end
# ══════════════════════════════════════════════════════════════════════════════

def run_demo(server: FeatureServer):
    print("\n" + "─" * 60)
    print("DEMO: Feature Store Retrieval")
    print("─" * 60)

    # 6a. Online user lookup
    sample_user = "USER_0042"
    print(f"\n[Online] User features for {sample_user}:")
    uf = server.get_user_features(sample_user, [
        "user_total_events", "user_purchase_rate",
        "user_avg_spend", "user_recency_days", "user_activity_score_norm"])
    for k, v in (uf or {}).items():
        print(f"  {k:40s}: {v}")

    # 6b. Online item lookup
    sample_item = "ITEM_0017"
    print(f"\n[Online] Item features for {sample_item}:")
    itf = server.get_item_features(sample_item, [
        "item_total_interactions", "item_conversion_rate",
        "item_avg_rating", "item_popularity_score", "item_avg_sentiment"])
    for k, v in (itf or {}).items():
        print(f"  {k:40s}: {v}")

    # 6c. Serving vector for inference
    print(f"\n[Inference] Serving vector for ({sample_user}, {sample_item}):")
    vec = server.get_serving_vector(sample_user, sample_item)
    if vec:
        print(f"  Feature vector length : {len(vec)} features")
        print(f"  Sample entries:")
        for k in list(vec.keys())[:6]:
            print(f"    {k:45s}: {vec[k]}")

    # 6d. Training dataset
    print("\n[Training] Building training dataset ...")
    t0 = time.time()
    ds = server.get_training_dataset(limit=500)
    elapsed = time.time() - t0
    print(f"  Shape             : {ds.shape}")
    print(f"  Label distribution: {ds['ui_has_purchased'].value_counts().to_dict()}")
    print(f"  Retrieval time    : {elapsed*1000:.1f} ms")
    print(f"  Columns           : {list(ds.columns[:8])} ...")

    # 6e. Cache stats
    print(f"\n[Cache] Online store stats: {server.online.stats()}")

    # 6f. Registry listing
    print("\n[Registry] Registered features by group:")
    for grp in ["user", "item", "user_item"]:
        df = server.registry.list_features(group=grp)
        print(f"  {grp:12s}: {len(df)} features")

    return ds


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    pipeline = FeaturePipeline(FEAT_DIR, STORE_DIR)
    server   = pipeline.run()
    ds       = run_demo(server)

    # Save training dataset for Task 9
    ds.to_csv(f"{STORE_DIR}/training_dataset.csv", index=False)
    print(f"\nTraining dataset saved: {STORE_DIR}/training_dataset.csv")

    pipeline.close()
    print("\nFeature Store ready.")
