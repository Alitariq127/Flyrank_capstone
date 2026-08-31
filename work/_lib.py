"""
Shared pipeline logic for the Refresh / Content Opportunity Scoring capstone.

IMPORTANT — READ THIS FIRST:
`load_search_data()` below returns SYNTHETIC data shaped like the real
FlyRank warehouse schema (page, date, clicks, impressions, ctr, position,
word_count, category, days_since_publish). It exists so the full pipeline
(labels -> features -> split -> model -> reason codes -> recommendations)
can be built, tested, and demonstrated end-to-end before real data is wired in.

TO SWAP IN REAL DATA: replace load_search_data() with the DuckDB + hf://
query from starter notebook 03, and keep the same output schema:
  columns = [page_id, date, clicks, impressions, ctr, position,
             word_count, category, days_since_publish]
Everything downstream (label logic, split, model, reason codes) works
unchanged as long as that schema is preserved.
"""

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# 1. Synthetic data generator (placeholder for the real hf:// / DuckDB query)
# ---------------------------------------------------------------------------
def load_search_data_REAL(use_sample=True):
    """
    REAL data loader against the FlyRank Hugging Face warehouse.
    Set use_sample=True first (fast, ~145KB file) to confirm the query works,
    then switch to use_sample=False for the full 78.8M-row fact table.
    """
    import os
    import duckdb
    from dotenv import load_dotenv

    load_dotenv()  # reads HF_TOKEN from your local .env file
    token = os.environ["HF_TOKEN"]

    con = duckdb.connect()
    con.sql(f"SET hf_token='{token}';")

    fact_path = (
        "hf://datasets/FlyRank/internship-warehouse/fact_content_daily_performance_sample.parquet"
        if use_sample else
        "hf://datasets/FlyRank/internship-warehouse/fact_content_daily_performance/*.parquet"
    )

    df = con.sql(f"""
        WITH perf AS (
            SELECT
                client_hash_id,
                content_hash_id,
                report_date,
                gsc_clicks AS clicks,
                gsc_impressions AS impressions,
                gsc_avg_position AS position
            FROM '{fact_path}'
            WHERE gsc_data_available = true
        ),
        content AS (
            SELECT client_hash_id, content_hash_id, word_count, keyword_created_date
            FROM 'hf://datasets/FlyRank/internship-warehouse/dim_content.parquet'
        ),
        joined AS (
            SELECT
                perf.client_hash_id || '_' || perf.content_hash_id AS page_id,
                perf.report_date,
                perf.clicks,
                perf.impressions,
                perf.position,
                content.word_count,
                content.keyword_created_date
            FROM perf
            JOIN content
              ON perf.client_hash_id = content.client_hash_id
             AND perf.content_hash_id = content.content_hash_id
        ),
        weekly AS (
            SELECT
                page_id,
                DATE_TRUNC('week', report_date) AS week_start,
                SUM(clicks) AS clicks,
                SUM(impressions) AS impressions,
                AVG(position) AS position,
                MAX(word_count) AS word_count,
                MAX(keyword_created_date) AS keyword_created_date
            FROM joined
            GROUP BY page_id, week_start
        )
        SELECT
            page_id,
            DENSE_RANK() OVER (ORDER BY week_start) - 1 AS week,
            clicks,
            impressions,
            CASE WHEN impressions > 0 THEN clicks::DOUBLE / impressions ELSE 0 END AS ctr,
            position,
            word_count,
            DATE_DIFF('day', keyword_created_date, week_start) AS days_since_publish
        FROM weekly
        ORDER BY page_id, week
    """).df()

    # No category dimension exists in this warehouse (checked dim_content columns) —
    # kept as a constant so build_features()'s one-hot step doesn't break, but it
    # carries no signal. Safe to drop later if you confirm it's unused.
    df["category"] = "unknown"
    return df


def load_search_data(n_pages=500, n_weeks=20, seed=42):
    """
    SYNTHETIC placeholder — currently still used by all notebooks.
    Once load_search_data_REAL() is confirmed working (see notebook 01, cell 1),
    come back here and rename load_search_data_REAL -> load_search_data
    (or just change the notebooks' import) to switch the whole pipeline to real data.
    """
    rng = np.random.default_rng(seed)

    categories = ["guide", "product", "comparison", "landing", "blog", "faq"]
    page_ids = [f"page_{i:04d}" for i in range(n_pages)]

    # Give each page a latent "trajectory" archetype so labels aren't random noise
    archetypes = rng.choice(
        ["growing", "declining", "stable", "volatile"],
        size=n_pages,
        p=[0.2, 0.2, 0.45, 0.15],
    )

    page_meta = pd.DataFrame({
        "page_id": page_ids,
        "category": rng.choice(categories, size=n_pages),
        "word_count": rng.integers(300, 4000, size=n_pages),
        "publish_week_offset": rng.integers(0, 200, size=n_pages),  # weeks before window start
        "archetype": archetypes,
        "base_impressions": rng.integers(200, 20000, size=n_pages),
        "base_ctr": np.clip(rng.normal(0.035, 0.015, size=n_pages), 0.002, 0.25),
    })

    rows = []
    for _, p in page_meta.iterrows():
        base_imp = p["base_impressions"]
        base_ctr = p["base_ctr"]
        for w in range(n_weeks):
            # trend multiplier depends on archetype
            if p["archetype"] == "growing":
                trend = 1 + 0.03 * w
            elif p["archetype"] == "declining":
                trend = max(0.15, 1 - 0.035 * w)
            elif p["archetype"] == "volatile":
                trend = 1 + rng.normal(0, 0.25)
            else:  # stable
                trend = 1 + rng.normal(0, 0.05)

            noise = rng.normal(1, 0.08)
            impressions = max(0, int(base_imp * trend * noise / n_weeks * 4))
            ctr = np.clip(base_ctr * rng.normal(1, 0.1), 0.001, 0.4)
            clicks = int(impressions * ctr)
            position = np.clip(rng.normal(15 - 8 * (ctr / base_ctr), 3), 1, 100)

            rows.append({
                "page_id": p["page_id"],
                "week": w,
                "clicks": clicks,
                "impressions": impressions,
                "ctr": round(clicks / impressions, 4) if impressions else 0.0,
                "position": round(position, 1),
                "word_count": p["word_count"],
                "category": p["category"],
                "days_since_publish": (p["publish_week_offset"] + w) * 7,
            })

    df = pd.DataFrame(rows)
    return df


# ---------------------------------------------------------------------------
# 2. Label definition (computed ONLY from a future, non-overlapping window)
# ---------------------------------------------------------------------------
FEATURE_WEEKS = range(0, 12)   # weeks 0-11: used for features
LABEL_WEEKS = range(12, 16)    # weeks 12-15: used ONLY for labels (held-out future)
TEST_LABEL_WEEKS = range(16, 20)  # weeks 16-19: final untouched test period


def assert_no_window_overlap():
    """Leakage check: feature and label windows must never overlap."""
    f, l, t = set(FEATURE_WEEKS), set(LABEL_WEEKS), set(TEST_LABEL_WEEKS)
    assert f.isdisjoint(l), "Feature and label windows overlap!"
    assert f.isdisjoint(t), "Feature and test windows overlap!"
    assert l.isdisjoint(t), "Label and test windows overlap!"
    print("Leakage check passed: feature / label / test windows are disjoint.")
    print(f"  feature weeks: {min(f)}-{max(f)}")
    print(f"  label weeks (train/val target): {min(l)}-{max(l)}")
    print(f"  test weeks (held out): {min(t)}-{max(t)}")


def compute_labels(df, label_weeks, growth_threshold=0.15):
    """
    Label = growing / declining / recovering / stable, based on primary
    metric = clicks, comparing the label window's total clicks against a
    trailing baseline immediately preceding it (also inside the label
    window construction — never touches the feature window's raw values
    beyond an already-computed trailing baseline that we also treat as
    label-side for the purposes of leakage).
    """
    sub = df[df["week"].isin(label_weeks)].copy()
    half = len(list(label_weeks)) // 2
    weeks_sorted = sorted(label_weeks)
    early_half, late_half = weeks_sorted[:half], weeks_sorted[half:]

    early = sub[sub["week"].isin(early_half)].groupby("page_id")["clicks"].sum()
    late = sub[sub["week"].isin(late_half)].groupby("page_id")["clicks"].sum()

    result = pd.DataFrame({"early_clicks": early, "late_clicks": late}).fillna(0)
    result["pct_change"] = np.where(
        result["early_clicks"] > 0,
        (result["late_clicks"] - result["early_clicks"]) / result["early_clicks"],
        np.where(result["late_clicks"] > 0, 1.0, 0.0),
    )

    def classify(pct):
        if pct >= growth_threshold:
            return "growing"
        elif pct <= -growth_threshold:
            return "declining"
        else:
            return "stable"

    result["label"] = result["pct_change"].apply(classify)
    return result[["label", "pct_change"]].reset_index()


def mark_recovering(labels_df, prior_labels_df):
    """A page is 'recovering' if it was declining in the prior period and growing now."""
    merged = labels_df.merge(
        prior_labels_df[["page_id", "label"]].rename(columns={"label": "prior_label"}),
        on="page_id", how="left",
    )
    was_declining = merged["prior_label"] == "declining"
    now_growing = merged["label"] == "growing"
    merged.loc[was_declining & now_growing, "label"] = "recovering"
    return merged.drop(columns=["prior_label"])


# ---------------------------------------------------------------------------
# 3. Feature engineering (computed ONLY from the feature window)
# ---------------------------------------------------------------------------
def build_features(df, feature_weeks):
    sub = df[df["week"].isin(feature_weeks)].copy()
    weeks_sorted = sorted(feature_weeks)

    agg = sub.groupby("page_id").agg(
        avg_clicks=("clicks", "mean"),
        avg_impressions=("impressions", "mean"),
        avg_ctr=("ctr", "mean"),
        avg_position=("position", "mean"),
        std_clicks=("clicks", "std"),
        word_count=("word_count", "first"),
        category=("category", "first"),
        days_since_publish=("days_since_publish", "max"),
    ).fillna(0)

    # simple trend slope within the feature window (early half vs late half)
    half = len(weeks_sorted) // 2
    early_w, late_w = weeks_sorted[:half], weeks_sorted[half:]
    early = sub[sub["week"].isin(early_w)].groupby("page_id")["clicks"].mean()
    late = sub[sub["week"].isin(late_w)].groupby("page_id")["clicks"].mean()
    agg["trend_slope"] = (late - early).reindex(agg.index).fillna(0)

    agg = pd.get_dummies(agg, columns=["category"], prefix="cat")
    return agg.reset_index()


# ---------------------------------------------------------------------------
# 4. Reason codes (human-readable "why" for each flagged page)
# ---------------------------------------------------------------------------
def generate_reason_code(row):
    reasons = []
    if row.get("trend_slope", 0) < -5:
        reasons.append("clicks trending down within observation window")
    elif row.get("trend_slope", 0) > 5:
        reasons.append("clicks trending up within observation window")

    if row.get("avg_ctr", 0) < 0.02:
        reasons.append("CTR below typical range for its position")
    if row.get("avg_position", 0) > 20:
        reasons.append("ranking outside top 20 on average")
    if row.get("days_since_publish", 0) > 730:
        reasons.append("content is 2+ years old, candidate for refresh")

    if not reasons:
        reasons.append("no strong signal — monitor")
    return "; ".join(reasons)


def recommend_action(predicted_label, row):
    if predicted_label == "declining":
        if row.get("avg_position", 0) <= 10:
            return "improve"       # still ranks well, worth defending/refreshing
        else:
            return "rewrite"
    elif predicted_label == "recovering":
        return "protect"
    elif predicted_label == "growing":
        return "protect"
    else:  # stable
        if row.get("word_count", 0) < 500:
            return "improve"
        return "monitor"
    
    