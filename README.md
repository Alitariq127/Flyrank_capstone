# Refresh / Content Opportunity Scoring — FlyRank Capstone

Lane: **Refresh / Content Opportunity Scoring**. Full pipeline (labels → features →
split → model vs. baseline → reason codes → ranked recommendations → paper), built
and validated end-to-end. Currently runs on synthetic data shaped like the real
warehouse schema — see the swap-in instructions below to point it at the real data.

## Repo structure
```
work/               notebooks 01-07, run in order, plus _lib.py (shared pipeline logic)
work/data_cache/    intermediate parquet/joblib files (gitignored — regenerate by rerunning notebooks)
docs/               the deployed paper (index.html) + charts/tables (assets/)
submission/         paper_url.txt — put your deployed URL here once live
```

## Swapping in the real FlyRank data (the one thing left to do)

Everything downstream of data loading is already built and tested. To go from
synthetic demo to real data:

1. Get your Hugging Face read token and confirm access to the FlyRank warehouse
   release (see the capstone card's resources / starter notebook 03 for the exact
   dataset path).
2. Open `work/_lib.py`, find `load_search_data()`, and replace its body with a
   DuckDB query over `hf://`, e.g.:
   ```python
   import duckdb

   def load_search_data():
       con = duckdb.connect()
       con.sql(f"SET hf_token='{YOUR_TOKEN}';")  # load from env var, not hardcoded
       df = con.sql("""
           SELECT page_id, week, clicks, impressions, ctr, position,
                  word_count, category, days_since_publish
           FROM 'hf://datasets/<org>/<dataset>/<path>/*.parquet'
       """).df()
       return df
   ```
3. **Keep the output schema identical**: `page_id, week, clicks, impressions, ctr,
   position, word_count, category, days_since_publish`. If the real warehouse uses
   different column names or a `date` column instead of `week`, adapt those inside
   `load_search_data()` only — don't touch the rest of `_lib.py` or the notebooks.
4. Re-run notebooks 01 → 07 in order (`jupyter nbconvert --to notebook --execute --inplace work/0X_*.ipynb`,
   or just open and run each one). New charts and the recommendation table will
   overwrite the ones in `docs/assets/`.
5. Re-check the numbers quoted in `docs/index.html` (Results section, the metric
   cards, and the ranked recommendations table) against the new output and update
   them to match — the current numbers are from the synthetic run.
6. Before publishing: confirm nothing in `docs/index.html` or the notebooks leaked
   a client name, real domain, private query string, credential, or causal claim
   about Google's algorithm.

## Deploying the paper
1. Push this repo to GitHub.
2. Repo **Settings → Pages → Source → Deploy from branch → `main` / `docs` folder**.
3. Copy the resulting `https://<username>.github.io/<repo>/` URL into
   `submission/paper_url.txt` as the only line in that file, and commit.

## Data & leakage discipline
- Feature window: weeks 0–11. Label window (train/val): weeks 12–15. Test window
  (held out until the very end): weeks 16–19.
- `work/_lib.py::assert_no_window_overlap()` runs at the top of the label and
  capstone notebooks and hard-fails if the windows ever overlap.
