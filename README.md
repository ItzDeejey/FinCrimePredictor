## 🚀 Financial Fraud Detection Pipeline

An end-to-end, **streaming** fraud-detection workflow built with Polars and Scikit-Learn, designed to handle large datasets efficiently by training incrementally in small batches.

---

### 🔍 Key Features

* **Preview & Sampling**
  – Quickly inspect the first 100 rows with Polars’ `read_csv(n_rows=…)` and export to CSV.
  – Download a lightweight preview of your dataset before full processing.

* **Lazy Streaming Ingestion**
  – Use `pl.scan_csv(...)` for zero-copy, chunked reading.
  – Push down filters and column projections for maximum I/O efficiency.

* **Robust Preprocessing**

  1. Drop unused columns (`time_since_last_transaction`).
  2. Deduplicate rows.
  3. Impute missing `fraud_type` (mark non-fraud as `"No Fraud"`).
  4. Parse ISO-8601 timestamps (with or without microseconds).
  5. Cast text fields (`sender_account`, `ip_address`, …) to UTF-8.
  6. Cast numeric scores and amounts to `Float64`.
  7. Cast categorical fields to `Categorical`, then convert to integer codes (`to_physical()`).

* **Incremental Model Training**
  – Split the preprocessed dataset into **train/test** with reproducible shuffling (`seed=42`).
  – Export training data to an Apache Arrow table and slice into batches (e.g. 5 000 rows each).
  – Train an `SGDClassifier(loss="log_loss")` **via** `partial_fit()`—no out-of-memory errors, perfect for streaming or giant CSVs.

* **Evaluation & Reporting**
  – After training, predict on the hold-out test set.
  – Generate a clear **classification report** (precision, recall, F1-score) using Scikit-Learn.

---

### 🛠 How It Works

```bash
# 1. Mount Drive & Preview
df_preview = pl.read_csv("…/dataset.csv", n_rows=100)
df_preview.write_csv("preview.csv")

# 2. Lazy Load & Preprocess
lf = pl.scan_csv("…/dataset.csv")\
       .drop("time_since_last_transaction")\
       .unique()\
       .with_columns([...])\
       .with_columns([...])\
       …\
       .collect()

# 3. Shuffle & Split
df = lf.sample(fraction=1.0, seed=42)
train, test = df.head(...), df.tail(...)

# 4. Batch & Incremental Train
arrow_tbl = train.to_arrow()
for batch in arrow_tbl.to_batches(batch_size=5000):
    chunk = pl.from_arrow(batch)
    model.partial_fit(X_chunk, y_chunk, classes=[0,1])

# 5. Evaluate
y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred))
```

---

### 📈 Results

* **Streaming-friendly**: never load the entire dataset at once.
* **Reproducible**: fixed random seeds for shuffling and model initialization.
* **Scalable**: process millions of rows in parallel without memory bloat.

---

> **Next steps**:
>
> * Deploy as an API endpoint using FastAPI + Polars
> * Add automated unit tests for each preprocessing step
> * Integrate hyperparameter tuning with `sklearn.model_selection`

---
