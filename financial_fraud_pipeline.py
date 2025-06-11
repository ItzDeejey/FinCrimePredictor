# %% [markdown]
# 🚀 Financial Fraud Detection Pipeline

This notebook provides an **end-to-end streaming fraud detection workflow** using Polars and Scikit-Learn. It is designed for large datasets, processing data in chunks and training an incremental model efficiently.

---

# %% [markdown]
## 1. Setup & Data Preview

Mount Google Drive, import libraries, and preview the first 100 rows of the dataset.

---

# %%
from google.colab import drive
import polars as pl, numpy as np
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import classification_report

drive.mount('/content/drive')  # Mount Drive

# Preview first 100 rows
df_preview = pl.read_csv(
    "/content/drive/MyDrive/Colab Notebooks/financial_fraud_detection_dataset.csv",
    n_rows=100
)
df_preview.head()

# %% [markdown]
## 2. Lazy Ingestion & Preprocessing

Load the full dataset lazily, apply transformations, and prepare features.

---

# %%
def parse_timestamp(col):
    return (
        pl.when(pl.col(col).str.contains(r"\.\d+$"))
          .then(pl.col(col)
                .str.strptime(pl.Datetime, "%Y-%m-%dT%H:%M:%S%.f", strict=False))
          .otherwise(pl.col(col)
                     .str.strptime(pl.Datetime, "%Y-%m-%dT%H:%M:%S", strict=False))
    )

# Column groups
num_cols = ['spending_deviation_score', 'velocity_score', 'geo_anomaly_score', 'amount']
cat_cols = ['transaction_type', 'merchant_category', 'location', 'device_used', 'payment_channel']
text_cols = ['sender_account', 'receiver_account', 'ip_address', 'device_hash']

# Lazy load and preprocess
lf = (
    pl.scan_csv(
        "/content/drive/MyDrive/Colab Notebooks/financial_fraud_detection_dataset.csv"
    )
    .drop('time_since_last_transaction')
    .unique()
    .with_columns([
        pl.when((pl.col("is_fraud") == False)
                & (pl.col("fraud_type").is_null() | (pl.col("fraud_type") == "")))
          .then(pl.lit("No Fraud"))
          .otherwise(pl.col("fraud_type"))
          .alias("fraud_type")
    ])
    .with_columns([
        *[pl.col(c).cast(pl.Utf8) for c in text_cols],
        parse_timestamp("timestamp").alias("timestamp"),
        *[pl.col(c).cast(pl.Float64) for c in num_cols],
        *[pl.col(c).cast(pl.Categorical) for c in cat_cols]
    ])
)

lf.schema

# %% [markdown]
### Convert categoricals to codes

To use categorical features in the model, convert them to integer codes.

---

# %%
for c in cat_cols:
    lf = lf.with_columns([pl.col(c).to_physical().cast(pl.Int64).alias(c + '_code')])

final_features = num_cols + [c + '_code' for c in cat_cols]

# %% [markdown]
## 3. Collect, Shuffle & Split

Collect the preprocessed data, shuffle for randomness, and split into train/test sets.

---

# %%
# Collect full dataset
df_full = lf.collect()

# Shuffle
df_full = df_full.sample(fraction=1.0, seed=42)

# Split
test_frac = 0.4
split_idx = int(df_full.shape[0] * (1 - test_frac))
train_df = df_full.head(split_idx)
test_df = df_full.tail(df_full.shape[0] - split_idx)

# %% [markdown]
## 4. Incremental Training

Train an `SGDClassifier` in batches using Apache Arrow slicing.

---

# %%
# Prepare Arrow table for batching
arrow_tbl = train_df.to_arrow()
batch_size = 5000
num_rows = arrow_tbl.num_rows
num_batches = (num_rows + batch_size - 1) // batch_size

# Initialize model
model = SGDClassifier(loss='log_loss', random_state=42)
classes = np.array([0, 1])
first = True

# Batch training
for i in range(num_batches):
    start, end = i * batch_size, min((i + 1) * batch_size, num_rows)
    batch = pl.from_arrow(arrow_tbl.slice(offset=start, length=end-start).to_batches()[0])
    X = batch.select(final_features).to_numpy()
    y = batch.select('is_fraud').to_numpy().ravel()
    if first:
        model.partial_fit(X, y, classes=classes)
        first = False
    else:
        model.partial_fit(X, y)
print("Training complete.")

# %% [markdown]
## 5. Evaluation

Generate precision, recall, F1-score on the test set.

---

# %%
X_test = test_df.select(final_features).to_numpy()
y_test = test_df.select('is_fraud').to_numpy().ravel()
y_pred = model.predict(X_test)
print(classification_report(y_test=y_test, y_pred=y_pred))
