"""Train fraud models, choose an alert threshold by business cost,
then save charts and the files used by the Power BI dashboard.
Run from the project folder:  python src/train.py
"""

from pathlib import Path
import matplotlib
matplotlib.use("Agg")  # draw charts to files, no pop-up windows
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

ROOT = Path(__file__).resolve().parent.parent
CSV_PATH = ROOT / "data" / "creditcard.csv"
FIG_DIR = ROOT / "reports" / "figures"
DASH_DIR = ROOT / "dashboard" / "data"
FIG_DIR.mkdir(parents=True, exist_ok=True)
DASH_DIR.mkdir(parents=True, exist_ok=True)

REVIEW_COST = 5.0  # assumption: cost of an analyst reviewing one flagged transaction
THRESHOLDS = np.round(np.arange(0.01, 1.00, 0.01), 2)
SEED = 42

# ----------------------------------------------------------------------------
# 1. Load the data and split it by time (never shuffle fraud data)
# ----------------------------------------------------------------------------
df = pd.read_csv(CSV_PATH)
df.insert(0, "txn_id", range(len(df)))  # same ids as the SQLite table
df = df.sort_values("Time", kind="stable").reset_index(drop=True)
df["log_amount"] = np.log1p(df["Amount"])
FEATURES = [f"V{i}" for i in range(1, 29)] + ["log_amount"]
n = len(df)
train_end, val_end = int(0.6 * n), int(0.8 * n)
df["split"] = "train"
df.loc[train_end:val_end - 1, "split"] = "validation"
df.loc[val_end:, "split"] = "test"
train = df[df.split == "train"]
val = df[df.split == "validation"]
test = df[df.split == "test"]
for name, part in [("train", train), ("validation", val), ("test", test)]:
    assert part["Class"].sum() > 0, f"No fraud rows in {name}; the split cannot be used."
    print(f"{name:<10} rows={len(part):>7,}  frauds={int(part['Class'].sum()):>4}  "
          f"fraud rate={100 * part['Class'].mean():.3f}%")
X_train, y_train = train[FEATURES], train["Class"]
X_val, y_val = val[FEATURES], val["Class"]
X_test, y_test = test[FEATURES], test["Class"]

# ----------------------------------------------------------------------------
# 2. Train two models
# ----------------------------------------------------------------------------
logreg = make_pipeline(
    StandardScaler(),
    LogisticRegression(class_weight="balanced", max_iter=2000, random_state=SEED),
)
logreg.fit(X_train, y_train)
pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
xgb = XGBClassifier(
    n_estimators=500,
    max_depth=4,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    scale_pos_weight=pos_weight,
    eval_metric="aucpr",
    early_stopping_rounds=30,
    random_state=SEED,
    n_jobs=-1,
)
xgb.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
models = {"Logistic regression": logreg, "XGBoost": xgb}
val_scores = {name: m.predict_proba(X_val)[:, 1] for name, m in models.items()}
test_scores = {name: m.predict_proba(X_test)[:, 1] for name, m in models.items()}

# ----------------------------------------------------------------------------
# 3. Cost-based threshold
#    A missed fraud costs the transaction amount.
#    A flagged transaction costs REVIEW_COST (analyst time), fraud or not.
# ----------------------------------------------------------------------------

def total_cost(y, score, amount, threshold):
    flagged = score >= threshold
    missed_loss = amount[(y == 1) & ~flagged].sum()
    return missed_loss + flagged.sum() * REVIEW_COST

def best_threshold(y, score, amount):
    costs = [total_cost(y, score, amount, t) for t in THRESHOLDS]
    return float(THRESHOLDS[int(np.argmin(costs))])  # first (lowest) threshold if tied

def top_share_metrics(y, score, share=0.01):
    """Precision and recall if analysts review only the top `share` of scores."""
    k = max(1, int(np.ceil(share * len(y))))
    top = np.argsort(-score, kind="stable")[:k]
    tp = int(y.iloc[top].sum())
    return tp / k, tp / int(y.sum())
rows = []
chosen = {}
for name in models:
    thr = best_threshold(y_val, val_scores[name], val["Amount"])  # chosen on validation only
    chosen[name] = thr
    cost_model = total_cost(y_test, test_scores[name], test["Amount"], thr)
    cost_none = test.loc[y_test == 1, "Amount"].sum()  # no model: every fraud is missed
    prec1, rec1 = top_share_metrics(y_test.reset_index(drop=True), test_scores[name])
    rows.append({
        "model": name,
        "val_average_precision": round(average_precision_score(y_val, val_scores[name]), 4),
        "test_average_precision": round(average_precision_score(y_test, test_scores[name]), 4),
        "test_roc_auc": round(roc_auc_score(y_test, test_scores[name]), 4),
        "test_precision_top1pct": round(prec1, 4),
        "test_recall_top1pct": round(rec1, 4),
        "chosen_threshold": thr,
        "test_total_cost": round(cost_model, 2),
        "test_cost_with_no_model": round(cost_none, 2),
        "cost_reduction_pct": round(100 * (1 - cost_model / cost_none), 1),
    })
comparison = pd.DataFrame(rows)
comparison.to_csv(DASH_DIR / "model_comparison.csv", index=False)
print("\n", comparison.T.to_string(header=False))
best_name = comparison.sort_values("val_average_precision", ascending=False).iloc[0]["model"]
best_thr = chosen[best_name]
print(f"\nBest model on validation: {best_name} (threshold {best_thr})")

# ----------------------------------------------------------------------------
# 4. Files for the Power BI dashboard
# ----------------------------------------------------------------------------
# 4a. Threshold table (test set): one row per threshold
y_t, s_t, a_t = y_test.to_numpy(), test_scores[best_name], test["Amount"].to_numpy()
rows = []
for t in THRESHOLDS:
    flagged = s_t >= t
    tp = int(((y_t == 1) & flagged).sum())
    fp = int(((y_t == 0) & flagged).sum())
    fn = int(((y_t == 1) & ~flagged).sum())
    n_flag = tp + fp
    missed = float(a_t[(y_t == 1) & ~flagged].sum())
    prevented = float(a_t[(y_t == 1) & flagged].sum())
    review = n_flag * REVIEW_COST
    rows.append({
        "threshold": float(t), "flagged": n_flag, "true_positives": tp,
        "false_positives": fp, "missed_frauds": fn,
        "precision": round(tp / n_flag, 4) if n_flag else 0.0,
        "recall": round(tp / (tp + fn), 4) if (tp + fn) else 0.0,
        "prevented_loss": round(prevented, 2), "missed_loss": round(missed, 2),
        "review_cost": round(review, 2), "total_cost": round(missed + review, 2),
    })
pd.DataFrame(rows).to_csv(DASH_DIR / "threshold_metrics.csv", index=False)
# 4b. Every transaction, with the chosen model's score
best_model = models[best_name]
df["score"] = best_model.predict_proba(df[FEATURES])[:, 1].round(4)
df["hour_of_day"] = (df["Time"] // 3600 % 24).astype(int)
df["amount_band"] = pd.cut(
    df["Amount"], [0, 10, 50, 100, 250, 500, 1000, np.inf], right=False,
    labels=["1: under 10", "2: 10-50", "3: 50-100", "4: 100-250",
            "5: 250-500", "6: 500-1000", "7: 1000 and above"],
)
df["score_band"] = pd.cut(
    df["score"], [-0.001, 0.2, 0.4, 0.6, 0.8, 1.0],
    labels=["1: 0.0-0.2", "2: 0.2-0.4", "3: 0.4-0.6", "4: 0.6-0.8", "5: 0.8-1.0"],
)
df["class_label"] = np.where(df["Class"] == 1, "Fraud", "Genuine")
df["fraud_rate_pct"] = df["Class"] * 100.0   # the Average of this column is the fraud rate in %
df["fraud_amount"] = df["Amount"] * df["Class"]
df["flagged"] = (df["score"] >= best_thr).astype(int)
df["outcome"] = np.select(
    [(df.Class == 1) & (df.flagged == 1), (df.Class == 0) & (df.flagged == 1),
     (df.Class == 1) & (df.flagged == 0)],
    ["Caught fraud", "False alert", "Missed fraud"], default="Genuine, not flagged",
)
export_cols = ["txn_id", "split", "Time", "hour_of_day", "Amount", "amount_band", "Class",
               "class_label", "fraud_rate_pct", "fraud_amount", "score", "score_band",
               "flagged", "outcome"]
df[export_cols].to_csv(DASH_DIR / "scored_transactions.csv", index=False)

# ----------------------------------------------------------------------------
# 5. Charts for the README
# ----------------------------------------------------------------------------
plt.figure(figsize=(7, 5))
for name in models:
    prec, rec, _ = precision_recall_curve(y_test, test_scores[name])
    ap = average_precision_score(y_test, test_scores[name])
    plt.plot(rec, prec, label=f"{name} (AP = {ap:.3f})")
plt.axhline(y_test.mean(), color="grey", linestyle="--", label="No model (fraud rate)")
plt.xlabel("Recall (share of fraud caught)")
plt.ylabel("Precision (share of alerts that are fraud)")
plt.title("Precision-recall curve, test set")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(FIG_DIR / "pr_curve.png", dpi=150)
plt.close()
val_cost = [total_cost(y_val, val_scores[best_name], val["Amount"], t) for t in THRESHOLDS]
test_cost = [total_cost(y_test, test_scores[best_name], test["Amount"], t) for t in THRESHOLDS]
plt.figure(figsize=(7, 5))
plt.plot(THRESHOLDS, val_cost, label="Validation")
plt.plot(THRESHOLDS, test_cost, label="Test")
plt.axvline(best_thr, color="red", linestyle="--", label=f"Chosen threshold {best_thr:.2f}")
plt.xlabel("Alert threshold")
plt.ylabel(f"Total cost (missed fraud + ${REVIEW_COST:.0f} per review)")
plt.title(f"Cost by threshold, {best_name}")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(FIG_DIR / "cost_curve.png", dpi=150)
plt.close()
sample = X_test.sample(n=min(2000, len(X_test)), random_state=SEED)
shap_values = shap.TreeExplainer(xgb).shap_values(sample)
shap.summary_plot(shap_values, sample, show=False, max_display=15)
plt.title("Which features drive the XGBoost fraud score")
plt.tight_layout()
plt.savefig(FIG_DIR / "shap_summary.png", dpi=150)
plt.close()
print("\nDone. Charts are in reports/figures and dashboard files are in dashboard/data.")
