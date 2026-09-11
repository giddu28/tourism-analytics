"""Train transparent, dependency-light tourism models and write app artifacts."""
from pathlib import Path
import json
import pickle
import numpy as np
import pandas as pd
from data_pipeline import ARTIFACTS, build_dataset

RANDOM_STATE = 42
FEATURES = ["Continent", "Region", "Country", "CityName", "AttractionType", "Attraction", "VisitYear", "VisitMonth"]


def split(df):
    rng = np.random.default_rng(RANDOM_STATE)
    return df.iloc[rng.random(len(df)) >= 0.2].copy(), df.iloc[rng.random(len(df)) < 0.2].copy()


def safe_div(a, b):
    return float(a / b) if b else 0.0


def classification_metrics(actual, predicted):
    labels = sorted(set(actual) | set(predicted))
    rows = []
    for label in labels:
        tp = int(((actual == label) & (predicted == label)).sum())
        fp = int(((actual != label) & (predicted == label)).sum())
        fn = int(((actual == label) & (predicted != label)).sum())
        p, r = safe_div(tp, tp + fp), safe_div(tp, tp + fn)
        rows.append({"mode": label, "precision": p, "recall": r, "f1": safe_div(2*p*r, p+r), "support": int((actual == label).sum())})
    return {"accuracy": safe_div((actual == predicted).sum(), len(actual)), "macro_f1": float(np.mean([r["f1"] for r in rows])), "by_class": rows}


def fit_naive_bayes(train, target, cols):
    classes = sorted(train[target].unique())
    priors = train[target].value_counts(normalize=True).to_dict()
    tables = {}
    for col in cols:
        counts = train.groupby([target, col]).size().to_dict()
        vocab = max(train[col].nunique(), 1)
        totals = train.groupby(target).size().to_dict()
        tables[col] = {"counts": counts, "vocab": vocab, "totals": totals}
    return {"classes": classes, "priors": priors, "tables": tables, "features": cols}


def predict_naive_bayes(model, records):
    out = []
    for _, row in records.iterrows():
        scores = {}
        for cls in model["classes"]:
            score = np.log(model["priors"].get(cls, 1e-9))
            for col in model["features"]:
                tab = model["tables"][col]
                score += np.log((tab["counts"].get((cls, row[col]), 0) + 1) / (tab["totals"][cls] + tab["vocab"]))
            scores[cls] = score
        out.append(max(scores, key=scores.get))
    return np.array(out)


def rating_predictor(train, test):
    global_mean = train.Rating.mean()
    attraction = train.groupby("Attraction").Rating.agg(["mean", "count"])
    type_mean = train.groupby("AttractionType").Rating.mean().to_dict()
    def pred(row):
        if row.Attraction in attraction.index:
            x = attraction.loc[row.Attraction]
            return (x["count"] * x["mean"] + 10 * type_mean.get(row.AttractionType, global_mean)) / (x["count"] + 10)
        return type_mean.get(row.AttractionType, global_mean)
    values = test.apply(pred, axis=1).to_numpy()
    return values, {"global_mean": float(global_mean), "attraction_stats": attraction.to_dict("index"), "type_mean": type_mean}


def main():
    ARTIFACTS.mkdir(exist_ok=True)
    df = build_dataset()
    df.to_csv(ARTIFACTS / "cleaned_tourism_data.csv", index=False)
    rng = np.random.default_rng(RANDOM_STATE)
    test_mask = rng.random(len(df)) < .20
    train, test = df.loc[~test_mask].copy(), df.loc[test_mask].copy()

    rating_pred, rating_model = rating_predictor(train, test)
    y = test.Rating.to_numpy()
    mse = float(np.mean((y - rating_pred) ** 2))
    r2 = float(1 - np.sum((y-rating_pred)**2) / np.sum((y-y.mean())**2))
    baseline = np.full(len(test), train.Rating.mean())
    baseline_mse = float(np.mean((y-baseline)**2))

    class_features = ["Continent", "Region", "Country", "AttractionType", "Attraction", "VisitMonth", "VisitYear"]
    classifier = fit_naive_bayes(train, "VisitModeName", class_features)
    predicted = predict_naive_bayes(classifier, test)
    class_result = classification_metrics(test.VisitModeName.to_numpy(), predicted)
    majority = train.VisitModeName.mode().iat[0]
    base_class = classification_metrics(test.VisitModeName.to_numpy(), np.full(len(test), majority))

    # Persist a compact user-preference table for recommendations; all seen items are excluded at serving time.
    interactions = train.groupby(["UserId", "AttractionId", "Attraction", "AttractionType"], as_index=False).agg(avg_rating=("Rating", "mean"), visits=("TransactionId", "count"))
    popularity = train.groupby(["AttractionId", "Attraction", "AttractionType"], as_index=False).agg(avg_rating=("Rating", "mean"), visits=("TransactionId", "count"))
    popularity["score"] = popularity.avg_rating * np.log1p(popularity.visits)
    artifacts = {"classifier": classifier, "rating_model": rating_model, "global_popularity": popularity.sort_values("score", ascending=False).to_dict("records"), "interactions": interactions.to_dict("records")}
    with open(ARTIFACTS / "models.pkl", "wb") as f: pickle.dump(artifacts, f)

    insights = {
        "records": int(len(df)), "users": int(df.UserId.nunique()), "attractions": int(df.AttractionId.nunique()),
        "mean_rating": float(df.Rating.mean()), "date_range": [str(df.VisitDate.min().date()), str(df.VisitDate.max().date())],
        "regression": {"model": "Smoothed attraction and type rating", "mse": mse, "rmse": float(np.sqrt(mse)), "r2": r2, "baseline_mse": baseline_mse},
        "classification": {"model": "Categorical naive Bayes", **class_result, "baseline_accuracy": base_class["accuracy"], "baseline_macro_f1": base_class["macro_f1"]},
        "top_attractions": popularity.sort_values(["visits", "avg_rating"], ascending=False).head(10).to_dict("records"),
        "top_regions": df.groupby("Region", as_index=False).agg(visits=("TransactionId", "count"), average_rating=("Rating", "mean")).sort_values("visits", ascending=False).head(10).to_dict("records"),
        "cleaning": ["Removed one duplicated transaction identifier.", "Kept ratings 1 through 5 and visit months 1 through 12.", "Replaced missing user city identifiers and city names with Unknown.", "Used Updated_Item.xlsx as the complete attraction master and joined all reference data."]
    }
    (ARTIFACTS / "metrics.json").write_text(json.dumps(insights, indent=2), encoding="utf-8")
    print(json.dumps(insights["regression"], indent=2)); print(json.dumps(insights["classification"], indent=2))


if __name__ == "__main__":
    main()
