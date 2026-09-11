"""Tourism Experience Analytics Streamlit application."""
from pathlib import Path
import json, pickle
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

ROOT = Path(__file__).resolve().parent.parent
ARTIFACTS = ROOT / "artifacts"

st.set_page_config(page_title="Tourism Experience Analytics", page_icon="✈️", layout="wide")

@st.cache_data
def load_data(): return pd.read_csv(ARTIFACTS / "cleaned_tourism_data.csv", parse_dates=["VisitDate"])
@st.cache_resource
def load_models():
    with open(ARTIFACTS / "models.pkl", "rb") as f: return pickle.load(f)
@st.cache_data
def load_metrics(): return json.loads((ARTIFACTS / "metrics.json").read_text(encoding="utf-8"))

def predict_mode(model, record):
    scores = {}
    for cls in model["classes"]:
        score = np.log(model["priors"].get(cls, 1e-9))
        for col in model["features"]:
            tab = model["tables"][col]
            score += np.log((tab["counts"].get((cls, record[col]), 0) + 1) / (tab["totals"][cls] + tab["vocab"]))
        scores[cls] = score
    best = max(scores, key=scores.get)
    exp = {k: np.exp(v-max(scores.values())) for k,v in scores.items()}; total=sum(exp.values())
    return best, {k: v/total for k,v in exp.items()}

def recommend(data, models, user_id, preferred_type, n=5):
    base = pd.DataFrame(models["global_popularity"])
    seen = set(data.loc[data.UserId == user_id, "AttractionId"])
    # For known users, reward types that earned high ratings; otherwise use their stated preference.
    user_history = data[data.UserId == user_id]
    if not user_history.empty:
        type_scores = user_history.groupby("AttractionType").Rating.mean().to_dict()
        base["personal_score"] = base.AttractionType.map(type_scores).fillna(0)
    else:
        base["personal_score"] = (base.AttractionType == preferred_type).astype(float) * 5
    base = base[~base.AttractionId.isin(seen)].copy()
    base["recommendation_score"] = .7 * base.score/base.score.max() + .3 * base.personal_score/5
    return base.sort_values("recommendation_score", ascending=False).head(n)

data, models, metrics = load_data(), load_models(), load_metrics()
st.title("Tourism Experience Analytics")
st.caption("Ratings prediction, visit-mode classification, and explainable attraction recommendations")
page = st.sidebar.radio("Explore", ["Executive overview", "Prediction and recommendations", "Tourism insights", "Model evaluation"])

if page == "Executive overview":
    c1,c2,c3,c4 = st.columns(4)
    c1.metric("Visits", f"{metrics['records']:,}"); c2.metric("Visitors", f"{metrics['users']:,}")
    c3.metric("Attractions", f"{metrics['attractions']:,}"); c4.metric("Average rating", f"{metrics['mean_rating']:.2f}/5")
    left,right = st.columns(2)
    monthly=data.groupby("VisitDate",as_index=False).size().rename(columns={"size":"Visits"})
    left.plotly_chart(px.line(monthly,x="VisitDate",y="Visits",title="Visit volume over time"), use_container_width=True)
    modes=data.VisitModeName.value_counts().rename_axis("Visit mode").reset_index(name="Visits")
    right.plotly_chart(px.bar(modes,x="Visit mode",y="Visits",color="Visit mode",title="Travel by visit mode"), use_container_width=True)
    st.subheader("Stakeholder takeaways")
    st.write("Use the recommendations page to surface unseen attractions, focus capacity and promotion on high-volume regions, and use predicted visit mode to tailor packages. Model scores are held-out estimates, not causal conclusions.")

elif page == "Prediction and recommendations":
    st.subheader("Plan a traveller experience")
    known = st.checkbox("Use a known visitor history", value=True)
    user_id = st.selectbox("Visitor", sorted(data.UserId.unique())) if known else -1
    default = data[data.UserId == user_id].iloc[0] if known else data.iloc[0]
    a,b,c,d = st.columns(4)
    continent=a.selectbox("Continent", sorted(data.Continent.unique()), index=sorted(data.Continent.unique()).index(default.Continent))
    region=b.selectbox("Region", sorted(data.Region.unique()), index=sorted(data.Region.unique()).index(default.Region))
    attraction_type=c.selectbox("Preferred attraction type", sorted(data.AttractionType.unique()), index=sorted(data.AttractionType.unique()).index(default.AttractionType))
    month=d.selectbox("Travel month", list(range(1,13)), index=int(default.VisitMonth)-1)
    attraction = st.selectbox("Attraction for rating estimate", sorted(data.Attraction.unique()))
    row = data[data.Attraction == attraction].iloc[0]
    record={"Continent":continent,"Region":region,"Country":default.Country,"AttractionType":attraction_type,"Attraction":attraction,"VisitMonth":month,"VisitYear":int(default.VisitYear)}
    mode, probs=predict_mode(models["classifier"], record)
    rm=models["rating_model"]; astats=rm["attraction_stats"].get(attraction)
    pred_rating=((astats["count"]*astats["mean"]+10*rm["type_mean"].get(row.AttractionType,rm["global_mean"]))/(astats["count"]+10)) if astats else rm["type_mean"].get(row.AttractionType,rm["global_mean"])
    x,y=st.columns(2); x.metric("Likely visit mode",mode); y.metric("Estimated rating",f"{pred_rating:.2f} / 5")
    st.caption("Mode prediction uses traveller geography, attraction, season, and historical patterns. Rating estimate is a regularized historical benchmark by attraction and type.")
    st.subheader("Recommended unseen attractions")
    recs=recommend(data,models,user_id,attraction_type)
    st.dataframe(recs[["Attraction","AttractionType","avg_rating","visits","recommendation_score"]].rename(columns={"avg_rating":"Average rating","visits":"Historical visits","recommendation_score":"Recommendation score"}),use_container_width=True,hide_index=True)

elif page == "Tourism insights":
    st.subheader("Attractions and regions")
    left,right=st.columns(2)
    regions=data.groupby("Region",as_index=False).agg(Visits=("TransactionId","count"),Average_rating=("Rating","mean")).sort_values("Visits",ascending=False).head(12)
    left.plotly_chart(px.bar(regions,x="Visits",y="Region",orientation="h",title="Top regions by visits"),use_container_width=True)
    types=data.groupby("AttractionType",as_index=False).agg(Visits=("TransactionId","count"),Average_rating=("Rating","mean"))
    right.plotly_chart(px.scatter(types,x="Visits",y="Average_rating",hover_name="AttractionType",size="Visits",title="Attraction type demand and satisfaction"),use_container_width=True)
    top=data.groupby(["Attraction","AttractionType"],as_index=False).agg(Visits=("TransactionId","count"),Average_rating=("Rating","mean")).sort_values("Visits",ascending=False).head(15)
    st.dataframe(top,use_container_width=True,hide_index=True)

else:
    st.subheader("Held-out model performance")
    r=metrics["regression"]; c=metrics["classification"]
    a,b,c1=st.columns(3); a.metric("Rating RMSE",f"{r['rmse']:.3f}"); b.metric("Rating R²",f"{r['r2']:.3f}"); c1.metric("Visit mode accuracy",f"{c['accuracy']:.1%}")
    st.write(f"The rating model MSE is {r['mse']:.3f}, compared with {r['baseline_mse']:.3f} for the global-mean baseline. The classifier macro F1 is {c['macro_f1']:.1%}, versus {c['baseline_macro_f1']:.1%} for the majority-class baseline.")
    st.plotly_chart(px.bar(pd.DataFrame(c['by_class']),x="mode",y=["precision","recall","f1"],barmode="group",title="Visit-mode classification by class"),use_container_width=True)
    st.info("Validation uses a seeded 80/20 random holdout. Recommendation ranking is evaluated operationally: candidates already visited are excluded and relevance is based on historical rating and attraction-type affinity.")
