# Tourism Experience Analytics

An end-to-end tourism analytics project that turns historical visitor activity into actionable insights, rating estimates, visit-mode predictions, and personalised attraction recommendations.

The project is delivered as a reproducible data pipeline and an interactive Streamlit application. It uses only the supplied tourism workbooks and does not modify the source data.

## Contents

- [Business problem](#business-problem)
- [Project objectives](#project-objectives)
- [Dataset](#dataset)
- [Data preparation](#data-preparation)
- [Exploratory findings](#exploratory-findings)
- [Modelling approach and results](#modelling-approach-and-results)
- [Recommendation approach](#recommendation-approach)
- [Streamlit application](#streamlit-application)
- [Installation and usage](#installation-and-usage)
- [Repository structure](#repository-structure)
- [Limitations and next steps](#limitations-and-next-steps)

## Business problem

Travel platforms need to understand where travellers go, what they enjoy, and how they travel. This project supports three business needs:

1. **Improve customer experience** by suggesting attractions aligned with a traveller's past preferences.
2. **Support targeted marketing** by estimating a visitor's likely travel mode, such as Couples, Family, Friends, Solo, or Business.
3. **Improve destination planning** by surfacing popular attractions, high-volume regions, and satisfaction patterns.

## Project objectives

| Objective | Output | Business use |
| --- | --- | --- |
| Rating regression | Estimated rating on the 1-5 scale | Flag attractions or experiences that may need attention and set traveller expectations. |
| Visit-mode classification | Most likely visit mode | Tailor packages, amenities, and marketing messages. |
| Attraction recommendations | Ranked unseen attractions | Increase engagement and cross-sell relevant destinations. |
| Exploratory analysis | Interactive trends and rankings | Help stakeholders identify hotspots and visitor patterns. |

## Dataset

The source files are kept in `data/raw/`. The final modelling table combines transaction history with visitor, location, attraction, attraction type, and visit-mode dimensions.

| Source | Main fields | Purpose |
| --- | --- | --- |
| `Transaction.xlsx` | user, attraction, year, month, mode, rating | The fact table for visits and satisfaction. |
| `User.xlsx` | continent, region, country, city | Visitor demographic geography. |
| `Updated_Item.xlsx` | attraction, city, type, address | Authoritative attraction master. |
| Geographic lookups | city, country, region, continent | Human-readable geographic context. |
| `Type.xlsx` and `Mode.xlsx` | attraction type and visit mode | Features and target labels. |

`Updated_Item.xlsx` is deliberately used instead of the smaller `Item.xlsx` because it contains **1,698** attractions, compared with 30 in the original subset. The transaction history references 30 observed attractions.

### Data coverage

| Measure | Value |
| --- | ---: |
| Cleaned visit records | 52,930 |
| Unique visitors | 33,530 |
| Attractions observed in transactions | 30 |
| Mean rating | 4.16 / 5 |
| Visit period | Jan 2013 to Oct 2022 |

## Data preparation

The reproducible cleaning and integration logic lives in [`src/data_pipeline.py`](src/data_pipeline.py).

1. Removes the duplicate transaction identifier.
2. Retains valid ratings from 1 to 5 and valid visit months from 1 to 12.
3. Replaces missing user city identifiers and missing city labels with `Unknown`.
4. Maps visit-mode identifiers to readable categories.
5. Joins transactions with visitor geography, attraction information, attraction types, and geographic dimensions.
6. Creates `VisitDate` and `Quarter` features for time-series analysis.
7. Writes the integrated modelling data to `artifacts/cleaned_tourism_data.csv` when the pipeline is run.

Generated files in `artifacts/` are intentionally excluded from Git because they can be regenerated locally.

## Exploratory findings

### Regional demand

| Region | Visits | Average rating |
| --- | ---: | ---: |
| Australia | 13,322 | 4.22 |
| South East Asia | 10,392 | 4.13 |
| Western Europe | 8,534 | 4.12 |
| Northern America | 7,847 | 4.17 |
| South Asia | 2,874 | 4.22 |

### Leading attractions by recorded visits

| Attraction | Type | Visits | Average rating |
| --- | --- | ---: | ---: |
| Sacred Monkey Forest Sanctuary | Nature and Wildlife Areas | 10,511 | 4.26 |
| Waterbom Bali | Water Parks | 5,177 | 4.64 |
| Tegalalang Rice Terrace | Points of Interest and Landmarks | 4,635 | 4.17 |
| Uluwatu Temple | Religious Sites | 2,706 | 4.21 |
| Tanah Lot Temple | Religious Sites | 2,683 | 4.20 |

### Business interpretation

- Australia and South East Asia account for the largest recorded volumes and should be primary markets for capacity planning and campaign measurement.
- Waterbom Bali combines strong demand with the highest satisfaction among the leading attractions, making it a compelling broad-acquisition feature.
- Couples is the largest visit-mode class. This imbalance affects classification performance, so prediction results should support rather than replace marketing judgement.

## Modelling approach and results

Model building is implemented in [`src/train_models.py`](src/train_models.py). Evaluation uses a seeded 80/20 holdout split, so the metrics below reflect records not used to fit the models.

### Rating prediction

The regression benchmark predicts a rating from a smoothed blend of each attraction's historical mean and its attraction-type mean. Smoothing prevents very small attraction samples from producing unstable estimates.

| Metric | Rating model | Global-mean baseline |
| --- | ---: | ---: |
| Mean squared error | **0.878** | 0.969 |
| Root mean squared error | **0.937** | 0.984 |
| R-squared | 0.093 | - |

The model improves on the global-rating baseline, but the low R-squared shows that satisfaction depends on factors not supplied in the data, such as itinerary, price, weather, service quality, and party composition.

### Visit-mode classification

A categorical naive Bayes classifier estimates visit mode from visitor geography, attraction type, attraction, month, and year. It provides a transparent baseline suitable for an initial product version.

| Metric | Categorical naive Bayes | Majority-class baseline |
| --- | ---: | ---: |
| Accuracy | **43.6%** | 41.2% |
| Macro F1 | **29.0%** | 11.7% |

| Visit mode | Precision | Recall | F1 | Holdout records |
| --- | ---: | ---: | ---: | ---: |
| Business | 4.9% | 13.8% | 7.2% | 130 |
| Couples | 51.7% | 61.1% | 56.0% | 4,349 |
| Family | 43.5% | 46.3% | 44.9% | 2,946 |
| Friends | 31.8% | 22.8% | 26.6% | 2,193 |
| Solo | 19.6% | 7.0% | 10.3% | 931 |

The model has useful lift over the majority baseline, particularly in macro F1, but minority modes remain difficult to detect. Do not use it for automated customer treatment without monitoring, fairness checks, and richer inputs.

## Recommendation approach

The app uses an explainable hybrid ranking approach:

1. Start with historical attraction popularity, combining average rating and visit volume.
2. For a known user, infer preferred attraction types from the ratings in their visit history.
3. For a new user, use their selected preferred attraction type.
4. Exclude attractions the known user has already visited.
5. Rank remaining candidates with a 70% popularity score and a 30% personal attraction-type affinity score.

This is deliberately transparent and robust for a sparse interaction dataset. A future version can compare matrix factorisation, nearest-neighbour collaborative filtering, and offline ranking metrics such as MAP@K or NDCG@K.

## Streamlit application

The application entry point is [`src/app.py`](src/app.py). It includes four views:

| View | What it provides |
| --- | --- |
| Executive overview | Core KPIs, travel volume over time, visit-mode mix, and stakeholder interpretation. |
| Prediction and recommendations | Visit-mode prediction, estimated rating, and ranked unseen attraction recommendations. |
| Tourism insights | Regional demand, attraction-type demand versus satisfaction, and top-attraction detail. |
| Model evaluation | Held-out regression and classification metrics, baseline comparisons, and per-class performance. |

## Installation and usage

### Prerequisites

- Python 3.10 or later
- `pip`

### Local setup

```bash
git clone https://github.com/giddu28/tourism-analytics.git
cd tourism-analytics
python -m venv .venv
```

Activate the environment:

```bash
# Windows PowerShell
.venv\Scripts\Activate.ps1

# macOS or Linux
source .venv/bin/activate
```

Install dependencies, create the generated data/model artifacts, and launch the app:

```bash
pip install -r requirements.txt
python src/train_models.py
streamlit run src/app.py
```

The app will open locally, usually at `http://localhost:8501`.

## Repository structure

```text
tourism-analytics/
├── data/raw/                  # Supplied Excel source files
├── docs/                      # Project brief and concise report
├── src/                       # Streamlit app, pipeline, and training logic
├── artifacts/                 # Generated locally; ignored by Git
├── .gitignore
├── LICENSE
├── pyproject.toml
├── requirements.txt
└── README.md
```

## Limitations and next steps

1. **Class imbalance:** Add class weights, resampling, and threshold calibration; assess performance by segment before activating targeted campaigns.
2. **Feature scope:** Collect party size, budget, trip duration, booking channel, weather, and text feedback to improve rating and visit-mode predictions.
3. **Recommendation evaluation:** Add a time-aware offline split and evaluate ranking quality with MAP@K, Recall@K, and NDCG@K.
4. **Model comparison:** Benchmark tree-based and gradient-boosted models against the transparent baselines.
5. **Deployment readiness:** Add automated tests, data validation, experiment tracking, and a Streamlit deployment configuration before production use.

## License

This project is available under the [MIT License](LICENSE).
