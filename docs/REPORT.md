# Tourism Experience Analytics Report

## Outcome

The project combines 52,930 cleaned visit records, 33,530 travellers, and 30 attractions observed in transactions. It delivers an interactive Streamlit application for tourism trends, visit-mode prediction, rating estimation, and ranked attraction recommendations.

## Data preparation

- Removed one duplicate transaction identifier.
- Validated ratings (1-5) and visit months (1-12).
- Filled four missing user city identifiers and the missing city label as `Unknown`.
- Joined transactions to user geography, visit mode, attraction type, attraction, city, country, region, and continent dimensions.
- Used `Updated_Item.xlsx` as the authoritative attraction master: it contains 1,698 attractions, versus 30 in the original Item file.

The reproducible cleaned table is `artifacts/cleaned_tourism_data.csv`. The source workbooks are untouched.

## Key tourism findings

- Mean visitor rating is 4.17 out of 5, indicating broadly positive recorded experiences.
- Australia leads recorded visit volume (13,322), followed by South East Asia (10,392), Western Europe (8,534), and Northern America (7,847).
- Sacred Monkey Forest Sanctuary has the most visits (10,511); Waterbom Bali combines high volume (5,177) with the highest rating among the leading attractions (4.64).
- Couples is the largest visit-mode class. This imbalance is reflected in the predictive-task evaluation and should be addressed before using the classifier to drive automated campaign decisions.

## Models and validation

An 80/20 seeded holdout is used to keep evaluation separate from fitting.

| Task | Model | Result | Comparison |
| --- | --- | --- | --- |
| Rating prediction | Smoothed attraction and attraction-type historical rating | RMSE 0.937; R2 0.093; MSE 0.878 | Global-rating baseline MSE 0.969 |
| Visit-mode classification | Categorical naive Bayes using traveller geography, attraction, month, and year | Accuracy 43.6%; macro F1 29.0% | Majority-class baseline accuracy 41.2%; macro F1 11.7% |
| Recommendations | Hybrid popularity and user attraction-type affinity | Excludes attractions a known visitor has already seen | Ranked on ratings, visit volume, and type affinity |

The rating model is a useful benchmark but has limited explanatory power. The classifier improves materially on the naive baseline in macro F1, while minority-mode performance remains weak. Use predictions for decision support and test targeted promotion with controlled experiments.

## Recommended actions

1. Feature high-rated, high-volume attractions such as Waterbom Bali in broad acquisition campaigns.
2. Balance campaigns by travel mode and collect richer itinerary, party-size, and budget features before automated segmentation.
3. Use recommendations to cross-sell unseen attraction types that match each visitor's highly rated history.
4. Monitor ratings and visit mix by region monthly; investigate major changes before changing supply or pricing.

## Running the application

Install the packages in `requirements.txt`, rebuild artifacts with `python src/train_models.py`, and launch with `streamlit run src/app.py`.
