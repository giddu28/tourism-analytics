# Tourism Experience Analytics

This project implements the supplied brief: data cleaning, EDA, rating prediction, visit-mode classification, and an explainable hybrid recommendation ranking in a Streamlit app.

## Run

1. Install dependencies with `pip install -r requirements.txt` in a Python 3.10+ environment.
2. Run `python src/train_models.py` to rebuild the cleaned data and model artifacts.
3. Run `streamlit run src/app.py`.

`Updated_Item.xlsx` is used as the authoritative attraction master because it includes the full set of 1,698 attraction records. The application reads generated artifacts from `artifacts/` and never modifies the supplied source workbooks.

## Repository layout

```text
data/raw/       Supplied source workbooks
docs/           Problem statement and project report
src/            Streamlit app, data pipeline, and model training code
artifacts/      Generated cleaned data and models (ignored; rebuild locally)
```
