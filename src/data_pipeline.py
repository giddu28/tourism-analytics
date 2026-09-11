"""Build the modelling table from the supplied tourism workbooks."""
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
RAW_DATA = ROOT / "data" / "raw"
ARTIFACTS = ROOT / "artifacts"


def read_book(name, sheet=0):
    return pd.read_excel(RAW_DATA / name, sheet_name=sheet)


def build_dataset():
    tx = read_book("Transaction.xlsx").drop_duplicates("TransactionId").copy()
    users = read_book("User.xlsx").copy()
    cities = read_book("City.xlsx").copy()
    countries = read_book("Country.xlsx")
    regions = read_book("Region.xlsx")
    continents = read_book("Continent.xlsx")
    modes = read_book("Mode.xlsx")
    types = read_book("Type.xlsx")
    # Updated_Item is the complete attraction master; Item only contains an early subset.
    items = read_book("Updated_Item.xlsx")

    tx = tx[tx["Rating"].between(1, 5) & tx["VisitMonth"].between(1, 12)]
    users["CityId"] = users["CityId"].fillna(0).astype(int)
    cities["CityName"] = cities["CityName"].fillna("Unknown")

    frame = tx.merge(users, on="UserId", how="left", validate="many_to_one")
    modes = modes.rename(columns={"VisitMode": "VisitModeName"})
    frame = frame.merge(modes, left_on="VisitMode", right_on="VisitModeId", how="left")
    frame = frame.merge(items, on="AttractionId", how="left", validate="many_to_one")
    frame["AttractionTypeId"] = pd.to_numeric(frame["AttractionTypeId"], errors="coerce")
    frame = frame.merge(types, on="AttractionTypeId", how="left")
    frame = frame.merge(continents, on="ContinentId", how="left")
    frame = frame.merge(regions[["RegionId", "Region"]], on="RegionId", how="left")
    frame = frame.merge(countries[["CountryId", "Country"]], on="CountryId", how="left")
    frame = frame.merge(cities[["CityId", "CityName"]], on="CityId", how="left")

    frame["VisitModeName"] = frame["VisitModeName"].fillna("Unknown").astype(str)
    frame["AttractionType"] = frame["AttractionType"].fillna("Other")
    for col in ["Continent", "Region", "Country", "CityName", "Attraction"]:
        frame[col] = frame[col].fillna("Unknown").astype(str)
    frame["VisitDate"] = pd.to_datetime(dict(year=frame.VisitYear, month=frame.VisitMonth, day=1))
    frame["Quarter"] = "Q" + frame["VisitDate"].dt.quarter.astype(str)
    return frame


def main():
    ARTIFACTS.mkdir(exist_ok=True)
    df = build_dataset()
    df.to_csv(ARTIFACTS / "cleaned_tourism_data.csv", index=False)
    print(f"Saved {len(df):,} clean records with {df.shape[1]} columns.")


if __name__ == "__main__":
    main()
