import os
from pathlib import Path

import pandas as pd
import requests
from dotenv import load_dotenv

load_dotenv()

API_KEY = os.getenv("USDA_API_KEY")
if not API_KEY:
    raise ValueError("USDA_API_KEY not found in .env")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_FOLDER = PROJECT_ROOT / "data" / "nutrition"
OUTPUT_FOLDER.mkdir(parents=True, exist_ok=True)
OUTPUT_FILE = OUTPUT_FOLDER / "nutrition_data.csv"

foods = [
    "apple pie", "hamburger", "caesar salad", "cheesecake", "chicken curry",
    "donuts", "french fries", "fried rice", "grilled salmon", "hot dog",
    "ice cream", "omelette", "pancakes", "pizza", "ramen", "steak",
    "sushi", "tacos", "waffles", "chocolate cake",
]

URL = "https://api.nal.usda.gov/fdc/v1/foods/search"


def extract_nutrients(item):
    """Return dict of nutrients, using Energy only when the unit is kcal."""
    out = {"Calories": None, "Protein": None, "Carbs": None, "Fat": None}
    for n in item.get("foodNutrients", []):
        name = n.get("nutrientName", "")
        unit = (n.get("unitName") or "").upper()
        value = n.get("value")
        if name == "Energy" and unit == "KCAL":
            out["Calories"] = value
        elif name == "Protein":
            out["Protein"] = value
        elif name == "Carbohydrate, by difference":
            out["Carbs"] = value
        elif name == "Total lipid (fat)":
            out["Fat"] = value
    return out


def pick_best(foods_list):
    """Pick the first result that has calories in kcal; fall back to first."""
    for item in foods_list:
        nutrients = extract_nutrients(item)
        if nutrients["Calories"] is not None:
            return nutrients
    return extract_nutrients(foods_list[0]) if foods_list else None


nutrition_data = []
print("Downloading nutrition information...\n")

for i, food in enumerate(foods, start=1):
    print(f"[{i}/{len(foods)}] {food}")

    params = {
        "query": food,
        "pageSize": 10,
        # Prefer curated data over crowd-sourced Branded entries
        "dataType": ["Foundation", "SR Legacy"],
        "api_key": API_KEY,
    }

    response = requests.get(URL, params=params, timeout=30)
    if response.status_code != 200:
        print(f"  Request failed ({response.status_code})\n")
        continue

    results = response.json().get("foods", [])
    if not results:
        print("  Not found\n")
        continue

    nutrients = pick_best(results)
    if nutrients is None:
        print("  No usable nutrients\n")
        continue

    nutrition_data.append({
        "Food": food.replace(" ", "_"),
        **nutrients,
    })

print("\nCreating CSV...")
df = pd.DataFrame(nutrition_data)
df.to_csv(OUTPUT_FILE, index=False)
print("\nDone!")
print(df)
print(f"\nCSV saved at:\n{OUTPUT_FILE}")