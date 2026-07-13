import os
import time
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
    "baklava", "bibimbap", "bruschetta", "dumplings", "edamame",
    "falafel", "fish and chips", "garlic bread", "greek salad",
    "grilled cheese sandwich", "guacamole", "lasagna",
    "macaroni and cheese", "miso soup", "nachos", "onion rings",
    "pad thai", "paella", "samosa", "spring rolls",
]

URL = "https://api.nal.usda.gov/fdc/v1/foods/search"

# Try FNDDS first — it's USDA's "as prepared/eaten" dataset, so dishes like
# "caesar salad" or "sushi" match a real recipe entry instead of a random
# raw ingredient. Foundation/SR Legacy (raw ingredients) is the fallback,
# for simple foods FNDDS doesn't cover.
DATA_TYPE_STAGES = [
    ["Survey (FNDDS)"],
    ["Foundation", "SR Legacy"],
]


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


def search_foods(query, data_types, retries=3):
    params = {
        "query": query,
        "pageSize": 10,
        "dataType": data_types,
        "api_key": API_KEY,
    }
    for attempt in range(1, retries + 1):
        response = requests.get(URL, params=params, timeout=30)
        if response.status_code == 200:
            return response.json().get("foods", [])
        # USDA's API intermittently 400s valid requests — retry a couple
        # times before treating it as a real failure.
        if attempt < retries:
            time.sleep(1)
    print(f"  Request failed ({response.status_code}) after {retries} attempts")
    return []


nutrition_data = []
print("Downloading nutrition information...\n")

for i, food in enumerate(foods, start=1):
    print(f"[{i}/{len(foods)}] {food}")

    nutrients = None
    for data_types in DATA_TYPE_STAGES:
        results = search_foods(food, data_types)
        if results:
            nutrients = pick_best(results)
            if nutrients is not None:
                break

    if nutrients is None:
        print("  Not found\n")
        continue

    calories = nutrients["Calories"] or 0
    macro_calories = (nutrients["Protein"] or 0) * 4 \
        + (nutrients["Carbs"] or 0) * 4 \
        + (nutrients["Fat"] or 0) * 9
    if calories and abs(macro_calories - calories) > 0.25 * calories:
        print(
            f"  WARNING - macro mismatch: label says {calories:.0f} kcal but "
            f"protein/carbs/fat imply ~{macro_calories:.0f} kcal - "
            "this match is probably wrong, double check it manually."
        )

    nutrition_data.append({
        "Food": food.replace(" ", "_"),
        **nutrients,
    })
    print()

print("\nCreating CSV...")
df = pd.DataFrame(nutrition_data)
df.to_csv(OUTPUT_FILE, index=False)
print("\nDone!")
print(df)
print(f"\nCSV saved at:\n{OUTPUT_FILE}")