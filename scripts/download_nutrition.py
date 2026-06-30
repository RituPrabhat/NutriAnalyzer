import os
from pathlib import Path

import pandas as pd
import requests
from dotenv import load_dotenv

# -----------------------------------
# Load API Key
# -----------------------------------
load_dotenv()

API_KEY = os.getenv("USDA_API_KEY")

if not API_KEY:
    raise ValueError("USDA_API_KEY not found in .env")

# -----------------------------------
# Project Paths
# -----------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_FOLDER = PROJECT_ROOT / "data" / "nutrition"
OUTPUT_FOLDER.mkdir(parents=True, exist_ok=True)

OUTPUT_FILE = OUTPUT_FOLDER / "nutrition_data.csv"

# -----------------------------------
# Food List
# -----------------------------------
foods = [
    "apple pie",
    "hamburger",
    "caesar salad",
    "cheesecake",
    "chicken curry",
    "donuts",
    "french fries",
    "fried rice",
    "grilled salmon",
    "hot dog",
    "ice cream",
    "omelette",
    "pancakes",
    "pizza",
    "ramen",
    "steak",
    "sushi",
    "tacos",
    "waffles",
    "chocolate cake"
]

nutrition_data = []

print("Downloading nutrition information...\n")

# -----------------------------------
# Download Data
# -----------------------------------
for i, food in enumerate(foods, start=1):

    print(f"[{i}/{len(foods)}] {food}")

    url = "https://api.nal.usda.gov/fdc/v1/foods/search"

    params = {
        "query": food,
        "pageSize": 1,
        "api_key": API_KEY
    }

    response = requests.get(url, params=params)

    if response.status_code != 200:
        print("Request Failed!\n")
        continue

    result = response.json()

    if len(result.get("foods", [])) == 0:
        print("Food Not Found\n")
        continue

    item = result["foods"][0]

    nutrients = {}

    for nutrient in item.get("foodNutrients", []):

        nutrients[nutrient["nutrientName"]] = nutrient["value"]

    nutrition_data.append({
        "Food": food.replace(" ", "_"),
        "Calories": nutrients.get("Energy"),
        "Protein": nutrients.get("Protein"),
        "Carbs": nutrients.get("Carbohydrate, by difference"),
        "Fat": nutrients.get("Total lipid (fat)")
    })

print("\nCreating CSV...")

df = pd.DataFrame(nutrition_data)

df.to_csv(OUTPUT_FILE, index=False)

print("\nDone!")
print(df)

print(f"\nCSV saved at:\n{OUTPUT_FILE}")