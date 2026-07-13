import pandas as pd

from src.config import NUTRITION_CSV


class NutritionLookup:

    def __init__(self):
        self.df = pd.read_csv(NUTRITION_CSV)

    def get_nutrition(self, food_name):
        result = self.df[self.df["Food"] == food_name]

        if result.empty:
            return None

        row = result.iloc[0]
        return {
            "Food": row["Food"],
            "Calories": float(row["Calories"]),
            "Protein": float(row["Protein"]),
            "Carbs": float(row["Carbs"]),
            "Fat": float(row["Fat"]),
        }