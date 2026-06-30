from pathlib import Path
import pandas as pd


class NutritionLookup:

    def __init__(self):

        csv_path = (
            Path(__file__).resolve().parents[2]
            / "data"
            / "nutrition"
            / "nutrition_data.csv"
        )

        self.df = pd.read_csv(csv_path)

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
    "Fat": float(row["Fat"])
}