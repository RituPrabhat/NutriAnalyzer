# 🥗 Nutrition Analyzer

AI Nutrition Analyzer is a deep learning-based web application that identifies food from an uploaded image and displays its nutritional information.

The project combines computer vision and a Flask web application to provide an easy way to estimate the nutritional values of different food items.

---

## 📸 Screenshots

<img width="1920" height="1080" alt="Screenshot (1688)" src="https://github.com/user-attachments/assets/b83fa331-d2f7-42ff-9728-d71b5ca693bf" />

<img width="1920" height="1080" alt="image" src="https://github.com/user-attachments/assets/026db6ba-d436-483b-b725-76e50c50a85e" />

<img width="1920" height="1080" alt="image" src="https://github.com/user-attachments/assets/45b4f7a6-e737-40d1-bb74-3f16660b55c4" />

---

## Features

- Upload a food image through the browser
- Predict the food item using a trained deep learning model
- Display confidence score
- Show nutritional information including:
  - Calories
  - Protein
  - Carbohydrates
  - Fat
- Simple and responsive user interface
- Built using Flask and PyTorch

  ---

## 🛠️ Technologies Used

- Python
- PyTorch
- Torchvision
- Flask
- HTML
- CSS
- Pandas
- Pillow

  ---

## 📂 Project Structure

```
Nutrition/
│
├── data/
│   └── nutrition/
│       └── nutrition_data.csv
│
├── notebooks/
│
├── scripts/
│
├── src/
│   ├── image_classifier/
│   └── nutrition_engine/
│
├── webapp/
│   ├── static/
│   ├── templates/
│   └── app.py
│
├── food_classifier.pth
└── README.md
```
---
##  How to Run

### Clone the repository

```bash
git clone https://github.com/RituPrabhat/NutriAnalyzer.git
```

Move into the project folder

```bash
cd NutriAnalyzer
```

Create a virtual environment

```bash
python -m venv venv
```

Activate the environment

**Windows**

```bash
venv\Scripts\activate
```

Install the required packages

```bash
pip install -r requirements.txt
```

Run the Flask application

```bash
python webapp/app.py
```

Open your browser and visit

```
http://127.0.0.1:5000
```

---
##  Model

The food classifier is built using **ResNet18** with transfer learning.

The model was trained on selected classes from the Food-101 dataset.

---
## 🥗 Nutrition Information

After predicting the food item, the application retrieves nutrition values from a nutrition dataset containing:

- Calories
- Protein
- Carbohydrates
- Fat

  ---

## 🔮 Future Improvements

- Support more food categories
- Improve model accuracy
- Deploy the application online
- Display Top-5 predictions
- Add image preview before prediction
- Make the UI fully responsive

---

## 👩‍💻 Author

**Ritu Prabhat**

GitHub: https://github.com/RituPrabhat






