# 🩺 Diabetes Predictor

> *“Prevention is better than cure. Let’s predict before it’s too late.”*

This project predicts the likelihood of a person having **diabetes** based on clinical features using **Machine Learning**.
It includes everything from **data preprocessing → model training → evaluation → deployment** with a simple, interactive **web app**.

---

## ✨ Features

* 📊 **Exploratory Data Analysis (EDA)** with charts & distributions
* 🧹 **Data Cleaning & Preprocessing** (missing values, scaling, outliers)
* 🤖 **Multiple ML Models** tried & compared (Logistic Regression, Decision Tree, Random Forest, etc.)
* 🏆 **Best Model Exported** as `model.pkl`
* 💻 **Web Application** for interactive predictions (Streamlit/Flask)
* 📈 **Performance Metrics**: Accuracy, Precision, Recall, F1-score, ROC-AUC
* 🔍 **Feature Importance** visualization
* ✅ **Input Validation** to handle invalid user inputs

---

## 🗂️ Project Structure

```
Diabetes_Predictor/
├── dataset/diabetes.csv              # Dataset (if included, else load via Kaggle/UCI)
├── notebooks/Diabetes_Predictor.ipynb # EDA + Training + Evaluation
├── model.pkl                          # Saved ML model
├── app.py                             # Streamlit/Flask app
├── requirements.txt                   # Dependencies
└── README.md                          # Project README
```

---

## ⚙️ Installation & Setup

### 1️⃣ Clone the Repository

```bash
git clone https://github.com/duanshi-26/Diabetes_Predictor.git
cd Diabetes_Predictor
```

### 2️⃣ Create Virtual Environment (Recommended)

```bash
python -m venv venv
# Windows
venv\Scripts\activate
# macOS/Linux
source venv/bin/activate
```

### 3️⃣ Install Dependencies

```bash
pip install -r requirements.txt
```

---

## 🚀 How to Run

### 🔬 Train Model (Optional)

If you want to retrain from scratch:

```bash
jupyter notebook notebooks/Diabetes_Predictor.ipynb
```

### 🖥️ Run the Web App

```bash
streamlit run app.py
```

➡️ Then open the local URL: `http://localhost:8501/`
---

## 🧪 Using Model in Python

```python
import pickle
import numpy as np

# Load trained model
with open("model.pkl", "rb") as f:
    model = pickle.load(f)

# Example input → [Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DPF, Age]
sample = np.array([[2, 148, 72, 35, 0, 33.6, 0.627, 50]])
prediction = model.predict(sample)
print("Diabetes:", "Yes" if prediction[0] == 1 else "No")
```

---

## 📊 Example Results

| Metric    | Score |
| --------- | ----- |
| Accuracy  | 85%   |
| Precision | 82%   |
| Recall    | 79%   |
| F1-Score  | 80%   |
| ROC-AUC   | 0.87  |

---

## 📷 Screenshots

<p align="center">
  <img src="docs/app-home.png" width="600" alt="App Home"/>
  <br/>
  <em>Prediction App Interface</em>
</p>

---

## 🤝 Contributing

1. Fork the repo
2. Create a branch: `git checkout -b feature/new-feature`
3. Commit changes: `git commit -m "Added new feature"`
4. Push: `git push origin feature/new-feature`
5. Open a Pull Request 🚀

---

## 📜 License

This project is licensed under the **MIT License** – feel free to use and improve it!

---

## 👨‍💻 Maintainer

**Duanshi Chawla**
🌐 [Your GitHub Profile](https://github.com/duanshi-26)




