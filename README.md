# 🧠 AttritionIQ — Employee Attrition Prediction Platform

> An intelligent HR analytics web application built with Streamlit and XGBoost that predicts employee attrition risk and provides workforce insights through an interactive dashboard.

---

[![Live Demo](https://img.shields.io/badge/🚀_Live_Demo-Streamlit-FF4B4B?style=for-the-badge&logo=streamlit)](https://employee-attrition-app-bgg3vfacmharof74v6ytoy.streamlit.app/)

## 📌 Table of Contents

- [Project Overview](#-project-overview)
- [Features](#-features)
- [Project Structure](#-project-structure)
- [Prerequisites](#-prerequisites)
- [Installation & Setup](#-installation--setup)
- [Running the App](#-running-the-app)
- [Using the App](#-using-the-app)
- [Model Details](#-model-details)
- [Troubleshooting](#-troubleshooting)
- [Deployment to Streamlit Cloud](#-deployment-to-streamlit-cloud)
- [Demo](#-demo)

---

## 📖 Project Overview

AttritionIQ is a full-stack HR intelligence platform that uses a trained **XGBoost classification model** to predict whether an employee is likely to leave the organization. It also provides visual analytics to help HR teams understand workforce trends and attrition drivers at a glance.

The model was trained on employee data with 25+ features covering personal info, job details, satisfaction scores, and work environment factors. It uses a **custom probability threshold** (tuned for best F1 score) rather than the default 0.5 cutoff.

---

## ✨ Features

### 🎯 Prediction Tab
- Full employee input form with all features used during model training
- Real-time attrition risk prediction with probability score
- Interactive gauge chart showing risk level
- Color-coded result badge — **HIGH RISK** (red) or **LOW RISK** (green)
- Automatic key risk factor detection based on inputs
- Demo mode if pickle files are missing (heuristic fallback)

### 📊 Dashboard Tab
- **KPI cards** — Total employees, attrition rate, at-risk count, avg tenure
- **Attrition by Department** — Horizontal bar chart with color-coded risk levels
- **Attrition by Job Level** — Entry vs Mid vs Senior breakdown
- **Overtime Impact** — Side-by-side attrition comparison
- **Work-Life Balance Trend** — Line chart showing attrition correlation
- **Age Distribution** — Overlapping histogram: Stayed vs Left
- **Income Distribution** — Box plot comparing income by attrition outcome
- **SHAP Feature Importance** — Top 10 attrition drivers with directional push

---

## 📁 Project Structure

```
Attrition_ML_Project/
│
├── notebooks/
│   └── EmployeAttrition_MLproject_Final.ipynb   ← Full ML training notebook
│
├── static/
│   └── demo.mp4                                  ← App demo video
│
├── attrition_env/                  ← Virtual environment (auto-created, don't edit)
│
├── app.py                          ← Main Streamlit application
├── requirements.txt                ← Python package dependencies
├── README.md                       ← This file
│
├── attrition_pipeline.pkl          ← Trained XGBoost pipeline (preprocessor + model)
├── best_threshold.pkl              ← Optimal classification threshold
└── feature_names.pkl               ← Feature column names used during training
```

> ⚠️ **Important:** The three `.pkl` files must be in the **same folder as `app.py`**, NOT inside `attrition_env/`.

---

## 🔧 Prerequisites

Before setting up the project, make sure you have:

- **Python 3.9 or higher** installed → [Download Python](https://www.python.org/downloads/)
- **pip** available (comes with Python)
- **VS Code** (recommended) → [Download VS Code](https://code.visualstudio.com/)
- The three pickle files exported from your Google Colab notebook:
  - `attrition_pipeline.pkl`
  - `best_threshold.pkl`
  - `feature_names.pkl`

---

## 🚀 Installation & Setup

### Step 1 — Open the project folder in VS Code
Open the `Attrition_ML_Project` folder in VS Code and open the integrated terminal:
```
Terminal → New Terminal
```

### Step 2 — Create a virtual environment
```powershell
python -m venv attrition_env --without-pip
```

### Step 3 — Activate the virtual environment
```powershell
attrition_env\Scripts\activate
```
You should now see `(attrition_env)` at the start of your terminal prompt.

### Step 4 — Install pip inside the venv
```powershell
python -m ensurepip --upgrade
```

### Step 5 — Install all dependencies
```powershell
pip install streamlit pandas numpy plotly scikit-learn xgboost imbalanced-learn shap
```
This may take **3–5 minutes** depending on your internet speed.

### Step 6 — Place your pickle files
Copy your three `.pkl` files into the project root (same folder as `app.py`):
```
Attrition_ML_Project/
├── attrition_pipeline.pkl   
├── best_threshold.pkl        
├── feature_names.pkl         
└── app.py
```

---

## ▶️ Running the App

Make sure your venv is active (`(attrition_env)` is visible in the terminal), then run:

```powershell
streamlit run app.py
```

The app will automatically open in your browser at:
```
http://localhost:8501
```

To **stop** the app, press `Ctrl + C` in the terminal.

---

## 🖥️ Using the App

### Making a Prediction

1. Click the **"🎯 Predict Attrition"** tab
2. Fill in the employee details across four sections:
   - **Personal Information** — Age, Gender, Marital Status, Education, Dependents
   - **Job Details** — Role, Level, Years at Company, Tenure, Promotions, Income, Distance
   - **Work Environment** — Company Size, Reputation, Overtime, Remote Work, Work-Life Balance
   - **Satisfaction & Growth** — Job Satisfaction, Performance Rating, Recognition, Leadership, Innovation
3. Click **"⚡ Predict Attrition Risk"**
4. View the result on the right panel:
   - Risk badge (HIGH / LOW)
   - Probability gauge (0–100%)
   - Key risk factors detected
   - Model threshold used

### Viewing the Dashboard

1. Click the **"📊 Dashboard & Analytics"** tab
2. Explore charts and KPIs across the workforce
3. Note: Dashboard uses **sample data** for visualization purposes

---

## 🤖 Model Details

| Property | Details |
|----------|---------|
| Algorithm | XGBoost Classifier |
| Pipeline | ColumnTransformer → XGBoost |
| Preprocessing | IterativeImputer, StandardScaler, OrdinalEncoder, OneHotEncoder |
| Imbalance Handling | Class weights via `scale_pos_weight` |
| Tuning | RandomizedSearchCV (40 iterations, 5-fold StratifiedKFold) |
| Threshold | Tuned for best F1 score (not default 0.5) |
| Explainability | SHAP TreeExplainer |

### Input Features Used

| Category | Features |
|----------|----------|
| Numerical | Age, Distance from Home, Years at Company, Number of Promotions, Number of Dependents, Company Tenure (Months), Monthly Income (log), Promotion Rate, Income vs Role Avg |
| Ordinal | Work-Life Balance, Performance Rating, Job Level, Company Reputation, Education Level, Job Satisfaction, Employee Recognition |
| Binary | Gender, Overtime, Remote Work, Leadership Opportunities, Innovation Opportunities, HighPerf_NoPromo |
| One-Hot | Job Role, Marital Status, Company Size |

> 📝 `Income_vs_Role_Avg` is set to `1.0` (average) for single-employee predictions since it requires population-level data to compute.

---

## 🛠️ Troubleshooting

### `streamlit: command not found`
Your venv may not be active. Run:
```powershell
attrition_env\Scripts\activate
```

### `ModuleNotFoundError: No module named 'xgboost'`
Packages weren't installed in the venv. Make sure `(attrition_env)` is active, then:
```powershell
pip install xgboost
```

### App shows "⚠️ Model files not found"
The `.pkl` files are not in the right place. Make sure all three pickle files are in the **same folder as `app.py`**, not inside `attrition_env/`.

### `KeyboardInterrupt` during venv creation
Use the `--without-pip` flag:
```powershell
python -m venv attrition_env --without-pip
python -m ensurepip --upgrade
```

### Port 8501 already in use
```powershell
streamlit run app.py --server.port 8502
```

---

## ☁️ Deployment to Streamlit Cloud (Free)

1. Push your project to a **public GitHub repository**
   ```bash
   git init
   git add app.py requirements.txt README.md *.pkl
   git commit -m "Initial commit"
   git remote add origin https://github.com/YOUR_USERNAME/YOUR_REPO.git
   git push -u origin main
   ```

2. Go to [share.streamlit.io](https://share.streamlit.io) and sign in with GitHub

3. Click **"New app"** → Select your repo → Set main file to `app.py`

4. Click **"Deploy"** 🚀

---

## 🎬 Demo

![AttritionIQ Demo](static/attrition_demo_video.gif)

---

## 👨‍💻 Author

Built by **Abhishek** as part of a Data Science & ML project on Employee Attrition Prediction.

---


