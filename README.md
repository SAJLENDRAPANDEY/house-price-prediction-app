<div align="center">

<img src="https://readme-typing-svg.demolab.com?font=Fira+Code&size=13&pause=1000&color=C9A84C&center=true&vCenter=true&width=500&lines=Machine+Learning+%7C+Flask+API+%7C+Docker+%7C+HF+Spaces" alt="Typing SVG" />

# 🏠 House Price Predictor

**End-to-end ML web application — from raw data to live deployment**

[![Live Demo](https://img.shields.io/badge/🚀%20Live%20Demo-Hugging%20Face-FFD21E?style=for-the-badge&logo=huggingface&logoColor=black)](https://huggingface.co/spaces/sajlendra/house-price-predictor-app)
[![GitHub](https://img.shields.io/badge/GitHub-Repository-181717?style=for-the-badge&logo=github&logoColor=white)](https://github.com/SAJLENDRAPANDEY/house-price-predictor)
[![Python](https://img.shields.io/badge/Python-3.10-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![Flask](https://img.shields.io/badge/Flask-API-000000?style=for-the-badge&logo=flask&logoColor=white)](https://flask.palletsprojects.com)

<br/>

> A production-ready ML application that predicts real estate prices using **Random Forest & Gradient Boosting**, served via a **Flask REST API**, and deployed on **Hugging Face Spaces with Docker**.

<br/>

![separator](https://raw.githubusercontent.com/andreasbm/readme/master/assets/lines/colored.png)

</div>

<br/>

## 📸 Preview

<div align="center">
<table>
<tr>
<td align="center"><b>🎨 Interactive UI</b></td>
<td align="center"><b>⚡ Instant Prediction</b></td>
</tr>
<tr>
<td>Gold-accented dark theme with animated inputs</td>
<td>Real-time Flask API response in milliseconds</td>
</tr>
</table>
</div>

<br/>

## ✨ Highlights

```
🤖  ML Pipeline     →   Data cleaning · Feature engineering · Model training · Cross-validation
🌐  REST API        →   Flask backend with /predict endpoint (JSON in/out)
🎨  Modern UI       →   Dark theme · Animated · Fully responsive HTML/CSS/JS
☁️  Cloud Deploy    →   Hugging Face Spaces · Dockerized · Zero-config startup
```

<br/>

## 🧠 Machine Learning Pipeline

<div align="center">

| Stage | Details |
|-------|---------|
| **Data Source** | King County House Sales dataset (21,000+ records) |
| **Cleaning** | Null handling, duplicate removal, type casting |
| **Feature Engineering** | Log transform on price, outlier removal (IQR), sqft ratios |
| **Encoding** | One-Hot Encoding for city (27 categories) |
| **Scaling** | `StandardScaler` on all numeric features |
| **Models** | Linear Regression → Random Forest → Gradient Boosting |
| **Evaluation** | R² Score, RMSE, Cross-Validation (5-fold) |
| **Serialization** | `joblib` → `model.pkl`, `scaler.pkl`, `columns.pkl` |

</div>

<br/>

## 🛠️ Tech Stack

<div align="center">

| Layer | Technology | Purpose |
|-------|-----------|---------|
| **Language** | ![Python](https://img.shields.io/badge/Python-3776AB?logo=python&logoColor=white) | Core logic & ML |
| **ML** | ![scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?logo=scikitlearn&logoColor=white) | Model training & evaluation |
| **Data** | ![Pandas](https://img.shields.io/badge/Pandas-150458?logo=pandas&logoColor=white) ![NumPy](https://img.shields.io/badge/NumPy-013243?logo=numpy&logoColor=white) | Data processing |
| **Backend** | ![Flask](https://img.shields.io/badge/Flask-000000?logo=flask&logoColor=white) | REST API (`/predict`) |
| **Frontend** | ![HTML5](https://img.shields.io/badge/HTML5-E34F26?logo=html5&logoColor=white) ![CSS3](https://img.shields.io/badge/CSS3-1572B6?logo=css3&logoColor=white) ![JS](https://img.shields.io/badge/JavaScript-F7DF1E?logo=javascript&logoColor=black) | Interactive UI |
| **Deploy** | ![Docker](https://img.shields.io/badge/Docker-2496ED?logo=docker&logoColor=white) ![HuggingFace](https://img.shields.io/badge/HuggingFace-FFD21E?logo=huggingface&logoColor=black) | Cloud hosting |

</div>

<br/>

## 📂 Project Structure

```
house-price-predictor/
│
├── 📄 app.py                    # Flask backend — serves UI + /predict endpoint
├── 🌐 house_predictor.html      # Frontend — dark themed, animated, responsive
│
├── 🤖 model.pkl                 # Trained ML model (Random Forest / GB)
├── ⚖️  scaler.pkl               # Fitted StandardScaler
├── 📋 columns.pkl               # Feature column names (incl. city OHE)
│
├── 📦 requirements.txt          # Python dependencies
├── 🐳 Dockerfile                # Container config for HF Spaces
└── 📖 README.md
```

<br/>

## ⚙️ Run Locally

### 1 · Clone & install

```bash
git clone https://github.com/SAJLENDRAPANDEY/house-price-predictor
cd house-price-predictor
pip install -r requirements.txt
```

### 2 · Start the server

```bash
python app.py
```

### 3 · Open in browser

```
http://localhost:7860
```

<br/>

## 🔌 API Reference

**`POST /predict`**

```json
// Request body
{
  "bedrooms":    3,
  "bathrooms":   2.0,
  "sqft_living": 2000,
  "sqft_lot":    5000,
  "floors":      2,
  "waterfront":  0,
  "condition":   4,
  "yr_built":    2005,
  "city":        "Seattle"
}

// Response
{
  "price": 524800.00
}
```

<br/>

## 🧪 Example Prediction

```
Input  →  3 bed · 2 bath · 2,000 sqft · 2 floors · Seattle · Condition 4
Output →  ₹ 5,24,800
```

<br/>

## 🚀 Deployment — Hugging Face + Docker

The app runs inside a Docker container on Hugging Face Spaces:

```dockerfile
FROM python:3.10-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
EXPOSE 7860
CMD ["python", "app.py"]
```

```
requirements.txt
─────────────────
flask
pandas
numpy
scikit-learn
joblib
```

<br/>

## 🔮 Roadmap

- [x] Random Forest & Gradient Boosting models
- [x] Flask REST API with JSON response
- [x] One-hot city encoding (27 cities)
- [x] Docker deployment on Hugging Face
- [ ] XGBoost integration for higher accuracy
- [ ] Analytics dashboard with price distribution charts
- [ ] Mobile-first UI redesign
- [ ] More location features (zip code, neighborhood)

<br/>

## 👨‍💻 Author

<div align="center">

<img src="https://avatars.githubusercontent.com/SAJLENDRAPANDEY" width="80" style="border-radius:50%"/>

### Sajlendra Pandey
*B.Tech CSE (Data Science) · MDU Rohtak · 2027*

[![Portfolio](https://img.shields.io/badge/Portfolio-sajlendrapandey.netlify.app-C9A84C?style=flat-square&logo=safari&logoColor=white)](https://sajlendrapandey.netlify.app/)
[![LinkedIn](https://img.shields.io/badge/LinkedIn-Connect-0A66C2?style=flat-square&logo=linkedin&logoColor=white)](https://www.linkedin.com/in/sajlendra-pandey-37378627b/)
[![GitHub](https://img.shields.io/badge/GitHub-Follow-181717?style=flat-square&logo=github&logoColor=white)](https://github.com/SAJLENDRAPANDEY)

</div>

<br/>

## ⭐ Support

If this project helped you or impressed you:

```
⭐ Star this repo  ·  🔗 Share on LinkedIn  ·  🍴 Fork and build on it
```

<br/>

<div align="center">

![separator](https://raw.githubusercontent.com/andreasbm/readme/master/assets/lines/colored.png)

*Built with ❤️ — demonstrating the complete ML lifecycle: Data → Model → API → UI → Cloud*

**[🚀 Try the Live App](https://huggingface.co/spaces/sajlendra/house-price-predictor-app)**

</div>
