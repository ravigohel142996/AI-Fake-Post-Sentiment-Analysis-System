# 🔍 AI Fake Post & Sentiment Analysis System

A **production-level AI web application** that detects whether a social media post is **FAKE or REAL** and analyses its **sentiment** (Positive / Negative / Neutral) using Machine Learning.

---

## 🚀 Features

| Feature | Description |
|---------|-------------|
| 🤖 Fake Detection | Logistic Regression classifies posts as Fake or Real |
| 💬 Sentiment Analysis | Multinomial Naive Bayes identifies tone (Positive / Negative / Neutral) |
| 📊 Rich Visualizations | Interactive Plotly gauge, bar, and pie charts |
| 📡 Live Simulation Feed | Automated real-time post analysis simulation |
| 🧠 Model Insights | Feature importance & TF-IDF keyword analysis |
| 📋 Analytics Dashboard | Cumulative stats, trend graphs, history table |

---

## 🖥️ Pages

1. **Home** — Session stats and system overview
2. **Analyze Post** — Paste any text and get instant predictions
3. **Live Feed** — Simulated real-time detection stream
4. **Analytics** — Charts and history across all analyzed posts
5. **Model Insights** — Accuracy, classification reports, and top TF-IDF keywords

---

## 🛠️ Tech Stack

- **Framework:** Streamlit
- **ML Library:** Scikit-learn
- **Fake Detection Model:** Logistic Regression
- **Sentiment Model:** Multinomial Naive Bayes
- **Vectorizer:** TF-IDF (5 000 features, 1–2 grams)
- **Data:** Pandas · NumPy (synthetic dataset generation)
- **Visualisation:** Plotly
- **Serialization:** Joblib

---

## ⚡ Quick Start

```bash
# 1. Clone the repository
git clone https://github.com/ravigohel142996/AI-Fake-Post-Sentiment-Analysis-System.git
cd AI-Fake-Post-Sentiment-Analysis-System

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run the app
streamlit run app.py
```

The app will open at **http://localhost:8501** in your browser.

---

## 📦 Requirements

```
streamlit>=1.32.0
scikit-learn>=1.4.0
pandas>=2.1.0
numpy>=1.26.0
plotly>=5.18.0
joblib>=1.3.0
```

---

## 🧠 How It Works

1. **Synthetic Dataset** — 2 000 labelled posts are generated at startup (text, fake_label, sentiment).
2. **TF-IDF Vectorizer** — Converts raw text into numerical feature vectors (max 5 000 features, 1–2 grams).
3. **Fake Detection** — A Logistic Regression model trained on the 80/20 split predicts FAKE vs REAL probabilities.
4. **Sentiment Analysis** — A Multinomial Naive Bayes model predicts Positive / Negative / Neutral tone.
5. **Risk Level** — Derived from the fake probability: Low (<40 %), Medium (40–70 %), High (>70 %).

---

## 📂 Project Structure

```
AI-Fake-Post-Sentiment-Analysis-System/
├── app.py            # Main Streamlit application
├── requirements.txt  # Python dependencies
└── README.md         # Project documentation
```

---

## 📸 UI Overview

- **Dark theme** with gradient cards and professional typography
- **Sidebar navigation** for seamless page switching
- **Plotly charts** — probability bars, sentiment pies, risk gauges
- **Enterprise-grade look** suitable for demos and production deployment

---

## 📄 License

MIT — feel free to use and adapt this project.
