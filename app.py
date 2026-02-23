"""
AI Fake Post & Sentiment Analysis System
A professional production-level AI web application built with Streamlit.
"""

import random
import time
import warnings

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.preprocessing import LabelEncoder

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────
# Page configuration
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="AI Fake Post & Sentiment Analysis",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────
# Custom CSS — dark theme + gradient cards
# ─────────────────────────────────────────────
CUSTOM_CSS = """
<style>
    /* ---- global background ---- */
    .stApp { background: #0d1117; color: #e6edf3; }

    /* ---- sidebar ---- */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #161b22 0%, #0d1117 100%);
        border-right: 1px solid #30363d;
    }
    [data-testid="stSidebar"] .stMarkdown h1,
    [data-testid="stSidebar"] .stMarkdown h2,
    [data-testid="stSidebar"] .stMarkdown h3 { color: #58a6ff; }

    /* ---- metric cards ---- */
    .metric-card {
        background: linear-gradient(135deg, #161b22 0%, #1c2128 100%);
        border: 1px solid #30363d;
        border-radius: 12px;
        padding: 20px;
        margin: 8px 0;
        box-shadow: 0 4px 15px rgba(0,0,0,0.3);
    }
    .metric-card h3 { color: #58a6ff; margin: 0 0 6px; font-size: 0.9rem; }
    .metric-card .value { font-size: 2rem; font-weight: 700; color: #e6edf3; }
    .metric-card .sub   { font-size: 0.8rem; color: #8b949e; }

    /* ---- result cards ---- */
    .result-fake {
        background: linear-gradient(135deg, #3d1c1c 0%, #2d1515 100%);
        border: 1px solid #f85149;
        border-radius: 12px; padding: 20px; margin: 8px 0;
    }
    .result-real {
        background: linear-gradient(135deg, #1c3d2a 0%, #15302a 100%);
        border: 1px solid #3fb950;
        border-radius: 12px; padding: 20px; margin: 8px 0;
    }
    .result-neutral {
        background: linear-gradient(135deg, #1c2a3d 0%, #152030 100%);
        border: 1px solid #58a6ff;
        border-radius: 12px; padding: 20px; margin: 8px 0;
    }

    /* ---- section title ---- */
    .section-title {
        font-size: 1.5rem; font-weight: 700; color: #58a6ff;
        border-bottom: 2px solid #21262d; padding-bottom: 8px; margin-bottom: 16px;
    }

    /* ---- hero banner ---- */
    .hero-banner {
        background: linear-gradient(135deg, #0d419d 0%, #1f6feb 50%, #388bfd 100%);
        border-radius: 16px; padding: 40px; text-align: center;
        margin-bottom: 24px;
        box-shadow: 0 8px 32px rgba(56,139,253,0.3);
    }
    .hero-banner h1 { font-size: 2.5rem; font-weight: 800; color: #fff; margin: 0; }
    .hero-banner p  { font-size: 1.1rem; color: #cae8ff; margin: 8px 0 0; }

    /* ---- badge ---- */
    .badge-fake     { background:#f85149; color:#fff; padding:4px 12px; border-radius:20px; font-weight:600; }
    .badge-real     { background:#3fb950; color:#fff; padding:4px 12px; border-radius:20px; font-weight:600; }
    .badge-positive { background:#3fb950; color:#fff; padding:4px 12px; border-radius:20px; font-weight:600; }
    .badge-negative { background:#f85149; color:#fff; padding:4px 12px; border-radius:20px; font-weight:600; }
    .badge-neutral  { background:#58a6ff; color:#fff; padding:4px 12px; border-radius:20px; font-weight:600; }
    .badge-high     { background:#f85149; color:#fff; padding:4px 12px; border-radius:20px; font-weight:600; }
    .badge-medium   { background:#e3b341; color:#000; padding:4px 12px; border-radius:20px; font-weight:600; }
    .badge-low      { background:#3fb950; color:#fff; padding:4px 12px; border-radius:20px; font-weight:600; }

    /* ---- text area ---- */
    .stTextArea textarea {
        background: #161b22 !important;
        color: #e6edf3 !important;
        border: 1px solid #30363d !important;
        border-radius: 8px !important;
    }

    /* ---- buttons ---- */
    .stButton>button {
        background: linear-gradient(135deg, #1f6feb, #388bfd) !important;
        color: #fff !important; border: none !important;
        border-radius: 8px !important; font-weight: 600 !important;
        padding: 8px 24px !important;
    }
    .stButton>button:hover {
        background: linear-gradient(135deg, #388bfd, #58a6ff) !important;
        box-shadow: 0 4px 15px rgba(56,139,253,0.4) !important;
    }

    /* ---- divider ---- */
    hr { border-color: #21262d !important; }

    /* ---- plotly chart background ---- */
    .js-plotly-plot { border-radius: 12px; overflow: hidden; }
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


# ─────────────────────────────────────────────
# Data generation helpers
# ─────────────────────────────────────────────
FAKE_POSTS = [
    "BREAKING: Scientists discover that drinking coffee reverses aging completely!",
    "Government secretly putting mind-control chips in COVID vaccines — insider reveals all",
    "This one weird trick will make you a millionaire in 30 days GUARANTEED",
    "Celebrity confirms they faked their own death — exclusive leaked video",
    "ALERT: 5G towers are causing mass bird deaths — SHARE before they delete this",
    "New study proves that the earth is actually flat — NASA admits cover-up",
    "FREE iPhone 15 giveaway — click link and claim yours NOW before it's too late",
    "Major bank collapse imminent — withdraw your money TODAY!",
    "Politicians planning to ban all cars by next month — sources inside government",
    "Miracle cure for cancer found but Big Pharma is HIDING it from the public",
    "URGENT: Tap water contains chemicals that lower IQ — scientists censored",
    "Global elite plans to reduce world population by 80% — leaked documents reveal",
    "Local doctor exposes dangerous secret your hospital doesn't want you to know",
    "Hidden camera footage proves moon landing was filmed in Hollywood studio",
    "New law will make home ownership illegal — government seizes all property",
]

REAL_POSTS = [
    "New climate report shows global temperatures have risen 1.2°C since pre-industrial times.",
    "The stock market closed slightly higher today after positive jobs data was released.",
    "Researchers at MIT have developed a new battery that charges in under two minutes.",
    "City council approved the new public transit expansion plan after months of debate.",
    "Local volunteers cleaned up over 500 kg of trash from the riverbank this weekend.",
    "A new study published in Nature suggests moderate exercise reduces heart disease risk.",
    "The central bank held interest rates steady amid ongoing inflation concerns.",
    "Tech company reports quarterly earnings above analyst expectations.",
    "UNESCO has added three new sites to the World Heritage List this year.",
    "Scientists successfully grew vegetables in simulated Mars soil in a lab environment.",
    "The annual marathon attracted over 10,000 runners from 40 different countries.",
    "New legislation aims to reduce plastic waste in oceans by 50% within a decade.",
    "Astronomers detected a new exoplanet in the habitable zone of a nearby star.",
    "Community garden project wins national award for urban sustainability innovation.",
    "Health authorities recommend updated flu vaccine formulation for the coming season.",
]

SENTIMENT_POSITIVE = [
    "This is absolutely wonderful news! I am so happy and excited about this!",
    "Amazing achievement! The team did an outstanding job — truly inspiring work.",
    "I love how positive this community is. Thank you for all your support!",
    "Best news I have heard all week! This is a great step forward for everyone.",
    "Incredibly proud of this milestone. The future looks bright and promising!",
]

SENTIMENT_NEGATIVE = [
    "This is a complete disaster. I am deeply disappointed and frustrated.",
    "Terrible decision by the leaders. This will hurt so many people — disgusting.",
    "I can't believe how bad this situation has become. Absolutely awful.",
    "Very sad to see this happening. Things keep getting worse every single day.",
    "Outrageous behavior! People should be ashamed of themselves for this.",
]

SENTIMENT_NEUTRAL = [
    "The report was released today and contains information about the current situation.",
    "The meeting took place at 3pm and various topics were discussed by attendees.",
    "According to the data, results varied across different regions of the country.",
    "The announcement was made on Tuesday following a review of existing policies.",
    "Officials stated that the process would continue as planned through next month.",
]


def generate_synthetic_dataset(n_samples: int = 2000) -> pd.DataFrame:
    """Generate a balanced synthetic dataset for training."""
    rng = np.random.default_rng(42)
    rows = []
    sentiments = ["Positive", "Negative", "Neutral"]
    sent_weights = [0.35, 0.35, 0.30]

    fake_templates = FAKE_POSTS * (n_samples // len(FAKE_POSTS) + 1)
    real_templates = REAL_POSTS * (n_samples // len(REAL_POSTS) + 1)
    pos_templates = SENTIMENT_POSITIVE * (n_samples // len(SENTIMENT_POSITIVE) + 1)
    neg_templates = SENTIMENT_NEGATIVE * (n_samples // len(SENTIMENT_NEGATIVE) + 1)
    neu_templates = SENTIMENT_NEUTRAL * (n_samples // len(SENTIMENT_NEUTRAL) + 1)

    noise_words = [
        "update", "news", "breaking", "report", "exclusive", "sources",
        "claim", "revealed", "confirmed", "official", "alleged", "shocking",
        "urgent", "secret", "truth", "hidden", "exposed", "leaked",
    ]

    for i in range(n_samples):
        fake_label = rng.integers(0, 2)
        sentiment = rng.choice(sentiments, p=sent_weights)

        if fake_label == 1:
            base = fake_templates[i % len(fake_templates)]
        else:
            base = real_templates[i % len(real_templates)]

        if sentiment == "Positive":
            extra = pos_templates[i % len(pos_templates)]
        elif sentiment == "Negative":
            extra = neg_templates[i % len(neg_templates)]
        else:
            extra = neu_templates[i % len(neu_templates)]

        # Add slight noise / variation
        n_noise = rng.integers(1, 4)
        chosen_noise = " ".join(rng.choice(noise_words, n_noise, replace=False))
        text = f"{base} {extra} {chosen_noise}".strip()

        rows.append({"text": text, "sentiment": sentiment, "fake_label": int(fake_label)})

    return pd.DataFrame(rows)


# ─────────────────────────────────────────────
# Model training / caching
# ─────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def train_models():
    """Train fake-detection and sentiment models; return them with metadata."""
    df = generate_synthetic_dataset(2000)

    # ── TF-IDF ──────────────────────────────
    tfidf = TfidfVectorizer(max_features=5000, ngram_range=(1, 2), stop_words="english")
    X = tfidf.fit_transform(df["text"])

    # ── Fake detection ──────────────────────
    X_train_f, X_test_f, y_train_f, y_test_f = train_test_split(
        X, df["fake_label"], test_size=0.2, random_state=42, stratify=df["fake_label"]
    )
    fake_model = LogisticRegression(max_iter=1000, random_state=42, C=1.0)
    fake_model.fit(X_train_f, y_train_f)
    fake_preds = fake_model.predict(X_test_f)
    fake_acc = accuracy_score(y_test_f, fake_preds)
    fake_report = classification_report(y_test_f, fake_preds, output_dict=True)

    # ── Sentiment analysis ──────────────────
    le = LabelEncoder()
    y_sent = le.fit_transform(df["sentiment"])
    X_train_s, X_test_s, y_train_s, y_test_s = train_test_split(
        X, y_sent, test_size=0.2, random_state=42, stratify=y_sent
    )
    sent_model = MultinomialNB()
    sent_model.fit(X_train_s, y_train_s)
    sent_preds = sent_model.predict(X_test_s)
    sent_acc = accuracy_score(y_test_s, sent_preds)
    sent_report = classification_report(y_test_s, sent_preds, output_dict=True)

    # ── Feature importance (top TF-IDF terms) ──
    feature_names = np.array(tfidf.get_feature_names_out())
    coef = fake_model.coef_[0]
    top_fake_idx = np.argsort(coef)[-20:][::-1]
    top_real_idx = np.argsort(coef)[:20]
    top_fake_terms = feature_names[top_fake_idx]
    top_real_terms = feature_names[top_real_idx]
    top_fake_scores = coef[top_fake_idx]
    top_real_scores = np.abs(coef[top_real_idx])

    return {
        "tfidf": tfidf,
        "fake_model": fake_model,
        "sent_model": sent_model,
        "label_encoder": le,
        "fake_accuracy": fake_acc,
        "sent_accuracy": sent_acc,
        "fake_report": fake_report,
        "sent_report": sent_report,
        "top_fake_terms": top_fake_terms,
        "top_real_terms": top_real_terms,
        "top_fake_scores": top_fake_scores,
        "top_real_scores": top_real_scores,
        "dataset": df,
    }


# ─────────────────────────────────────────────
# Prediction helper
# ─────────────────────────────────────────────
def predict(text: str, models: dict) -> dict:
    """Run fake detection + sentiment analysis on a single text."""
    vec = models["tfidf"].transform([text])

    fake_proba = models["fake_model"].predict_proba(vec)[0]
    fake_prob = float(fake_proba[1]) * 100
    real_prob = float(fake_proba[0]) * 100
    is_fake = fake_prob > 50

    sent_idx = models["sent_model"].predict(vec)[0]
    sentiment = models["label_encoder"].inverse_transform([sent_idx])[0]

    if fake_prob >= 70:
        risk = "High"
    elif fake_prob >= 40:
        risk = "Medium"
    else:
        risk = "Low"

    return {
        "fake_prob": fake_prob,
        "real_prob": real_prob,
        "is_fake": is_fake,
        "sentiment": sentiment,
        "risk": risk,
    }


# ─────────────────────────────────────────────
# Session state bootstrap
# ─────────────────────────────────────────────
if "history" not in st.session_state:
    st.session_state.history = []
if "live_feed" not in st.session_state:
    st.session_state.live_feed = []
if "live_running" not in st.session_state:
    st.session_state.live_running = False


# ─────────────────────────────────────────────
# Plotly chart helpers
# ─────────────────────────────────────────────
PLOTLY_LAYOUT = dict(
    paper_bgcolor="#161b22",
    plot_bgcolor="#161b22",
    font=dict(color="#e6edf3", family="Inter, sans-serif"),
    margin=dict(l=20, r=20, t=40, b=20),
)


def probability_bar_chart(fake_prob: float, real_prob: float) -> go.Figure:
    fig = go.Figure(
        go.Bar(
            x=["FAKE", "REAL"],
            y=[fake_prob, real_prob],
            marker_color=["#f85149", "#3fb950"],
            text=[f"{fake_prob:.1f}%", f"{real_prob:.1f}%"],
            textposition="outside",
            textfont=dict(size=14, color="#e6edf3"),
        )
    )
    fig.update_layout(
        title="Fake vs Real Probability",
        yaxis=dict(range=[0, 115], title="Probability (%)", gridcolor="#21262d"),
        xaxis=dict(title="Classification"),
        showlegend=False,
        **PLOTLY_LAYOUT,
    )
    return fig


def sentiment_pie_chart(sentiment: str) -> go.Figure:
    if sentiment == "Positive":
        values = [75, 15, 10]
    elif sentiment == "Negative":
        values = [10, 75, 15]
    else:
        values = [20, 20, 60]

    fig = go.Figure(
        go.Pie(
            labels=["Positive", "Negative", "Neutral"],
            values=values,
            hole=0.45,
            marker=dict(colors=["#3fb950", "#f85149", "#58a6ff"]),
            textfont=dict(color="#e6edf3"),
        )
    )
    fig.update_layout(
        title="Sentiment Distribution",
        legend=dict(font=dict(color="#e6edf3")),
        **PLOTLY_LAYOUT,
    )
    return fig


def gauge_chart(fake_prob: float) -> go.Figure:
    if fake_prob >= 70:
        bar_color = "#f85149"
    elif fake_prob >= 40:
        bar_color = "#e3b341"
    else:
        bar_color = "#3fb950"

    fig = go.Figure(
        go.Indicator(
            mode="gauge+number+delta",
            value=fake_prob,
            title={"text": "Fake Risk Score", "font": {"color": "#e6edf3"}},
            number={"suffix": "%", "font": {"color": "#e6edf3"}},
            gauge={
                "axis": {"range": [0, 100], "tickcolor": "#8b949e"},
                "bar": {"color": bar_color},
                "bgcolor": "#21262d",
                "bordercolor": "#30363d",
                "steps": [
                    {"range": [0, 40], "color": "#1c3d2a"},
                    {"range": [40, 70], "color": "#3d2e1c"},
                    {"range": [70, 100], "color": "#3d1c1c"},
                ],
                "threshold": {
                    "line": {"color": "#ffffff", "width": 3},
                    "thickness": 0.75,
                    "value": fake_prob,
                },
            },
        )
    )
    fig.update_layout(**PLOTLY_LAYOUT)
    return fig


# ─────────────────────────────────────────────
# Sidebar navigation
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown(
        """
        <div style='text-align:center;padding:16px 0 8px'>
            <span style='font-size:2.5rem'>🔍</span>
            <h2 style='color:#58a6ff;margin:4px 0'>AI Analysis</h2>
            <p style='color:#8b949e;font-size:0.8rem;margin:0'>Fake Post & Sentiment</p>
        </div>
        <hr style='border-color:#21262d'>
        """,
        unsafe_allow_html=True,
    )

    page = st.radio(
        "Navigate",
        ["🏠 Home", "🔬 Analyze Post", "📡 Live Feed", "📊 Analytics", "🧠 Model Insights"],
        label_visibility="collapsed",
    )

    st.markdown("<hr style='border-color:#21262d'>", unsafe_allow_html=True)
    st.markdown(
        """
        <div style='padding:8px 0'>
            <p style='color:#8b949e;font-size:0.75rem;text-align:center'>
                Powered by<br>
                <span style='color:#58a6ff;font-weight:600'>Scikit-learn · Streamlit · Plotly</span>
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

# Load models (cached after first run)
with st.spinner("Loading AI models…"):
    models = train_models()


# ─────────────────────────────────────────────
# PAGE 1 — Home
# ─────────────────────────────────────────────
if page == "🏠 Home":
    st.markdown(
        """
        <div class='hero-banner'>
            <h1>🔍 AI Fake Post & Sentiment Analysis</h1>
            <p>Detect misinformation and analyze emotional tone in social media posts using Machine Learning</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    col1, col2, col3, col4 = st.columns(4)
    total = len(st.session_state.history)
    fake_count = sum(1 for h in st.session_state.history if h["is_fake"])
    real_count = total - fake_count
    pos_count = sum(1 for h in st.session_state.history if h["sentiment"] == "Positive")

    with col1:
        st.markdown(
            f"<div class='metric-card'><h3>📋 Posts Analyzed</h3>"
            f"<div class='value'>{total}</div>"
            f"<div class='sub'>Total since session start</div></div>",
            unsafe_allow_html=True,
        )
    with col2:
        st.markdown(
            f"<div class='metric-card'><h3>🚨 Fake Detected</h3>"
            f"<div class='value' style='color:#f85149'>{fake_count}</div>"
            f"<div class='sub'>Potentially misleading posts</div></div>",
            unsafe_allow_html=True,
        )
    with col3:
        st.markdown(
            f"<div class='metric-card'><h3>✅ Real Detected</h3>"
            f"<div class='value' style='color:#3fb950'>{real_count}</div>"
            f"<div class='sub'>Credible posts</div></div>",
            unsafe_allow_html=True,
        )
    with col4:
        st.markdown(
            f"<div class='metric-card'><h3>😊 Positive Sentiment</h3>"
            f"<div class='value' style='color:#58a6ff'>{pos_count}</div>"
            f"<div class='sub'>Posts with positive tone</div></div>",
            unsafe_allow_html=True,
        )

    st.markdown("<br>", unsafe_allow_html=True)

    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown("<div class='section-title'>🚀 Key Features</div>", unsafe_allow_html=True)
        features = [
            ("🤖", "ML-Powered Detection", "Logistic Regression classifies posts as fake or real with high accuracy"),
            ("💬", "Sentiment Analysis", "Multinomial Naive Bayes identifies positive, negative, and neutral tone"),
            ("📊", "Rich Visualizations", "Interactive Plotly charts including gauges, bars, and pie charts"),
            ("📡", "Live Simulation Feed", "Automated real-time post analysis simulation"),
            ("🧠", "Model Insights", "Explore feature importance and TF-IDF keyword analysis"),
        ]
        for icon, title, desc in features:
            st.markdown(
                f"""
                <div class='metric-card' style='padding:14px'>
                    <h3>{icon} {title}</h3>
                    <div class='sub' style='font-size:0.85rem'>{desc}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    with col_b:
        st.markdown("<div class='section-title'>📈 Tech Stack</div>", unsafe_allow_html=True)
        tech = {
            "Framework": "Streamlit",
            "ML Library": "Scikit-learn",
            "Data": "Pandas · NumPy",
            "Visualisation": "Plotly",
            "Fake Model": "Logistic Regression",
            "Sentiment Model": "Multinomial Naive Bayes",
            "Vectorizer": "TF-IDF (5 000 features, 1–2 grams)",
            "Serialization": "Joblib",
        }
        for k, v in tech.items():
            st.markdown(
                f"""
                <div style='display:flex;justify-content:space-between;padding:10px 14px;
                            border-bottom:1px solid #21262d'>
                    <span style='color:#8b949e'>{k}</span>
                    <span style='color:#58a6ff;font-weight:600'>{v}</span>
                </div>
                """,
                unsafe_allow_html=True,
            )

        acc_fake = f"{models['fake_accuracy']*100:.1f}%"
        acc_sent = f"{models['sent_accuracy']*100:.1f}%"
        st.markdown(
            f"""
            <div class='metric-card' style='margin-top:16px'>
                <h3>🎯 Model Accuracy</h3>
                <div style='display:flex;gap:24px;margin-top:8px'>
                    <div>
                        <div style='color:#8b949e;font-size:0.8rem'>Fake Detection</div>
                        <div style='font-size:1.5rem;font-weight:700;color:#3fb950'>{acc_fake}</div>
                    </div>
                    <div>
                        <div style='color:#8b949e;font-size:0.8rem'>Sentiment</div>
                        <div style='font-size:1.5rem;font-weight:700;color:#58a6ff'>{acc_sent}</div>
                    </div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )


# ─────────────────────────────────────────────
# PAGE 2 — Analyze Post
# ─────────────────────────────────────────────
elif page == "🔬 Analyze Post":
    st.markdown("<div class='section-title'>🔬 Analyze Post</div>", unsafe_allow_html=True)
    st.markdown(
        "<p style='color:#8b949e'>Paste any social media post below and click <strong>Analyze</strong> "
        "to detect if it is fake and identify its sentiment.</p>",
        unsafe_allow_html=True,
    )

    sample_texts = [
        "Select a sample post…",
        "BREAKING: Scientists discover that drinking coffee reverses aging completely!",
        "New climate report shows global temperatures have risen 1.2°C since pre-industrial times.",
        "This is absolutely wonderful news! I am so happy and excited about this!",
        "Government secretly putting mind-control chips in COVID vaccines — insider reveals all",
        "The stock market closed slightly higher today after positive jobs data was released.",
    ]

    selected_sample = st.selectbox("📋 Try a sample post", sample_texts)
    default_text = "" if selected_sample == sample_texts[0] else selected_sample

    user_text = st.text_area(
        "✏️ Enter post text",
        value=default_text,
        height=150,
        placeholder="Paste or type a social media post here…",
    )

    analyze_btn = st.button("🔍 Analyze Post", use_container_width=True)

    if analyze_btn:
        if not user_text.strip():
            st.warning("⚠️ Please enter some text before analyzing.")
        else:
            with st.spinner("Analyzing post…"):
                result = predict(user_text, models)
                st.session_state.history.append(
                    {
                        "text": user_text[:80] + ("…" if len(user_text) > 80 else ""),
                        "is_fake": result["is_fake"],
                        "sentiment": result["sentiment"],
                        "risk": result["risk"],
                        "fake_prob": result["fake_prob"],
                    }
                )

            st.markdown("<hr>", unsafe_allow_html=True)
            st.markdown("<div class='section-title'>📊 Analysis Results</div>", unsafe_allow_html=True)

            # ── Summary cards ──
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                verdict = "FAKE" if result["is_fake"] else "REAL"
                badge_cls = "badge-fake" if result["is_fake"] else "badge-real"
                st.markdown(
                    f"<div class='metric-card'><h3>🏷️ Verdict</h3>"
                    f"<div class='value'><span class='{badge_cls}'>{verdict}</span></div></div>",
                    unsafe_allow_html=True,
                )
            with col2:
                st.markdown(
                    f"<div class='metric-card'><h3>🚨 Fake Probability</h3>"
                    f"<div class='value' style='color:#f85149'>{result['fake_prob']:.1f}%</div></div>",
                    unsafe_allow_html=True,
                )
            with col3:
                sent = result["sentiment"]
                sent_cls = f"badge-{sent.lower()}"
                st.markdown(
                    f"<div class='metric-card'><h3>💬 Sentiment</h3>"
                    f"<div class='value'><span class='{sent_cls}'>{sent}</span></div></div>",
                    unsafe_allow_html=True,
                )
            with col4:
                risk = result["risk"]
                risk_cls = f"badge-{risk.lower()}"
                st.markdown(
                    f"<div class='metric-card'><h3>⚠️ Risk Level</h3>"
                    f"<div class='value'><span class='{risk_cls}'>{risk}</span></div></div>",
                    unsafe_allow_html=True,
                )

            # ── Charts ──
            st.markdown("<br>", unsafe_allow_html=True)
            col_l, col_m, col_r = st.columns(3)
            with col_l:
                st.plotly_chart(
                    probability_bar_chart(result["fake_prob"], result["real_prob"]),
                    use_container_width=True,
                )
            with col_m:
                st.plotly_chart(
                    sentiment_pie_chart(result["sentiment"]),
                    use_container_width=True,
                )
            with col_r:
                st.plotly_chart(
                    gauge_chart(result["fake_prob"]),
                    use_container_width=True,
                )

            # ── Analyzed text ──
            card_cls = "result-fake" if result["is_fake"] else "result-real"
            icon = "🚨" if result["is_fake"] else "✅"
            st.markdown(
                f"<div class='{card_cls}'>"
                f"<strong>{icon} Analyzed Text:</strong><br>"
                f"<span style='color:#c9d1d9'>{user_text}</span>"
                f"</div>",
                unsafe_allow_html=True,
            )


# ─────────────────────────────────────────────
# PAGE 3 — Live Feed
# ─────────────────────────────────────────────
elif page == "📡 Live Feed":
    st.markdown("<div class='section-title'>📡 Live Detection Feed</div>", unsafe_allow_html=True)
    st.markdown(
        "<p style='color:#8b949e'>Simulated real-time social media post analysis. "
        "Press <strong>Start</strong> to begin automatic detection.</p>",
        unsafe_allow_html=True,
    )

    col_btn1, col_btn2, col_btn3 = st.columns([1, 1, 4])
    with col_btn1:
        start_btn = st.button("▶ Start Feed", use_container_width=True)
    with col_btn2:
        clear_btn = st.button("🗑 Clear Feed", use_container_width=True)

    if clear_btn:
        st.session_state.live_feed = []
        st.rerun()

    feed_placeholder = st.empty()
    stats_placeholder = st.empty()

    all_posts = FAKE_POSTS + REAL_POSTS + SENTIMENT_POSITIVE + SENTIMENT_NEGATIVE + SENTIMENT_NEUTRAL

    if start_btn:
        for batch in range(10):
            post = random.choice(all_posts)
            result = predict(post, models)
            st.session_state.live_feed.insert(
                0,
                {
                    "post": post,
                    "is_fake": result["is_fake"],
                    "sentiment": result["sentiment"],
                    "risk": result["risk"],
                    "fake_prob": result["fake_prob"],
                },
            )
            # Keep only latest 30
            st.session_state.live_feed = st.session_state.live_feed[:30]

            with feed_placeholder.container():
                for entry in st.session_state.live_feed[:8]:
                    verdict = "FAKE 🚨" if entry["is_fake"] else "REAL ✅"
                    card_cls = "result-fake" if entry["is_fake"] else "result-real"
                    sent = entry["sentiment"]
                    sent_cls = f"badge-{sent.lower()}"
                    risk = entry["risk"]
                    risk_cls = f"badge-{risk.lower()}"
                    st.markdown(
                        f"""
                        <div class='{card_cls}' style='margin-bottom:8px'>
                            <div style='display:flex;justify-content:space-between;align-items:center'>
                                <strong style='color:#e6edf3'>{verdict}</strong>
                                <span>
                                    <span class='{sent_cls}'>{sent}</span>&nbsp;
                                    <span class='{risk_cls}'>Risk: {risk}</span>&nbsp;
                                    <span style='color:#8b949e;font-size:0.8rem'>
                                        Fake: {entry['fake_prob']:.1f}%
                                    </span>
                                </span>
                            </div>
                            <div style='color:#8b949e;font-size:0.85rem;margin-top:6px'>
                                {entry['post'][:120]}…
                            </div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

            time.sleep(0.5)

    # Static display when feed is populated but not running
    if not start_btn and st.session_state.live_feed:
        with feed_placeholder.container():
            for entry in st.session_state.live_feed[:8]:
                verdict = "FAKE 🚨" if entry["is_fake"] else "REAL ✅"
                card_cls = "result-fake" if entry["is_fake"] else "result-real"
                sent = entry["sentiment"]
                sent_cls = f"badge-{sent.lower()}"
                risk = entry["risk"]
                risk_cls = f"badge-{risk.lower()}"
                st.markdown(
                    f"""
                    <div class='{card_cls}' style='margin-bottom:8px'>
                        <div style='display:flex;justify-content:space-between;align-items:center'>
                            <strong style='color:#e6edf3'>{verdict}</strong>
                            <span>
                                <span class='{sent_cls}'>{sent}</span>&nbsp;
                                <span class='{risk_cls}'>Risk: {risk}</span>&nbsp;
                                <span style='color:#8b949e;font-size:0.8rem'>
                                    Fake: {entry['fake_prob']:.1f}%
                                </span>
                            </span>
                        </div>
                        <div style='color:#8b949e;font-size:0.85rem;margin-top:6px'>
                            {entry['post'][:120]}…
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

    if not st.session_state.live_feed:
        st.markdown(
            "<div class='result-neutral' style='text-align:center;padding:40px'>"
            "<span style='font-size:2rem'>📡</span><br>"
            "<span style='color:#8b949e'>Press <strong>Start Feed</strong> to begin live simulation</span>"
            "</div>",
            unsafe_allow_html=True,
        )


# ─────────────────────────────────────────────
# PAGE 4 — Analytics Dashboard
# ─────────────────────────────────────────────
elif page == "📊 Analytics":
    st.markdown("<div class='section-title'>📊 Analytics Dashboard</div>", unsafe_allow_html=True)

    history = st.session_state.history
    live = st.session_state.live_feed
    all_records = [
        {
            "source": "Manual",
            "is_fake": h["is_fake"],
            "sentiment": h["sentiment"],
            "risk": h["risk"],
            "fake_prob": h["fake_prob"],
        }
        for h in history
    ] + [
        {
            "source": "Live Feed",
            "is_fake": l["is_fake"],
            "sentiment": l["sentiment"],
            "risk": l["risk"],
            "fake_prob": l["fake_prob"],
        }
        for l in live
    ]

    total = len(all_records)
    fake_count = sum(1 for r in all_records if r["is_fake"])
    real_count = total - fake_count
    sent_counts = {"Positive": 0, "Negative": 0, "Neutral": 0}
    for r in all_records:
        sent_counts[r["sentiment"]] = sent_counts.get(r["sentiment"], 0) + 1

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown(
            f"<div class='metric-card'><h3>📋 Total Posts</h3>"
            f"<div class='value'>{total}</div>"
            f"<div class='sub'>Across all sessions</div></div>",
            unsafe_allow_html=True,
        )
    with col2:
        fake_pct = (fake_count / total * 100) if total else 0
        st.markdown(
            f"<div class='metric-card'><h3>🚨 Fake Posts</h3>"
            f"<div class='value' style='color:#f85149'>{fake_count}</div>"
            f"<div class='sub'>{fake_pct:.1f}% of total</div></div>",
            unsafe_allow_html=True,
        )
    with col3:
        real_pct = (real_count / total * 100) if total else 0
        st.markdown(
            f"<div class='metric-card'><h3>✅ Real Posts</h3>"
            f"<div class='value' style='color:#3fb950'>{real_count}</div>"
            f"<div class='sub'>{real_pct:.1f}% of total</div></div>",
            unsafe_allow_html=True,
        )
    with col4:
        pos_pct = (sent_counts["Positive"] / total * 100) if total else 0
        st.markdown(
            f"<div class='metric-card'><h3>😊 Positive Tone</h3>"
            f"<div class='value' style='color:#58a6ff'>{sent_counts['Positive']}</div>"
            f"<div class='sub'>{pos_pct:.1f}% of total</div></div>",
            unsafe_allow_html=True,
        )

    st.markdown("<br>", unsafe_allow_html=True)

    if total > 0:
        col_left, col_right = st.columns(2)

        with col_left:
            # Fake vs Real bar
            fig_fvr = go.Figure(
                go.Bar(
                    x=["Fake", "Real"],
                    y=[fake_count, real_count],
                    marker_color=["#f85149", "#3fb950"],
                    text=[fake_count, real_count],
                    textposition="outside",
                    textfont=dict(color="#e6edf3"),
                )
            )
            fig_fvr.update_layout(
                title="Fake vs Real Count",
                yaxis=dict(gridcolor="#21262d"),
                showlegend=False,
                **PLOTLY_LAYOUT,
            )
            st.plotly_chart(fig_fvr, use_container_width=True)

        with col_right:
            # Sentiment distribution pie
            fig_sent = go.Figure(
                go.Pie(
                    labels=list(sent_counts.keys()),
                    values=list(sent_counts.values()),
                    hole=0.4,
                    marker=dict(colors=["#3fb950", "#f85149", "#58a6ff"]),
                    textfont=dict(color="#e6edf3"),
                )
            )
            fig_sent.update_layout(
                title="Sentiment Distribution",
                legend=dict(font=dict(color="#e6edf3")),
                **PLOTLY_LAYOUT,
            )
            st.plotly_chart(fig_sent, use_container_width=True)

        # Risk level distribution
        risk_counts = {"Low": 0, "Medium": 0, "High": 0}
        for r in all_records:
            risk_counts[r["risk"]] = risk_counts.get(r["risk"], 0) + 1

        fig_risk = go.Figure(
            go.Bar(
                x=list(risk_counts.keys()),
                y=list(risk_counts.values()),
                marker_color=["#3fb950", "#e3b341", "#f85149"],
                text=list(risk_counts.values()),
                textposition="outside",
                textfont=dict(color="#e6edf3"),
            )
        )
        fig_risk.update_layout(
            title="Risk Level Distribution",
            yaxis=dict(gridcolor="#21262d"),
            showlegend=False,
            **PLOTLY_LAYOUT,
        )
        st.plotly_chart(fig_risk, use_container_width=True)

        # Trend / cumulative posts over time
        if len(all_records) >= 2:
            trend_df = pd.DataFrame(
                {"Index": range(1, total + 1), "Cumulative Posts": range(1, total + 1)}
            )
            fig_trend = px.line(
                trend_df,
                x="Index",
                y="Cumulative Posts",
                title="Cumulative Posts Analyzed",
                color_discrete_sequence=["#58a6ff"],
            )
            fig_trend.update_layout(**PLOTLY_LAYOUT)
            fig_trend.update_traces(line=dict(width=2))
            st.plotly_chart(fig_trend, use_container_width=True)

        # Recent history table
        st.markdown("<div class='section-title'>📋 Recent Analysis History</div>", unsafe_allow_html=True)
        table_data = [
            {
                "Text (truncated)": r.get("text", "—")[:60] if "text" in r else "Live feed post",
                "Verdict": "🚨 Fake" if r["is_fake"] else "✅ Real",
                "Sentiment": r["sentiment"],
                "Risk": r["risk"],
                "Fake %": f"{r['fake_prob']:.1f}%",
            }
            for r in (history + live)[:20]
        ]
        if table_data:
            st.dataframe(
                pd.DataFrame(table_data),
                use_container_width=True,
                hide_index=True,
            )
    else:
        st.markdown(
            "<div class='result-neutral' style='text-align:center;padding:60px'>"
            "<span style='font-size:3rem'>📊</span><br><br>"
            "<span style='color:#8b949e;font-size:1.1rem'>No data yet. "
            "Go to <strong>Analyze Post</strong> or <strong>Live Feed</strong> to generate analytics.</span>"
            "</div>",
            unsafe_allow_html=True,
        )


# ─────────────────────────────────────────────
# PAGE 5 — Model Insights
# ─────────────────────────────────────────────
elif page == "🧠 Model Insights":
    st.markdown("<div class='section-title'>🧠 Model Insights</div>", unsafe_allow_html=True)

    # ── Accuracy cards ──
    col1, col2 = st.columns(2)
    with col1:
        acc_fake = models["fake_accuracy"] * 100
        st.markdown(
            f"<div class='metric-card'>"
            f"<h3>🤖 Fake Detection Model (Logistic Regression)</h3>"
            f"<div class='value' style='color:#3fb950'>{acc_fake:.2f}%</div>"
            f"<div class='sub'>Test set accuracy</div>"
            f"</div>",
            unsafe_allow_html=True,
        )
    with col2:
        acc_sent = models["sent_accuracy"] * 100
        st.markdown(
            f"<div class='metric-card'>"
            f"<h3>💬 Sentiment Model (Multinomial Naive Bayes)</h3>"
            f"<div class='value' style='color:#58a6ff'>{acc_sent:.2f}%</div>"
            f"<div class='sub'>Test set accuracy</div>"
            f"</div>",
            unsafe_allow_html=True,
        )

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Classification reports ──
    col_l, col_r = st.columns(2)
    with col_l:
        st.markdown("<div class='section-title'>📋 Fake Detection Report</div>", unsafe_allow_html=True)
        report = models["fake_report"]
        rows = []
        for label_key, label_name in [("0", "Real"), ("1", "Fake")]:
            if label_key in report:
                m = report[label_key]
                rows.append({
                    "Class": label_name,
                    "Precision": f"{m['precision']:.3f}",
                    "Recall": f"{m['recall']:.3f}",
                    "F1-Score": f"{m['f1-score']:.3f}",
                    "Support": int(m["support"]),
                })
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    with col_r:
        st.markdown("<div class='section-title'>📋 Sentiment Report</div>", unsafe_allow_html=True)
        report_s = models["sent_report"]
        le = models["label_encoder"]
        rows_s = []
        for i, cls_name in enumerate(le.classes_):
            key = str(i)
            if key in report_s:
                m = report_s[key]
                rows_s.append({
                    "Class": cls_name,
                    "Precision": f"{m['precision']:.3f}",
                    "Recall": f"{m['recall']:.3f}",
                    "F1-Score": f"{m['f1-score']:.3f}",
                    "Support": int(m["support"]),
                })
        st.dataframe(pd.DataFrame(rows_s), use_container_width=True, hide_index=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Feature importance ──
    st.markdown("<div class='section-title'>🔑 Feature Importance (TF-IDF Keywords)</div>", unsafe_allow_html=True)
    col_fi1, col_fi2 = st.columns(2)

    with col_fi1:
        fig_fake_terms = go.Figure(
            go.Bar(
                x=models["top_fake_scores"].tolist(),
                y=models["top_fake_terms"].tolist(),
                orientation="h",
                marker_color="#f85149",
                text=[f"{s:.3f}" for s in models["top_fake_scores"]],
                textposition="outside",
                textfont=dict(color="#e6edf3"),
            )
        )
        fig_fake_terms.update_layout(
            title="Top Keywords → FAKE",
            yaxis=dict(autorange="reversed"),
            xaxis=dict(title="Coefficient", gridcolor="#21262d"),
            **PLOTLY_LAYOUT,
        )
        st.plotly_chart(fig_fake_terms, use_container_width=True)

    with col_fi2:
        fig_real_terms = go.Figure(
            go.Bar(
                x=models["top_real_scores"].tolist(),
                y=models["top_real_terms"].tolist(),
                orientation="h",
                marker_color="#3fb950",
                text=[f"{s:.3f}" for s in models["top_real_scores"]],
                textposition="outside",
                textfont=dict(color="#e6edf3"),
            )
        )
        fig_real_terms.update_layout(
            title="Top Keywords → REAL",
            yaxis=dict(autorange="reversed"),
            xaxis=dict(title="|Coefficient|", gridcolor="#21262d"),
            **PLOTLY_LAYOUT,
        )
        st.plotly_chart(fig_real_terms, use_container_width=True)

    # ── TF-IDF vocabulary size ──
    vocab_size = len(models["tfidf"].vocabulary_)
    st.markdown(
        f"<div class='metric-card' style='margin-top:16px'>"
        f"<h3>📚 TF-IDF Vectorizer Stats</h3>"
        f"<div style='display:flex;gap:40px;margin-top:8px'>"
        f"<div><div style='color:#8b949e;font-size:0.8rem'>Vocabulary Size</div>"
        f"    <div style='font-size:1.5rem;font-weight:700;color:#58a6ff'>{vocab_size:,}</div></div>"
        f"<div><div style='color:#8b949e;font-size:0.8rem'>Max Features</div>"
        f"    <div style='font-size:1.5rem;font-weight:700;color:#58a6ff'>5,000</div></div>"
        f"<div><div style='color:#8b949e;font-size:0.8rem'>N-gram Range</div>"
        f"    <div style='font-size:1.5rem;font-weight:700;color:#58a6ff'>(1, 2)</div></div>"
        f"<div><div style='color:#8b949e;font-size:0.8rem'>Stop Words</div>"
        f"    <div style='font-size:1.5rem;font-weight:700;color:#58a6ff'>English</div></div>"
        f"</div></div>",
        unsafe_allow_html=True,
    )
