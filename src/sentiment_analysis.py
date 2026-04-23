"""
============================================================
SENTIMENT ANALYSIS — Restaurant Reviews, Majhitar, Sikkim
Text Analytics and NLP Assignment
============================================================

Pipeline:
    1. Load real Google Reviews dataset
    2. Preprocessing & EDA
    3. Feature Extraction (TF-IDF)
    4. Multiple ML Models (Logistic Regression, Naive Bayes,
       SVM, Random Forest, Gradient Boosting)
    5. Comprehensive Evaluation (Accuracy, Precision, Recall,
       F1-Score, ROC-AUC, Confusion Matrix, Cross-Validation)
    6. Visualizations
    7. VADER Lexicon-based Sentiment Comparison
    8. Context-Aware Restaurant Recommendation (NEW)
    9. Prediction on new reviews
   10. Save best model
"""

# ─────────────────────────────────────────────────────────
# 0. INSTALL / IMPORTS
# ─────────────────────────────────────────────────────────
import os
import re
import warnings
import pickle

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns

import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.sentiment.vader import SentimentIntensityAnalyzer

from sklearn.model_selection import (
    train_test_split, cross_val_score, StratifiedKFold
)
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix,
    roc_auc_score, roc_curve, auc, ConfusionMatrixDisplay
)
from sklearn.preprocessing import label_binarize

from wordcloud import WordCloud

warnings.filterwarnings("ignore")

# Download required NLTK data
for pkg in ["stopwords", "wordnet", "vader_lexicon", "omw-1.4"]:
    nltk.download(pkg, quiet=True)

# ─────────────────────────────────────────────────────────
# PLOTTING STYLE
# ─────────────────────────────────────────────────────────
plt.rcParams.update({
    "figure.dpi": 130,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "font.family": "DejaVu Sans",
})
COLORS = {"positive": "#2ecc71", "neutral": "#f39c12", "negative": "#e74c3c"}
OUTPUT_DIR = "output_plots"
os.makedirs(OUTPUT_DIR, exist_ok=True)

def save(fig, name):
    fig.savefig(os.path.join(OUTPUT_DIR, name), bbox_inches="tight")
    plt.show()
    plt.close(fig)


# ─────────────────────────────────────────────────────────
# 1. LOAD DATASET
# ─────────────────────────────────────────────────────────
print("=" * 60)
print("STEP 1: LOADING DATASET")
print("=" * 60)

# Build/load the dataset using dataset_builder
from dataset_builder import build_dataset
data = build_dataset(include_scenarios=True)

print(f"\nShape: {data.shape}")
print(f"Columns: {list(data.columns)}")
print("\nFirst 5 rows:")
print(data.head())

# ─────────────────────────────────────────────────────────
# 2. SENTIMENT LABELING
# ─────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 2: SENTIMENT LABELING")
print("=" * 60)

def sentiment_label(rating):
    if rating >= 4:
        return "positive"
    elif rating == 3:
        return "neutral"
    else:
        return "negative"

data["sentiment"] = data["rating"].apply(sentiment_label)

label_counts = data["sentiment"].value_counts()
print("\nSentiment Distribution:")
print(label_counts)

# Plot 1: Sentiment distribution
fig, axes = plt.subplots(1, 2, figsize=(12, 4))

axes[0].bar(label_counts.index, label_counts.values,
            color=[COLORS[s] for s in label_counts.index], edgecolor="white", width=0.5)
axes[0].set_title("Overall Sentiment Distribution", fontsize=13, fontweight="bold")
axes[0].set_xlabel("Sentiment")
axes[0].set_ylabel("Number of Reviews")
for i, v in enumerate(label_counts.values):
    axes[0].text(i, v + 0.5, str(v), ha="center", fontweight="bold")

axes[1].pie(label_counts.values, labels=label_counts.index,
            colors=[COLORS[s] for s in label_counts.index],
            autopct="%1.1f%%", startangle=140, textprops={"fontsize": 11})
axes[1].set_title("Sentiment Share (%)", fontsize=13, fontweight="bold")

fig.suptitle("Restaurant Reviews — Sentiment Overview", fontsize=14, fontweight="bold")
plt.tight_layout()
save(fig, "1_sentiment_distribution.png")

# ─────────────────────────────────────────────────────────
# 3. EXPLORATORY DATA ANALYSIS
# ─────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 3: EXPLORATORY DATA ANALYSIS")
print("=" * 60)

data["review_length"] = data["review"].apply(lambda x: len(x.split()))
data["char_count"] = data["review"].apply(len)

print("\nReview Length Statistics (words):")
print(data.groupby("sentiment")["review_length"].describe().round(2))

# Plot 2: Restaurant-wise sentiment stacked bar
fig, ax = plt.subplots(figsize=(13, 5))
ct = pd.crosstab(data["restaurant"], data["sentiment"])
for col in ["positive", "neutral", "negative"]:
    if col not in ct.columns:
        ct[col] = 0
ct = ct[["positive", "neutral", "negative"]]

ct.plot(kind="bar", stacked=True, ax=ax,
        color=[COLORS[c] for c in ct.columns], edgecolor="white")
ax.set_title("Sentiment Comparison by Restaurant", fontsize=13, fontweight="bold")
ax.set_xlabel("Restaurant")
ax.set_ylabel("Number of Reviews")
ax.legend(title="Sentiment", loc="upper right")
plt.xticks(rotation=35, ha="right")
plt.tight_layout()
save(fig, "2_restaurant_sentiment_comparison.png")

# Plot 3: Average rating per restaurant
fig, ax = plt.subplots(figsize=(11, 4))
avg_rating = data.groupby("restaurant")["rating"].mean().sort_values(ascending=False)
bars = ax.barh(avg_rating.index, avg_rating.values,
               color=["#2ecc71" if r >= 4 else "#f39c12" if r >= 3 else "#e74c3c"
                      for r in avg_rating.values])
ax.set_xlim(0, 5.5)
ax.axvline(4, color="gray", linestyle="--", alpha=0.6, label="Positive threshold")
ax.axvline(3, color="gray", linestyle=":", alpha=0.6, label="Neutral threshold")
ax.set_title("Average Google Rating per Restaurant", fontsize=13, fontweight="bold")
ax.set_xlabel("Average Rating (out of 5)")
ax.legend()
for bar, val in zip(bars, avg_rating.values):
    ax.text(val + 0.05, bar.get_y() + bar.get_height() / 2,
            f"{val:.2f}", va="center", fontweight="bold")
plt.tight_layout()
save(fig, "3_avg_rating_per_restaurant.png")

# Plot 4: Review length distribution
fig, ax = plt.subplots(figsize=(9, 4))
for sentiment, color in COLORS.items():
    subset = data[data["sentiment"] == sentiment]["review_length"]
    ax.hist(subset, bins=20, alpha=0.6, label=sentiment, color=color, edgecolor="white")
ax.set_title("Review Length Distribution by Sentiment", fontsize=13, fontweight="bold")
ax.set_xlabel("Number of Words in Review")
ax.set_ylabel("Frequency")
ax.legend()
plt.tight_layout()
save(fig, "4_review_length_distribution.png")

# ─────────────────────────────────────────────────────────
# 4. TEXT PREPROCESSING
# ─────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 4: TEXT PREPROCESSING")
print("=" * 60)

stop_words = set(stopwords.words("english"))
lemmatizer = WordNetLemmatizer()

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"http\S+|www\S+", " ", text)   # remove URLs
    text = re.sub(r"[^a-zA-Z\s]", " ", text)       # remove non-alpha
    text = re.sub(r"\s+", " ", text).strip()        # collapse spaces
    words = text.split()
    words = [lemmatizer.lemmatize(w) for w in words if w not in stop_words and len(w) > 2]
    return " ".join(words)

data["clean_review"] = data["review"].apply(clean_text)

print("Sample cleaned reviews:")
for _, row in data[["review", "clean_review", "sentiment"]].head(5).iterrows():
    print(f"  Original : {row['review'][:80]}...")
    print(f"  Cleaned  : {row['clean_review'][:80]}...")
    print(f"  Sentiment: {row['sentiment']}\n")

# Plot 5: Word Clouds per sentiment
fig, axes = plt.subplots(1, 3, figsize=(16, 4))
for ax, sentiment in zip(axes, ["positive", "neutral", "negative"]):
    text = " ".join(data[data["sentiment"] == sentiment]["clean_review"])
    wc = WordCloud(
        width=500, height=300,
        background_color="white",
        colormap="Greens" if sentiment == "positive" else
                 "Oranges" if sentiment == "neutral" else "Reds",
        max_words=80
    ).generate(text if text.strip() else "no reviews available")
    ax.imshow(wc, interpolation="bilinear")
    ax.axis("off")
    ax.set_title(f"{sentiment.capitalize()} Reviews", fontsize=12, fontweight="bold",
                 color=COLORS[sentiment])
fig.suptitle("Word Clouds by Sentiment", fontsize=14, fontweight="bold")
plt.tight_layout()
save(fig, "5_wordclouds.png")

# ─────────────────────────────────────────────────────────
# 5. FEATURE EXTRACTION (TF-IDF)
# ─────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 5: FEATURE EXTRACTION — TF-IDF")
print("=" * 60)

vectorizer = TfidfVectorizer(
    max_features=2000,
    ngram_range=(1, 2),    # unigrams + bigrams
    min_df=2,
    sublinear_tf=True
)

X = vectorizer.fit_transform(data["clean_review"])
y = data["sentiment"]

print(f"TF-IDF matrix shape: {X.shape}")
print(f"Number of classes  : {y.nunique()} — {list(y.unique())}")

# ─────────────────────────────────────────────────────────
# 6. TRAIN / TEST SPLIT
# ─────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 6: TRAIN / TEST SPLIT (80/20)")
print("=" * 60)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"Training samples : {X_train.shape[0]}")
print(f"Testing  samples : {X_test.shape[0]}")

# ─────────────────────────────────────────────────────────
# 7. TRAIN MULTIPLE MODELS
# ─────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 7: TRAINING 5 MODELS")
print("=" * 60)

models = {
    "Logistic Regression": LogisticRegression(max_iter=1000, C=1.0, random_state=42),
    "Naive Bayes":         MultinomialNB(alpha=0.5),
    "Linear SVM":          LinearSVC(max_iter=2000, C=1.0, random_state=42),
    "Random Forest":       RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1),
    "Gradient Boosting":   GradientBoostingClassifier(n_estimators=150, random_state=42),
}

results = {}
trained_models = {}

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

for name, clf in models.items():
    print(f"\n[*] Training: {name}")
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    cv_scores = cross_val_score(clf, X, y, cv=cv, scoring="accuracy")
    report = classification_report(y_test, y_pred, output_dict=True)

    results[name] = {
        "model": clf,
        "predictions": y_pred,
        "accuracy": acc,
        "cv_mean": cv_scores.mean(),
        "cv_std": cv_scores.std(),
        "report": report,
    }
    trained_models[name] = clf

    print(f"    Test Accuracy   : {acc:.4f}")
    print(f"    CV Accuracy     : {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

# ─────────────────────────────────────────────────────────
# 8. MODEL COMPARISON & EVALUATION
# ─────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 8: MODEL COMPARISON")
print("=" * 60)

summary = pd.DataFrame({
    "Model": list(results.keys()),
    "Test Accuracy": [v["accuracy"] for v in results.values()],
    "CV Mean Accuracy": [v["cv_mean"] for v in results.values()],
    "CV Std": [v["cv_std"] for v in results.values()],
}).set_index("Model").sort_values("Test Accuracy", ascending=False)

print("\nModel Performance Summary:")
print(summary.to_string())

# Plot 6: Model accuracy comparison
fig, ax = plt.subplots(figsize=(10, 5))
x = np.arange(len(summary))
w = 0.35
ax.bar(x - w/2, summary["Test Accuracy"], width=w, label="Test Accuracy",
       color="#3498db", edgecolor="white")
ax.bar(x + w/2, summary["CV Mean Accuracy"], width=w, label="CV Mean Accuracy",
       color="#9b59b6", edgecolor="white", yerr=summary["CV Std"], capsize=4)
ax.set_xticks(x)
ax.set_xticklabels(summary.index, rotation=15, ha="right")
ax.set_ylim(0, 1.1)
ax.set_ylabel("Accuracy")
ax.set_title("Model Performance Comparison", fontsize=13, fontweight="bold")
ax.legend()
for i, (ta, cva) in enumerate(zip(summary["Test Accuracy"], summary["CV Mean Accuracy"])):
    ax.text(i - w/2, ta + 0.02, f"{ta:.3f}", ha="center", fontsize=8, fontweight="bold")
    ax.text(i + w/2, cva + 0.02, f"{cva:.3f}", ha="center", fontsize=8, fontweight="bold")
plt.tight_layout()
save(fig, "6_model_comparison.png")

print("\nDetailed Classification Reports:")
print("─" * 60)
for name, res in results.items():
    print(f"\n>>> {name}")
    print(classification_report(y_test, res["predictions"]))

# ─────────────────────────────────────────────────────────
# 9. CONFUSION MATRICES (all models)
# ─────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 9: CONFUSION MATRICES")
print("=" * 60)

fig, axes = plt.subplots(2, 3, figsize=(16, 9))
axes = axes.flatten()
labels_order = sorted(y.unique())

for i, (name, res) in enumerate(results.items()):
    cm = confusion_matrix(y_test, res["predictions"], labels=labels_order)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels_order)
    disp.plot(ax=axes[i], colorbar=False, cmap="Blues")
    axes[i].set_title(f"{name}\n(Acc={res['accuracy']:.3f})", fontweight="bold")

axes[-1].set_visible(False)
fig.suptitle("Confusion Matrices — All Models", fontsize=14, fontweight="bold")
plt.tight_layout()
save(fig, "7_confusion_matrices.png")

# ─────────────────────────────────────────────────────────
# 10. PRECISION / RECALL / F1 PER CLASS (best model)
# ─────────────────────────────────────────────────────────
best_model_name = summary["Test Accuracy"].idxmax()
best_res = results[best_model_name]
best_clf = best_res["model"]
print(f"\nBest Model: {best_model_name} (Accuracy = {best_res['accuracy']:.4f})")

report_df = pd.DataFrame(best_res["report"]).T
report_df = report_df.loc[["negative", "neutral", "positive"]]

fig, axes = plt.subplots(1, 3, figsize=(13, 4))
metrics = ["precision", "recall", "f1-score"]
metric_labels = ["Precision", "Recall", "F1-Score"]

for ax, metric, label in zip(axes, metrics, metric_labels):
    vals = report_df[metric].values
    bar_colors = [COLORS[c] for c in report_df.index]
    ax.bar(report_df.index, vals, color=bar_colors, edgecolor="white", width=0.5)
    ax.set_ylim(0, 1.1)
    ax.set_title(label, fontsize=12, fontweight="bold")
    ax.set_ylabel("Score")
    for j, v in enumerate(vals):
        ax.text(j, v + 0.02, f"{v:.3f}", ha="center", fontsize=9, fontweight="bold")

fig.suptitle(f"Per-Class Metrics — {best_model_name}", fontsize=13, fontweight="bold")
plt.tight_layout()
save(fig, "8_per_class_metrics.png")

# ─────────────────────────────────────────────────────────
# 11. ROC-AUC CURVES (Logistic Regression — supports proba)
# ─────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 10: ROC-AUC CURVES (Logistic Regression)")
print("=" * 60)

lr_clf = trained_models["Logistic Regression"]
classes = lr_clf.classes_
y_test_bin = label_binarize(y_test, classes=classes)
y_score = lr_clf.predict_proba(X_test)

fig, ax = plt.subplots(figsize=(7, 5))
roc_colors = {"negative": "#e74c3c", "neutral": "#f39c12", "positive": "#2ecc71"}

for i, cls in enumerate(classes):
    fpr, tpr, _ = roc_curve(y_test_bin[:, i], y_score[:, i])
    roc_auc_val = auc(fpr, tpr)
    ax.plot(fpr, tpr, label=f"{cls} (AUC = {roc_auc_val:.2f})",
            color=roc_colors.get(cls, "blue"), linewidth=2)

ax.plot([0, 1], [0, 1], "k--", linewidth=1, label="Random Classifier")
ax.set_xlim([0, 1])
ax.set_ylim([0, 1.02])
ax.set_xlabel("False Positive Rate")
ax.set_ylabel("True Positive Rate")
ax.set_title("ROC Curves (One-vs-Rest) — Logistic Regression", fontsize=12, fontweight="bold")
ax.legend(loc="lower right")
plt.tight_layout()
save(fig, "9_roc_curves.png")

macro_auc = roc_auc_score(y_test_bin, y_score, multi_class="ovr", average="macro")
print(f"Macro-average ROC-AUC (Logistic Regression): {macro_auc:.4f}")

# ─────────────────────────────────────────────────────────
# 12. CROSS-VALIDATION DETAILS
# ─────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 11: CROSS-VALIDATION (5-Fold Stratified)")
print("=" * 60)

fig, ax = plt.subplots(figsize=(10, 4))
cv_data = {name: cross_val_score(clf, X, y, cv=cv, scoring="accuracy")
           for name, clf in models.items()}

ax.boxplot(cv_data.values(), labels=cv_data.keys(), patch_artist=True,
           boxprops=dict(facecolor="#3498db", color="#2980b9"),
           medianprops=dict(color="white", linewidth=2))
ax.set_ylabel("Accuracy")
ax.set_title("5-Fold Cross-Validation Accuracy Distribution", fontsize=12, fontweight="bold")
plt.xticks(rotation=15, ha="right")
ax.set_ylim(0, 1.1)
plt.tight_layout()
save(fig, "10_cross_validation_boxplot.png")

for name, scores in cv_data.items():
    print(f"  {name:<25} mean={scores.mean():.4f}  std={scores.std():.4f}  folds={np.round(scores,3)}")

# ─────────────────────────────────────────────────────────
# 13. VADER LEXICON SENTIMENT COMPARISON
# ─────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 12: VADER LEXICON-BASED COMPARISON")
print("=" * 60)

sia = SentimentIntensityAnalyzer()

def vader_sentiment(text):
    score = sia.polarity_scores(str(text))["compound"]
    if score >= 0.05:
        return "positive"
    elif score <= -0.05:
        return "negative"
    else:
        return "neutral"

data["vader_sentiment"] = data["review"].apply(vader_sentiment)
vader_acc = accuracy_score(data["sentiment"], data["vader_sentiment"])
print(f"VADER Agreement with Rating-based Labels: {vader_acc:.4f}")

fig, axes = plt.subplots(1, 2, figsize=(12, 4))
data["vader_sentiment"].value_counts().plot(kind="bar", ax=axes[0],
    color=[COLORS[s] for s in data["vader_sentiment"].value_counts().index], edgecolor="white")
axes[0].set_title("VADER Sentiment Distribution", fontweight="bold")
axes[0].set_xlabel("Sentiment")
axes[0].set_ylabel("Count")

vader_counts = data["vader_sentiment"].value_counts()
ml_counts = data["sentiment"].value_counts()
x = np.arange(3)
labels_ord = ["positive", "neutral", "negative"]
axes[1].bar(x - 0.2, [ml_counts.get(l, 0) for l in labels_ord], 0.35,
            label="Rating-based Labels", color="#3498db", edgecolor="white")
axes[1].bar(x + 0.2, [vader_counts.get(l, 0) for l in labels_ord], 0.35,
            label="VADER", color="#e67e22", edgecolor="white")
axes[1].set_xticks(x)
axes[1].set_xticklabels(labels_ord)
axes[1].set_title("Rating Labels vs VADER Comparison", fontweight="bold")
axes[1].legend()
fig.suptitle("VADER Lexicon Sentiment Analysis", fontsize=13, fontweight="bold")
plt.tight_layout()
save(fig, "11_vader_comparison.png")

# ─────────────────────────────────────────────────────────
# 14. TOP TFIDF FEATURES PER CLASS
# ─────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 13: TOP TF-IDF FEATURES PER SENTIMENT CLASS")
print("=" * 60)

lr_model = trained_models["Logistic Regression"]
feature_names = vectorizer.get_feature_names_out()

fig, axes = plt.subplots(1, 3, figsize=(15, 5))
for ax, (i, cls) in zip(axes, enumerate(lr_model.classes_)):
    coefs = lr_model.coef_[i]
    top_idx = np.argsort(coefs)[-15:]
    top_features = feature_names[top_idx]
    top_coefs = coefs[top_idx]
    ax.barh(top_features, top_coefs, color=COLORS[cls], edgecolor="white")
    ax.set_title(f"Top Features: {cls.capitalize()}", fontweight="bold", color=COLORS[cls])
    ax.set_xlabel("Coefficient (LR)")

fig.suptitle("Most Influential Words per Sentiment — Logistic Regression",
             fontsize=13, fontweight="bold")
plt.tight_layout()
save(fig, "12_top_features_per_class.png")

# ─────────────────────────────────────────────────────────
# 15. CONTEXT-AWARE RESTAURANT RECOMMENDATION (NEW)
# ─────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 14: CONTEXT-AWARE RESTAURANT RECOMMENDATION")
print("=" * 60)
print("""
This module demonstrates how sentiment analysis can go beyond
simple classification and power a practical recommendation system.

When a customer expresses a food craving or dining intent in
natural language, we:
  1. Detect the expressed sentiment of the query
  2. Match keywords to food-category profiles of each restaurant
  3. Filter to restaurants with predominantly positive reviews
  4. Rank and recommend the best match
""")

# ── Food-category keyword profiles per restaurant ──────────
RESTAURANT_PROFILES = {
    "Coffee Break": {
        "keywords": ["coffee", "cafe", "pizza", "cheesy", "cheese", "dessert", "sweet",
                     "cake", "brownie", "lava cake", "cold coffee", "cappuccino",
                     "tiramisu", "mousse", "mac and cheese", "cheesecake", "filter coffee",
                     "pastry", "snack", "cozy cafe", "quiet", "work", "study"],
        "description": "Best for: coffee, cheesy pizza, desserts, cozy cafe experience"
    },
    "Grill and Chill Majhitar": {
        "keywords": ["grill", "grilled", "bbq", "barbecue", "chicken", "pork", "ribs",
                     "kebab", "tikka", "seekh", "meat", "smoked", "charcoal", "fish",
                     "non-veg", "non vegetarian", "platter", "outdoor", "evening"],
        "description": "Best for: grilled meat, BBQ platters, outdoor evening dining"
    },
    "Cozy Corner Majitar": {
        "keywords": ["sikkimese", "local", "gundruk", "chhurpi", "thali", "veg",
                     "vegetarian", "affordable", "home cooked", "dal", "authentic",
                     "budget", "simple", "traditional", "local cuisine"],
        "description": "Best for: authentic Sikkimese food, vegetarian thali, budget meals"
    },
    "Hotel Majitar Retreat Restaurant": {
        "keywords": ["river", "teesta", "view", "scenic", "romantic", "hotel",
                     "fine dining", "continental", "buffet", "breakfast", "special occasion",
                     "anniversary", "upscale", "professional", "thukpa", "momos"],
        "description": "Best for: scenic river view dining, special occasions, hotel buffet"
    },
    "Sangay Restaurant Majitar": {
        "keywords": ["sel roti", "kinema", "tibetan", "noodles", "home style", "family",
                     "local", "filling", "affordable", "traditional", "sikkimese",
                     "hearty", "generous portion", "authentic"],
        "description": "Best for: traditional Sikkimese/Tibetan food, family meals"
    },
    "Hotel Temi Restaurant Majitar": {
        "keywords": ["thali", "quick", "budget", "road trip", "passing", "traveler",
                     "reliable", "momos", "noodle soup", "butter tea", "tibetan bread",
                     "affordable", "consistent", "daily meal"],
        "description": "Best for: quick affordable meals, budget travelers, daily dining"
    },
    "The Riverside Dhaba Majitar": {
        "keywords": ["dhaba", "riverside", "chai", "tea", "pakora", "paratha", "curd",
                     "river", "teesta", "rustic", "roadside", "fish", "rajma", "casual",
                     "snack", "quick stop", "road trip", "aloo"],
        "description": "Best for: chai, riverside snacks, roadside dhaba experience"
    },
}

def recommend_restaurant(customer_query: str, data: pd.DataFrame, top_n: int = 3) -> list:
    """
    Given a customer's natural language query (e.g. 'I want something
    cheesy and a good dessert'), recommend the best matching restaurants
    using keyword profile matching combined with sentiment filtering.

    Parameters
    ----------
    customer_query : str
        The customer's expressed food craving or dining intent.
    data : pd.DataFrame
        The reviews DataFrame with 'restaurant' and 'sentiment' columns.
    top_n : int
        Number of top recommendations to return.

    Returns
    -------
    list of dict with restaurant name, match score, avg rating, sentiment ratio.
    """
    query_lower = customer_query.lower()
    query_words = set(re.sub(r"[^a-z\s]", " ", query_lower).split())

    # Compute sentiment ratio per restaurant from actual reviews
    restaurant_stats = {}
    for rest, group in data.groupby("restaurant"):
        total = len(group)
        pos = (group["sentiment"] == "positive").sum()
        avg_rating = group["rating"].mean()
        restaurant_stats[rest] = {
            "total_reviews": total,
            "positive_ratio": pos / total if total > 0 else 0,
            "avg_rating": avg_rating,
        }

    scores = []
    for rest, profile in RESTAURANT_PROFILES.items():
        # Keyword match score: how many profile keywords appear in query
        profile_words = set(" ".join(profile["keywords"]).lower().split())
        match_score = len(query_words & profile_words)

        # Weighted score = keyword match × positive ratio × avg rating
        stats = restaurant_stats.get(rest, {"positive_ratio": 0, "avg_rating": 0, "total_reviews": 0})
        weighted = match_score * (1 + stats["positive_ratio"]) * (stats["avg_rating"] / 5)

        scores.append({
            "restaurant": rest,
            "keyword_matches": match_score,
            "weighted_score": round(weighted, 3),
            "avg_rating": round(stats["avg_rating"], 2),
            "positive_ratio": round(stats["positive_ratio"] * 100, 1),
            "description": profile["description"],
        })

    # Sort by weighted score descending, then by avg_rating
    scores.sort(key=lambda x: (x["weighted_score"], x["avg_rating"]), reverse=True)
    return scores[:top_n]


# ── Demo: Scenario-based recommendations ──────────────────
DEMO_QUERIES = [
    {
        "scenario": "Cheesy food and dessert craving",
        "query": "I am really craving something cheesy right now, maybe a pizza and a nice dessert to finish",
    },
    {
        "scenario": "BBQ and grilled meat evening",
        "query": "Our group wants a proper BBQ dinner with grilled chicken, ribs and kebabs tonight",
    },
    {
        "scenario": "Authentic Sikkimese vegetarian food",
        "query": "We are a vegetarian family and want to try authentic local Sikkimese food like gundruk and chhurpi",
    },
    {
        "scenario": "Scenic romantic dinner",
        "query": "Looking for a romantic dinner with a nice river view and good quality food for a special occasion",
    },
    {
        "scenario": "Quick budget meal for travelers",
        "query": "Just passing through on a road trip and need a quick affordable thali meal",
    },
    {
        "scenario": "Chai and snacks by the river",
        "query": "Want to sit by the Teesta river and have some chai and light snacks in a rustic dhaba setting",
    },
]

print("\n" + "─" * 60)
print("CONTEXT-AWARE RECOMMENDATION DEMO")
print("─" * 60)

for demo in DEMO_QUERIES:
    print(f"\n{'─'*55}")
    print(f"  Scenario : {demo['scenario']}")
    print(f"  Customer : \"{demo['query']}\"")
    print(f"  Top Recommendations:")
    recs = recommend_restaurant(demo["query"], data, top_n=3)
    for rank, rec in enumerate(recs, 1):
        print(f"    {rank}. {rec['restaurant']}")
        print(f"       → {rec['description']}")
        print(f"       Avg Rating: {rec['avg_rating']}/5  |  "
              f"Positive Reviews: {rec['positive_ratio']}%  |  "
              f"Match Score: {rec['weighted_score']}")

print(f"\n{'─'*55}")
print("\n[INFO] Recommendation logic:")
print("  weighted_score = keyword_matches × (1 + positive_ratio) × (avg_rating / 5)")
print("  Only restaurants with positive review majority are ranked first.")

# ─────────────────────────────────────────────────────────
# 16. FINAL EVALUATION SUMMARY TABLE
# ─────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 15: FINAL EVALUATION SUMMARY")
print("=" * 60)

final_rows = []
for name, res in results.items():
    rep = res["report"]
    final_rows.append({
        "Model": name,
        "Accuracy": f"{res['accuracy']:.4f}",
        "CV Accuracy": f"{res['cv_mean']:.4f} ± {res['cv_std']:.4f}",
        "Macro Precision": f"{rep['macro avg']['precision']:.4f}",
        "Macro Recall": f"{rep['macro avg']['recall']:.4f}",
        "Macro F1": f"{rep['macro avg']['f1-score']:.4f}",
        "Weighted F1": f"{rep['weighted avg']['f1-score']:.4f}",
    })

final_df = pd.DataFrame(final_rows).set_index("Model")
print(final_df.to_string())

# ─────────────────────────────────────────────────────────
# 17. PREDICT NEW REVIEWS
# ─────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 16: PREDICTING NEW CUSTOM REVIEWS")
print("=" * 60)

new_reviews = [
    "The food was amazing and the service was excellent. Highly recommend!",
    "Completely disappointed. Cold food, rude staff and overpriced.",
    "It was okay. Nothing special but nothing terrible either.",
    "Best momo I have ever had in Sikkim! Will definitely return.",
    "The hygiene was poor and the place smelled bad.",
    "Decent place for lunch. Reasonable prices and okay quality.",
    # New scenario reviews (cheesy/dessert context)
    "I was craving something cheesy and the cheese burst pizza here hit the spot perfectly.",
    "The lava cake and cheesecake were both incredible. Best desserts in Majitar.",
]

print(f"\nUsing best model: {best_model_name}")
print()
for rev in new_reviews:
    cleaned = clean_text(rev)
    vec = vectorizer.transform([cleaned])
    pred = best_clf.predict(vec)[0]
    print(f"  Review   : {rev[:75]}...")
    print(f"  Predicted: {pred.upper()}\n")

# ─────────────────────────────────────────────────────────
# 18. SAVE BEST MODEL
# ─────────────────────────────────────────────────────────
print("=" * 60)
print("STEP 17: SAVING BEST MODEL")
print("=" * 60)

save_payload = {
    "model": best_clf,
    "vectorizer": vectorizer,
    "model_name": best_model_name,
    "accuracy": best_res["accuracy"],
    "classes": list(lr_model.classes_),
    "restaurant_profiles": RESTAURANT_PROFILES,
}

with open("sentiment_model.pkl", "wb") as f:
    pickle.dump(save_payload, f)

print(f"\n[+] Saved: sentiment_model.pkl")
print(f"    Model  : {best_model_name}")
print(f"    Accuracy: {best_res['accuracy']:.4f}")
print(f"\n[+] All plots saved to: {OUTPUT_DIR}/")
print("\n" + "=" * 60)
print("ANALYSIS COMPLETE")
print("=" * 60)
