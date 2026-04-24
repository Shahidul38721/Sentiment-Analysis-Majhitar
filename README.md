# Sentiment Analysis of Restaurant Reviews in Majhitar, Sikkim

## Project Overview

This project performs **Sentiment Analysis** on restaurant reviews collected from restaurants located in **Majhitar (Majitar), Sikkim**.

The goal is to automatically classify customer reviews into **positive, neutral, or negative sentiment** using **Natural Language Processing (NLP) and Machine Learning techniques**, and to use that sentiment data to power a **context-aware restaurant recommendation system**.

The system analyzes textual customer feedback, helps understand overall customer satisfaction for different restaurants, and recommends the most suitable restaurant based on a customer's expressed food craving or dining intent.

---

## Problem Statement

Restaurants receive large numbers of customer reviews online. Manually analyzing these reviews to understand customer satisfaction is time-consuming and inefficient.

This project aims to build a **machine learning model that automatically analyzes restaurant reviews, determines their sentiment, and uses the derived insights to recommend the right restaurant** based on what a customer is looking for.

**Example use case:** A customer says *"I am really craving something cheesy right now, maybe a pizza and a nice dessert to finish."* The system analyzes this intent, matches it against restaurant profiles built from real review sentiment, and recommends **Coffee Break** — which has the highest concentration of positive reviews about cheesy food and desserts in Majitar.

---

## Dataset Description

The dataset contains restaurant reviews collected from **7 restaurants in Majhitar, Sikkim**.

Each entry in the dataset contains:

| Column | Description |
|--------|-------------|
| restaurant | Name of the restaurant |
| review | Customer review text |
| rating | Rating given by the customer (1–5) |

### Restaurants Covered

| Restaurant | Specialty |
|------------|-----------|
| Coffee Break | Coffee, cheesy pizza, desserts, cozy cafe |
| Grill and Chill Majhitar | BBQ, grilled meat, outdoor dining |
| Cozy Corner Majitar | Authentic Sikkimese food, vegetarian thali |
| Hotel Majitar Retreat Restaurant | Scenic river-view dining, hotel restaurant |
| Sangay Restaurant Majitar | Traditional Sikkimese/Tibetan cuisine |
| Hotel Temi Restaurant Majitar | Quick affordable meals, budget travelers |
| The Riverside Dhaba Majitar | Chai, dhaba food, riverside snacks |

### Sample Dataset

| restaurant | review | rating |
|------------|--------|--------|
| Coffee Break | Nice coffee and peaceful environment | 4 |
| Grill and Chill Majhitar | Very tasty grilled food and BBQ | 5 |
| Cozy Corner Majitar | Food quality poor | 2 |
| Coffee Break | Had their cheese burst pizza and it was absolutely divine | 5 |

---

## Sentiment Labeling

Rating values are converted into sentiment categories:

| Rating | Sentiment |
|--------|-----------|
| 4 – 5 | Positive |
| 3 | Neutral |
| 1 – 2 | Negative |

---

## Sentiment Distribution Results

From the analysis of the dataset:

| Sentiment | Count | Share |
|-----------|-------|-------|
| Positive | 51+ | ~60% |
| Negative | 20+ | ~23.5% |
| Neutral | 14+ | ~16.5% |

The dataset is **majority positive**, reflecting that most customers who visit and review local Majhitar restaurants are generally satisfied.

---

## Technologies Used

| Category | Technology |
|----------|-----------|
| Language | Python 3 |
| Data Handling | Pandas, NumPy |
| NLP / Text Processing | NLTK, TF-IDF (scikit-learn), VADER |
| Machine Learning | scikit-learn |
| Visualization | Matplotlib, Seaborn, WordCloud |
| Dataset Collection | Bundled real-style reviews / Outscraper API (optional) |

---

## Project Structure

```
sentiment-analysis-majhitar/
│
├── dataset_builder.py           # Dataset builder with real reviews + scenario reviews
├── sentiment_analysis.py        # Full NLP pipeline + recommendation engine
├── majitar_restaurant_reviews.csv  # Generated dataset (created on first run)
├── sentiment_model.pkl          # Saved best model (created on first run)
│
├── output_plots/                # All generated visualization files
│   ├── 1_sentiment_distribution.png
│   ├── 2_restaurant_sentiment_comparison.png
│   ├── 3_avg_rating_per_restaurant.png
│   ├── 4_review_length_distribution.png
│   ├── 5_wordclouds.png
│   ├── 6_model_comparison.png
│   ├── 7_confusion_matrices.png
│   ├── 8_per_class_metrics.png
│   ├── 9_roc_curves.png
│   ├── 10_cross_validation_boxplot.png
│   ├── 11_vader_comparison.png
│   └── 12_top_features_per_class.png
│
└── README.md
```

---

## Project Workflow

```
Dataset Builder
     │
     ▼
Real Reviews + Scenario Reviews
     │
     ▼
Data Cleaning & Preprocessing
     │
     ▼
Sentiment Labeling (Rating → Positive/Neutral/Negative)
     │
     ▼
TF-IDF Feature Extraction
     │
     ▼
Train 5 ML Models
     │
     ▼
Evaluate (Accuracy, F1, ROC-AUC, Cross-Validation)
     │
     ▼
VADER Lexicon Comparison
     │
     ▼
Context-Aware Restaurant Recommendation  ← NEW
     │
     ▼
Predict New Reviews + Save Model
```

---

## Step-by-Step Explanation

### Step 1: Loading the Dataset

The dataset is built by `dataset_builder.py` which provides two options — fetching real reviews via the Outscraper API, or using the bundled real-style review dataset (no API key needed).

```python
from dataset_builder import build_dataset
data = build_dataset(include_scenarios=True)
```

The `include_scenarios=True` flag appends a set of **scenario-based reviews** that simulate real customer intents such as craving cheesy pizza, wanting authentic Sikkimese food, or looking for a riverside dining experience.

---

### Step 2: Data Preprocessing

Text data is cleaned before being used for machine learning:

**Lowercase conversion** — all text is lowercased for consistency.

**Special character removal** — punctuation and symbols are removed using regular expressions.

**Stopword removal** — common English words (`the`, `is`, `and`, etc.) are removed using the NLTK stopwords library.

**Lemmatization** — words are reduced to their base form (`running` → `run`, `tasty` → `tasty`) using WordNetLemmatizer.

Example:

```
Original : "The food was AMAZING!!! Service was great."
Cleaned  : "food amazing service great"
```

---

### Step 3: Feature Extraction — TF-IDF

Machine learning models require numerical input. Text is converted to numerical vectors using **TF-IDF (Term Frequency — Inverse Document Frequency)**.

TF-IDF measures how important a word is in a document relative to the entire dataset. Words that appear frequently in one review but rarely across all reviews get a higher weight.

Configuration used:
- `max_features = 2000`
- `ngram_range = (1, 2)` — unigrams and bigrams
- `min_df = 2` — minimum document frequency
- `sublinear_tf = True` — log normalization

---

### Step 4: Splitting the Dataset

| Split | Percentage |
|-------|-----------|
| Training Data | 80% |
| Testing Data | 20% |

Stratified splitting ensures balanced class distribution in both splits.

---

### Step 5: Model Training

Five machine learning models are trained and compared:

| Model | Description |
|-------|-------------|
| Logistic Regression | Efficient linear classifier, good baseline for text |
| Naive Bayes (Multinomial) | Fast probabilistic model, strong with TF-IDF |
| Linear SVM | High-performing linear classifier for text |
| Random Forest | Ensemble of decision trees, handles non-linearity |
| Gradient Boosting | Sequential ensemble, strong for small datasets |

---

### Step 6: Model Evaluation

Each model is evaluated using:

**Accuracy** — percentage of correctly classified reviews.

**Precision, Recall, F1-Score** — per-class metrics for negative, neutral, and positive sentiment.

**Confusion Matrix** — shows exact classification breakdown for each class.

**ROC-AUC Curves** — plotted for Logistic Regression using One-vs-Rest strategy.

**5-Fold Stratified Cross-Validation** — measures generalization with variance analysis.

#### Sample Results (Linear SVM — Best Performer)

| Class | Precision | Recall | F1-Score |
|-------|-----------|--------|----------|
| Negative | 0.500 | 0.250 | 0.333 |
| Neutral | 1.000 | 0.333 | 0.500 |
| Positive | 0.714 | 1.000 | 0.833 |

Test Accuracy: **0.706** | CV Mean: **0.729**

> Note: Lower performance on negative/neutral classes is expected due to the smaller number of negative and neutral reviews relative to positive ones in the dataset.

---

### Step 7: VADER Lexicon Comparison

**VADER (Valence Aware Dictionary and sEntiment Reasoner)** is a rule-based sentiment analysis tool that assigns a compound sentiment score to text without any training.

It is used here as a **baseline comparison** against the trained ML models to show the difference between lexicon-based and ML-based approaches.

The VADER distribution closely matches the rating-based labels, validating that our sentiment labeling is consistent with how the text actually reads.

---

### Step 8: Context-Aware Restaurant Recommendation *(New Feature)*

This is the most practically useful component of the project. It demonstrates how sentiment analysis can directly power a recommendation engine.

#### How It Works

When a customer expresses a food craving or dining intent in natural language, the system:

1. Parses the customer's query into keywords
2. Matches those keywords against **food-category profiles** built for each restaurant
3. Weights the match score by the restaurant's **positive review ratio** and **average rating**
4. Returns the top ranked restaurant recommendations

#### Recommendation Formula

```
weighted_score = keyword_matches × (1 + positive_ratio) × (avg_rating / 5)
```

#### Example Scenarios

**Scenario 1: Cheesy food and dessert craving**

> Customer: *"I am really craving something cheesy right now, maybe a pizza and a nice dessert to finish."*

Recommended: **Coffee Break** ← cheesy pizza, cheese burst pizza, desserts, lava cake, cheesecake

**Scenario 2: BBQ / grilled meat dinner**

> Customer: *"Our group wants a proper BBQ dinner with grilled chicken, ribs and kebabs tonight."*

Recommended: **Grill and Chill Majhitar** ← BBQ, grilled chicken, pork ribs, seekh kebab

**Scenario 3: Authentic Sikkimese vegetarian food**

> Customer: *"We are a vegetarian family and want to try authentic local Sikkimese food like gundruk and chhurpi."*

Recommended: **Cozy Corner Majitar** ← gundruk soup, chhurpi, veg thali, authentic local food

**Scenario 4: Scenic romantic dinner**

> Customer: *"Looking for a romantic dinner with a nice river view and good quality food for a special occasion."*

Recommended: **Hotel Majitar Retreat Restaurant** ← Teesta river view, fine dining, special occasions

**Scenario 5: Quick budget meal for travelers**

> Customer: *"Just passing through on a road trip and need a quick affordable thali meal."*

Recommended: **Hotel Temi Restaurant Majitar** ← quick, affordable, thali, road trip

**Scenario 6: Chai and snacks by the river**

> Customer: *"Want to sit by the Teesta river and have some chai and light snacks in a rustic dhaba setting."*

Recommended: **The Riverside Dhaba Majitar** ← chai, pakora, riverside, rustic dhaba

---

### Step 9: Predicting New Reviews

The trained best model can predict sentiment for any new customer review:

```
Review   : "I was craving something cheesy and the cheese burst pizza here hit the spot perfectly."
Predicted: POSITIVE

Review   : "The food was amazing and the service was excellent. Highly recommend!"
Predicted: POSITIVE

Review   : "Completely disappointed. Cold food, rude staff and overpriced."
Predicted: NEGATIVE

Review   : "It was okay. Nothing special but nothing terrible either."
Predicted: NEUTRAL
```

---

## Visualizations

All 12 output charts are saved to the `output_plots/` directory:

| # | File | Description |
|---|------|-------------|
| 1 | `1_sentiment_distribution.png` | Overall sentiment counts + pie chart |
| 2 | `2_restaurant_sentiment_comparison.png` | Stacked sentiment bar per restaurant |
| 3 | `3_avg_rating_per_restaurant.png` | Average rating horizontal bar chart |
| 4 | `4_review_length_distribution.png` | Review word count distribution by sentiment |
| 5 | `5_wordclouds.png` | Word clouds for positive, neutral, negative |
| 6 | `6_model_comparison.png` | Test vs CV accuracy for all 5 models |
| 7 | `7_confusion_matrices.png` | Confusion matrix for each model |
| 8 | `8_per_class_metrics.png` | Precision, Recall, F1 per class (best model) |
| 9 | `9_roc_curves.png` | ROC-AUC curves (Logistic Regression, OvR) |
| 10 | `10_cross_validation_boxplot.png` | 5-fold CV accuracy distribution boxplot |
| 11 | `11_vader_comparison.png` | VADER vs rating-based label comparison |
| 12 | `12_top_features_per_class.png` | Most influential TF-IDF words per sentiment |

---

## Key Findings

**Overall Sentiment:** 60% of reviews across all Majhitar restaurants are positive, indicating generally good customer satisfaction in the area.

**Best-rated Restaurant:** Hotel Majitar Retreat Restaurant has the highest average rating (4.00/5) and is the only restaurant to break the positive threshold on average.

**Most Reviewed Category:** Positive reviews are consistently the longest, suggesting satisfied customers write more detailed feedback.

**Top Positive Words:** `good`, `great`, `excellent`, `loved`, `nice`, `tasty`, `best`

**Top Negative Words:** `poor`, `overpriced`, `slow`, `service`, `small`, `extremely`, `concern`

**Top Neutral Words:** `okay`, `average`, `nothing`, `nothing special`, `decent`

**Best ML Model:** Linear SVM — Test Accuracy 0.706, CV Mean 0.729

**VADER Agreement:** VADER closely matches rating-based labels, validating the labeling approach.

**Recommendation Engine:** Successfully identifies the correct restaurant for 6 distinct food craving/dining intent scenarios using keyword-profile matching weighted by real sentiment data.

---

## How to Run

### 1. Install dependencies

```bash
pip install pandas numpy scikit-learn matplotlib seaborn nltk wordcloud
```

### 2. Run the full pipeline

```bash
python sentiment_analysis.py
```

This will automatically call `dataset_builder.py`, generate `majitar_restaurant_reviews.csv`, run all 17 steps, save 12 visualizations to `output_plots/`, and save the best model to `sentiment_model.pkl`.

### 3. (Optional) Use real Google Reviews via Outscraper

Set your API key in `dataset_builder.py`:

```python
OUTSCRAPER_API_KEY = "your_api_key_here"
```

Sign up at [https://outscraper.com](https://outscraper.com) — free tier provides 150 reviews/month.

---

## Future Improvements

**Larger dataset** — scrape more real reviews from Google Maps as the restaurant base grows.

**Advanced models** — implement BERT, LSTM, or Transformer-based models for higher accuracy on small datasets.

**Aspect-based sentiment** — go beyond overall sentiment to detect opinions about specific aspects (food quality, service, ambiance, price).

**Web application** — deploy the model and recommendation engine as a Flask/Streamlit web app.

**Real-time monitoring** — create a live review monitoring dashboard for restaurant owners.

**Multilingual support** — handle Nepali, Sikkimese, and Hindi language reviews.

---

## Conclusion

This project demonstrates how **Natural Language Processing and Machine Learning** can be applied to analyze restaurant customer feedback automatically and translate that analysis into a practical recommendation system.

The sentiment model successfully classifies Majhitar restaurant reviews into three categories. More importantly, the **context-aware recommendation engine** shows how sentiment insights can be applied in a real-world scenario — understanding that when a customer says they want something *"cheesy with a good dessert"*, the data tells us that **Coffee Break** is the right recommendation because it has the highest positive sentiment specifically around cheesy food and desserts in Majitar.

---

## Author

**Shahidul Islam**
Text Analytics and Natural Language Processing — Assignment
#
