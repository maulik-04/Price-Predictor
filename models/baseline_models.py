"""
BASELINE MODELS: From Simple to Sophisticated
===============================================

PURPOSE:
    Establish performance benchmarks before training an LLM.
    
    CRITICAL INSIGHT: Without baselines, you can't answer
    "Is my LLM good?" You need to know: good compared to WHAT?

THE BASELINE LADDER:
    We deliberately start with the simplest possible model
    and add complexity one step at a time. Each step should
    improve MAE — if it doesn't, that complexity isn't helping.

    Random → Constant → Linear (3 feat) → Linear (BoW) 
                                          → Random Forest → XGBoost

MODEL PHILOSOPHY:
    - Models 1-2: Do we even need to look at the product?
    - Models 3:   Do simple numerical features help?
    - Models 4-6: Does product text help? How much?

KEY FINDING:
    XGBoost on Bag-of-Words achieves $68.23 MAE.
    Our fine-tuned LLM achieves $58.74 MAE — 13.9% better.
    The improvement comes from semantic understanding, not just
    word counting.
"""

import re
import random
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.ensemble import RandomForestRegressor
import xgboost as xgb


# ── Shared Vectorizer ─────────────────────────────────────────
# Created once, shared across all text-based models so they
# use exactly the same vocabulary and feature space.
# max_features=2000: keeps top 2000 words by frequency
# stop_words='english': removes 'the', 'a', 'is' etc.
vectorizer = CountVectorizer(
    max_features=2000,
    stop_words='english'
)


# ── MODEL 1: Random Pricer ────────────────────────────────────
def make_random_pricer():
    """
    Returns a function that predicts a random price $1-$999.
    
    PURPOSE: This is the absolute floor. Any model that can't
    beat random predictions is completely useless.
    
    Expected MAE: ~$285 (roughly 1/3 of the $0-$999 range)
    """
    def predict(product) -> float:
        # Ignore the product entirely — pure random
        return random.randrange(1, 1000)
    
    predict.__name__ = "Random Pricer"
    return predict


# ── MODEL 2: Constant Pricer ──────────────────────────────────
def make_constant_pricer(train_products: list):
    """
    Returns a function that always predicts the training mean.
    
    PURPOSE: Tests whether knowing the average price beats
    random. It always does — this is the second baseline.
    
    This represents a model that learned the DISTRIBUTION
    but nothing about INDIVIDUAL products.
    
    Args:
        train_products: Training set (we compute its mean price)
    """
    mean_price = sum(p.price for p in train_products) / len(train_products)
    print(f"Training set mean price: ${mean_price:.2f}")
    
    def predict(product) -> float:
        # Ignore the product — always return training mean
        return mean_price
    
    predict.__name__ = "Constant Pricer (mean)"
    return predict


# ── MODEL 3: Linear Regression (3 numerical features) ─────────
def make_linear_regression_pricer(train_products: list):
    """
    Returns a function that uses 3 hand-crafted features.
    
    FEATURES:
        weight:         product weight in pounds
                        Intuition: heavier things cost more
        weight_missing: 1 if weight is unknown, 0 if known
                        Intuition: cheap items often skip weight
        text_length:    number of characters in the summary
                        Intuition: expensive items have longer
                        descriptions with more technical detail
    
    WHY LINEAR REGRESSION:
        Simple, interpretable. The learned coefficients tell us
        exactly how much each feature influences the predicted
        price. Good for understanding feature importance.
    
    LIMITATION:
        Only 3 features → misses all information in the text
        (brand, category, specific product type etc.)
    """
    def get_features(product) -> dict:
        return {
            'weight':         product.weight or 0.0,
            'weight_missing': 1 if (product.weight or 0) == 0 else 0,
            'text_length':    len(product.summary or product.text),
        }
    
    # Build training DataFrame
    X_train = pd.DataFrame([get_features(p) for p in train_products])
    y_train = [p.price for p in train_products]
    
    # Train the model
    model = LinearRegression()
    model.fit(X_train, y_train)
    
    # Print learned coefficients — useful for interpretation
    print("Linear Regression coefficients:")
    for col, coef in zip(X_train.columns, model.coef_):
        direction = '↑' if coef > 0 else '↓'
        print(f"  {col:<20} {coef:>8.4f} {direction}")
    print(f"  {'intercept':<20} {model.intercept_:>8.4f}")
    
    def predict(product) -> float:
        features = get_features(product)
        X        = pd.DataFrame([features])
        return max(0, model.predict(X)[0])
    
    predict.__name__ = "Linear Regression (3 features)"
    return predict


# ── MODEL 4: Linear Regression (Bag of Words) ─────────────────
def make_bow_linear_pricer(train_products: list):
    """
    Returns a function using 2000 word-count features.
    
    BAG OF WORDS (BoW):
        Each product description becomes a vector of word counts:
        "Sony noise cancelling headphones" →
        [0, 0, 1, 0, 1, 0, 1, 1, 0, ...]
         ↑           ↑      ↑  ↑
        'apple'    'sony' 'noise' 'cancelling'
    
    The model learns which words predict higher/lower prices.
    Example learned patterns:
        'commercial'  → +$150 (commercial grade = expensive)
        'professional'→ +$80  (professional = expensive)
        'replacement' → -$40  (replacement parts = cheap)
        'cable'       → -$30  (cables = cheap accessories)
    
    LIMITATION:
        Treats words independently. Doesn't understand that
        "Sony noise cancelling" means premium electronics —
        it just sees three separate word counts.
    """
    # Use summary if available (LLM-cleaned), otherwise raw text
    documents = [p.summary or p.text for p in train_products]
    prices    = np.array([p.price for p in train_products])
    
    # Fit vectorizer — learns vocabulary from training data
    X = vectorizer.fit_transform(documents)
    print(f"Vocabulary: {len(vectorizer.get_feature_names_out()):,} words")
    print(f"Matrix: {X.shape[0]:,} items × {X.shape[1]:,} features")
    
    # Train linear regression on word count features
    model = LinearRegression()
    model.fit(X, prices)
    
    # Show most price-predictive words
    words = vectorizer.get_feature_names_out()
    coefs = model.coef_
    top_pos = np.argsort(coefs)[-5:][::-1]
    top_neg = np.argsort(coefs)[:5]
    
    print("Top price-INCREASING words:", [(words[i], f"+${coefs[i]:.0f}") for i in top_pos])
    print("Top price-DECREASING words:", [(words[i], f"-${abs(coefs[i]):.0f}") for i in top_neg])
    
    def predict(product) -> float:
        text = product.summary or product.text
        x    = vectorizer.transform([text])
        return max(0, model.predict(x)[0])
    
    predict.__name__ = "Linear Regression (BoW)"
    return predict


# ── MODEL 5: Random Forest ─────────────────────────────────────
def make_random_forest_pricer(train_products: list, subset: int = 15_000):
    """
    Returns a function using an ensemble of 100 decision trees.
    
    HOW RANDOM FOREST WORKS:
        Builds 100 decision trees, each trained on a random
        subset of data AND a random subset of features.
        Final prediction = average of all 100 trees.
    
    EACH DECISION TREE IS LIKE:
        IF 'television' count > 2:
          IF 'oled' count > 0:
            → predict $800
          ELSE:
            → predict $400
        ELSE:
          ...
    
    WHY BETTER THAN LINEAR REGRESSION:
        Can capture non-linear relationships. "television" alone
        might not predict high price, but "television" + "oled"
        together signals premium product. Linear regression misses
        these interactions.
    
    WHY subset=15_000:
        Full 800k Random Forest takes 15+ hours to train.
        15k gives us a representative result in ~5 minutes.
    """
    documents = [p.summary or p.text for p in train_products[:subset]]
    prices    = np.array([p.price for p in train_products[:subset]])
    
    # If vectorizer not yet fit, fit it now
    if not hasattr(vectorizer, 'vocabulary_'):
        vectorizer.fit(documents)
    
    X = vectorizer.transform(documents)
    
    print(f"Training Random Forest on {subset:,} items with 100 trees...")
    model = RandomForestRegressor(
        n_estimators=100,  # 100 trees — more = better but slower
        random_state=42,   # reproducible
        n_jobs=-1          # use all CPU cores for parallel training
    )
    model.fit(X, prices)
    print("✅ Random Forest trained")
    
    def predict(product) -> float:
        text = product.summary or product.text
        x    = vectorizer.transform([text])
        return max(0, model.predict(x)[0])
    
    predict.__name__ = "Random Forest (100 trees)"
    return predict


# ── MODEL 6: XGBoost ──────────────────────────────────────────
def make_xgboost_pricer(train_products: list):
    """
    Returns a function using gradient boosting.
    
    HOW XGBOOST DIFFERS FROM RANDOM FOREST:
        Random Forest: 100 independent trees, average result
        XGBoost:       1000 trees built SEQUENTIALLY
                       Each tree learns from the ERRORS of all
                       previous trees (gradient boosting)
    
    THE GRADIENT BOOSTING IDEA:
        Start: predict the mean price ($141) for everything
        Error: some products predicted too high, some too low
        Tree 1: learn to correct those errors
        Tree 2: learn to correct Tree 1's errors
        ...
        Tree 1000: correct Tree 999's errors
        Final: sum of all corrections = good prediction
    
    WHY learning_rate=0.1:
        Each tree only corrects 10% of remaining error.
        If we corrected 100%, one bad tree ruins everything.
        0.1 is conservative — 1000 small steps vs 100 large steps.
    
    WHY XGBOOST BEATS RANDOM FOREST HERE:
        - Trains on full dataset (not just 15k subset)
        - Sequential correction finds patterns RF misses
        - Generally better at tabular/text data tasks
    """
    documents = [p.summary or p.text for p in train_products]
    prices    = np.array([p.price for p in train_products])
    
    if not hasattr(vectorizer, 'vocabulary_'):
        vectorizer.fit(documents)
    
    X = vectorizer.transform(documents)
    
    print(f"Training XGBoost on {len(train_products):,} items...")
    model = xgb.XGBRegressor(
        n_estimators=1000,   # 1000 trees
        learning_rate=0.1,   # conservative step size
        random_state=42,
        n_jobs=-1
    )
    model.fit(X, prices)
    print("✅ XGBoost trained")
    
    def predict(product) -> float:
        text = product.summary or product.text
        x    = vectorizer.transform([text])
        return max(0, model.predict(x)[0])
    
    predict.__name__ = "XGBoost (gradient boosting)"
    return predict
