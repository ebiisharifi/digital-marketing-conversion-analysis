"""
Data Mining Analysis of Conversion Patterns in Digital Marketing Campaigns

This script performs:
  1. Data loading and cleaning
  2. Descriptive statistics and exploratory data analysis (EDA)
  3. Target variable analysis on Conversion
  4. Predictive analysis using Logistic Regression and Decision Tree classifiers
  5. Apriori association rule mining to discover hidden patterns

Dataset: 8,000 customer records, 20 columns, binary target (Conversion).
"""

import os
import warnings
from itertools import combinations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
)

warnings.filterwarnings("ignore")

# Project paths
DATA_PATH = "data/digital_marketing_campaign.csv"
FIGURES_DIR = "figures"
RESULTS_DIR = "results"

# Make sure output folders exist
os.makedirs(FIGURES_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

# Consistent plotting style
sns.set_style("whitegrid")
plt.rcParams["figure.dpi"] = 100
plt.rcParams["savefig.dpi"] = 150
plt.rcParams["savefig.bbox"] = "tight"

RANDOM_STATE = 42


# ---------------------------------------------------------------------------
# 1. Load and clean the data
# ---------------------------------------------------------------------------

def load_data(path):
    """Load the dataset and drop columns that are not useful for analysis."""
    df = pd.read_csv(path)

    # CustomerID is just an identifier; AdvertisingPlatform and AdvertisingTool
    # are listed as confidential and contain only a single value each.
    columns_to_drop = ["CustomerID", "AdvertisingPlatform", "AdvertisingTool"]
    df = df.drop(columns=[c for c in columns_to_drop if c in df.columns])

    return df


def basic_data_checks(df):
    """Print a quick overview: shape, missing values, duplicates, dtypes."""
    print("=" * 60)
    print("DATA OVERVIEW")
    print("=" * 60)
    print(f"Shape: {df.shape[0]} rows, {df.shape[1]} columns")
    print(f"Duplicate rows: {df.duplicated().sum()}")
    print(f"Total missing values: {df.isnull().sum().sum()}")

    print("\nColumn types:")
    print(df.dtypes)

    print("\nFirst 5 rows:")
    print(df.head())


# ---------------------------------------------------------------------------
# 2. Descriptive analysis
# ---------------------------------------------------------------------------

def descriptive_statistics(df):
    """Save descriptive statistics for numeric variables to a CSV."""
    numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
    stats = df[numeric_cols].describe().T
    stats["median"] = df[numeric_cols].median()
    stats = stats[["count", "mean", "median", "std", "min", "25%", "50%", "75%", "max"]]

    out_path = os.path.join(RESULTS_DIR, "descriptive_statistics.csv")
    stats.to_csv(out_path)
    print(f"\nDescriptive statistics saved to {out_path}")
    return stats


def plot_numeric_distributions(df):
    """Plot histograms for all numeric variables."""
    numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
    numeric_cols = [c for c in numeric_cols if c != "Conversion"]

    n_cols = 3
    n_rows = (len(numeric_cols) + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 4 * n_rows))
    axes = axes.flatten()

    for i, col in enumerate(numeric_cols):
        axes[i].hist(df[col], bins=25, color="steelblue", edgecolor="white")
        axes[i].axvline(df[col].mean(), color="red", linestyle="--",
                        label=f"Mean = {df[col].mean():.1f}")
        axes[i].axvline(df[col].median(), color="orange", linestyle=":",
                        label=f"Median = {df[col].median():.1f}")
        axes[i].set_title(col)
        axes[i].legend(fontsize=8)

    # Hide unused subplots
    for j in range(len(numeric_cols), len(axes)):
        axes[j].set_visible(False)

    plt.suptitle("Distribution of Numeric Variables", fontsize=14, y=1.00)
    plt.tight_layout()
    out_path = os.path.join(FIGURES_DIR, "01_numeric_distributions.png")
    plt.savefig(out_path)
    plt.close()
    print(f"Saved: {out_path}")


def plot_categorical_distributions(df):
    """Plot bar charts for categorical variables."""
    cat_cols = ["Gender", "CampaignChannel", "CampaignType"]
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    for ax, col in zip(axes, cat_cols):
        counts = df[col].value_counts()
        ax.bar(counts.index, counts.values, color="steelblue", edgecolor="white")
        ax.set_title(f"Count by {col}")
        ax.set_ylabel("Count")
        ax.tick_params(axis="x", rotation=30)

    plt.tight_layout()
    out_path = os.path.join(FIGURES_DIR, "02_categorical_distributions.png")
    plt.savefig(out_path)
    plt.close()
    print(f"Saved: {out_path}")


# ---------------------------------------------------------------------------
# 3. Target variable (Conversion) analysis
# ---------------------------------------------------------------------------

def conversion_overview(df):
    """Class balance + conversion rate broken down by each categorical."""
    print("\n" + "=" * 60)
    print("TARGET VARIABLE: CONVERSION")
    print("=" * 60)

    counts = df["Conversion"].value_counts()
    total = len(df)
    print(f"Converted (1):     {counts[1]:>5d}  ({counts[1] / total * 100:.1f}%)")
    print(f"Not converted (0): {counts[0]:>5d}  ({counts[0] / total * 100:.1f}%)")

    # Conversion rate by each categorical
    for col in ["Gender", "CampaignChannel", "CampaignType"]:
        rates = df.groupby(col)["Conversion"].mean() * 100
        print(f"\nConversion rate by {col}:")
        print(rates.round(2).to_string())

    # Plot
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))

    # Pie chart for class balance
    axes[0].pie(counts.values, labels=["Converted", "Not Converted"],
                colors=["#4CAF50", "#E74C3C"], autopct="%.1f%%", startangle=90)
    axes[0].set_title("Class Balance")

    # Bar charts for each categorical
    for ax, col in zip(axes[1:], ["Gender", "CampaignChannel", "CampaignType"]):
        rates = df.groupby(col)["Conversion"].mean() * 100
        rates = rates.sort_values(ascending=False)
        ax.bar(rates.index, rates.values, color="steelblue", edgecolor="white")
        ax.set_title(f"Conversion Rate by {col}")
        ax.set_ylabel("Conversion Rate (%)")
        ax.set_ylim(0, 100)
        ax.tick_params(axis="x", rotation=30)
        for i, v in enumerate(rates.values):
            ax.text(i, v + 1, f"{v:.1f}%", ha="center", fontsize=9)

    plt.tight_layout()
    out_path = os.path.join(FIGURES_DIR, "03_conversion_breakdown.png")
    plt.savefig(out_path)
    plt.close()
    print(f"\nSaved: {out_path}")


def numeric_vs_conversion(df):
    """Boxplots showing how numeric features differ between converters and non-converters."""
    numeric_cols = ["Age", "Income", "AdSpend", "ClickThroughRate",
                    "WebsiteVisits", "PagesPerVisit", "TimeOnSite",
                    "EmailClicks", "PreviousPurchases", "LoyaltyPoints"]

    fig, axes = plt.subplots(2, 5, figsize=(18, 8))
    axes = axes.flatten()

    for i, col in enumerate(numeric_cols):
        sns.boxplot(data=df, x="Conversion", y=col, ax=axes[i],
                    hue="Conversion", palette=["#E74C3C", "#4CAF50"],
                    legend=False)
        axes[i].set_title(f"{col} by Conversion")
        axes[i].set_xlabel("")
        axes[i].set_xticks([0, 1])
        axes[i].set_xticklabels(["Not Converted", "Converted"])

    plt.tight_layout()
    out_path = os.path.join(FIGURES_DIR, "04_numeric_vs_conversion.png")
    plt.savefig(out_path)
    plt.close()
    print(f"Saved: {out_path}")

    # Group means table
    means = df.groupby("Conversion")[numeric_cols].mean().T
    means.columns = ["Not Converted", "Converted"]
    means["Difference (%)"] = ((means["Converted"] - means["Not Converted"])
                                / means["Not Converted"] * 100).round(2)

    out_path = os.path.join(RESULTS_DIR, "conversion_group_means.csv")
    means.to_csv(out_path)
    print(f"Group means saved to {out_path}")
    return means


def correlation_heatmap(df):
    """Correlation matrix among numeric variables."""
    numeric_df = df.select_dtypes(include=np.number)
    corr = numeric_df.corr()

    plt.figure(figsize=(12, 9))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm",
                center=0, square=True, linewidths=0.5,
                cbar_kws={"shrink": 0.8})
    plt.title("Correlation Heatmap of Numeric Variables")
    plt.tight_layout()
    out_path = os.path.join(FIGURES_DIR, "05_correlation_heatmap.png")
    plt.savefig(out_path)
    plt.close()
    print(f"Saved: {out_path}")

    # Correlation with the target only
    target_corr = corr["Conversion"].drop("Conversion").sort_values(key=abs, ascending=False)
    print("\nCorrelation with Conversion (strongest first):")
    print(target_corr.round(3).to_string())
    return target_corr


# ---------------------------------------------------------------------------
# 4. Predictive analysis
# ---------------------------------------------------------------------------

def prepare_features(df):
    """One-hot encode categoricals and return X (features) and y (target)."""
    df_enc = pd.get_dummies(df, columns=["Gender", "CampaignChannel", "CampaignType"],
                            drop_first=True)
    y = df_enc["Conversion"]
    X = df_enc.drop(columns=["Conversion"])
    return X, y


def evaluate_model(name, model, X_test, y_test):
    """Print accuracy, classification report, and ROC-AUC for a fitted model."""
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    acc = accuracy_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_proba)

    print(f"\n--- {name} ---")
    print(f"Accuracy: {acc:.4f}")
    print(f"ROC-AUC:  {auc:.4f}")
    print("\nClassification report:")
    print(classification_report(y_test, y_pred, target_names=["Not Converted", "Converted"]))

    return {
        "model": name,
        "accuracy": round(acc, 4),
        "roc_auc": round(auc, 4),
        "y_pred": y_pred,
        "y_proba": y_proba,
    }


def run_predictive_models(df):
    """Train Logistic Regression, Decision Tree, and Random Forest and compare them."""
    print("\n" + "=" * 60)
    print("PREDICTIVE MODELS")
    print("=" * 60)

    X, y = prepare_features(df)

    # Train/test split (stratified to keep class proportions)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, stratify=y, random_state=RANDOM_STATE
    )

    # Scale features for Logistic Regression (tree models don't need scaling)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    results = []

    # Logistic Regression
    logreg = LogisticRegression(max_iter=1000, class_weight="balanced",
                                 random_state=RANDOM_STATE)
    logreg.fit(X_train_scaled, y_train)
    results.append(evaluate_model("Logistic Regression", logreg, X_test_scaled, y_test))

    # Decision Tree
    tree = DecisionTreeClassifier(max_depth=5, class_weight="balanced",
                                   random_state=RANDOM_STATE)
    tree.fit(X_train, y_train)
    results.append(evaluate_model("Decision Tree", tree, X_test, y_test))

    # Random Forest
    rf = RandomForestClassifier(n_estimators=200, max_depth=10,
                                 class_weight="balanced",
                                 random_state=RANDOM_STATE, n_jobs=-1)
    rf.fit(X_train, y_train)
    results.append(evaluate_model("Random Forest", rf, X_test, y_test))

    # Save model comparison
    summary = pd.DataFrame([{"model": r["model"],
                              "accuracy": r["accuracy"],
                              "roc_auc": r["roc_auc"]} for r in results])
    summary.to_csv(os.path.join(RESULTS_DIR, "model_comparison.csv"), index=False)
    print("\nModel comparison:")
    print(summary.to_string(index=False))

    # ROC curves
    plt.figure(figsize=(8, 6))
    for r in results:
        fpr, tpr, _ = roc_curve(y_test, r["y_proba"])
        plt.plot(fpr, tpr, label=f"{r['model']} (AUC = {r['roc_auc']:.3f})")
    plt.plot([0, 1], [0, 1], "k--", label="Random guess")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curves")
    plt.legend()
    plt.tight_layout()
    out_path = os.path.join(FIGURES_DIR, "06_roc_curves.png")
    plt.savefig(out_path)
    plt.close()
    print(f"Saved: {out_path}")

    # Confusion matrix for the best model (Random Forest)
    cm = confusion_matrix(y_test, rf.predict(X_test))
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["Not Converted", "Converted"],
                yticklabels=["Not Converted", "Converted"])
    plt.title("Confusion Matrix - Random Forest")
    plt.ylabel("Actual")
    plt.xlabel("Predicted")
    plt.tight_layout()
    out_path = os.path.join(FIGURES_DIR, "07_confusion_matrix.png")
    plt.savefig(out_path)
    plt.close()
    print(f"Saved: {out_path}")

    # Feature importance from Random Forest
    importance = pd.DataFrame({
        "feature": X.columns,
        "importance": rf.feature_importances_
    }).sort_values("importance", ascending=False)

    plt.figure(figsize=(10, 8))
    plt.barh(importance["feature"][:15][::-1], importance["importance"][:15][::-1],
             color="steelblue")
    plt.xlabel("Importance")
    plt.title("Top 15 Feature Importances (Random Forest)")
    plt.tight_layout()
    out_path = os.path.join(FIGURES_DIR, "08_feature_importance.png")
    plt.savefig(out_path)
    plt.close()
    print(f"Saved: {out_path}")

    importance.to_csv(os.path.join(RESULTS_DIR, "feature_importance.csv"), index=False)
    print("\nTop 10 features driving conversion:")
    print(importance.head(10).to_string(index=False))

    # Decision tree visualization (limited depth for readability)
    plt.figure(figsize=(20, 10))
    plot_tree(tree, feature_names=X.columns, class_names=["Not Converted", "Converted"],
              filled=True, rounded=True, fontsize=8, max_depth=3)
    plt.title("Decision Tree (top 3 levels)")
    out_path = os.path.join(FIGURES_DIR, "09_decision_tree.png")
    plt.savefig(out_path)
    plt.close()
    print(f"Saved: {out_path}")

    return summary, importance


# ---------------------------------------------------------------------------
# 5. Apriori association rule mining (manual implementation)
# ---------------------------------------------------------------------------

def discretize_for_apriori(df):
    """Convert numeric variables into bins, return a list of transactions
    (each transaction is a list of 'prefix=value' strings)."""

    df_d = df.copy()

    # Age bins
    df_d["Age_bin"] = pd.cut(df_d["Age"], bins=[0, 30, 45, 60, 100],
                             labels=["Young", "Middle", "Senior", "Elder"])

    # Income, AdSpend, LoyaltyPoints - equal-width into 3 bins
    df_d["Income_bin"] = pd.cut(df_d["Income"], bins=3, labels=["Low", "Medium", "High"])
    df_d["AdSpend_bin"] = pd.cut(df_d["AdSpend"], bins=3, labels=["Low", "Medium", "High"])
    df_d["Loyalty_bin"] = pd.cut(df_d["LoyaltyPoints"], bins=3,
                                  labels=["Bronze", "Silver", "Gold"])

    # PreviousPurchases - custom bins
    df_d["Purchase_bin"] = pd.cut(df_d["PreviousPurchases"], bins=[-1, 2, 5, 10],
                                   labels=["New", "Regular", "Loyal"])

    # Build transactions
    transactions = []
    for _, row in df_d.iterrows():
        items = [
            f"Age={row['Age_bin']}",
            f"Income={row['Income_bin']}",
            f"AdSpend={row['AdSpend_bin']}",
            f"Loyalty={row['Loyalty_bin']}",
            f"Purchase={row['Purchase_bin']}",
            f"Channel={row['CampaignChannel']}",
            f"Type={row['CampaignType']}",
            f"Gender={row['Gender']}",
            "Converted" if row["Conversion"] == 1 else "NotConverted",
        ]
        transactions.append(items)

    return transactions


def get_frequent_itemsets(transactions, min_support):
    """Generate frequent itemsets of size 1 to 4 with given minimum support.
    Returns a dict {itemset (frozenset): support}."""

    n = len(transactions)
    min_count = min_support * n

    # Count single items
    item_counts = {}
    for t in transactions:
        for item in set(t):
            item_counts[item] = item_counts.get(item, 0) + 1

    # Keep only frequent 1-itemsets
    L1 = {frozenset([item]): count / n
          for item, count in item_counts.items() if count >= min_count}

    all_itemsets = dict(L1)
    current_L = L1

    # Generate Lk for k = 2, 3, 4
    for k in range(2, 5):
        if not current_L:
            break

        # Generate candidate itemsets of size k by joining items from L(k-1)
        current_items = set()
        for itemset in current_L.keys():
            for item in itemset:
                current_items.add(item)

        candidates = set()
        for combo in combinations(current_items, k):
            candidates.add(frozenset(combo))

        # Count support
        Lk = {}
        for cand in candidates:
            count = sum(1 for t in transactions if cand.issubset(set(t)))
            if count >= min_count:
                Lk[cand] = count / n

        all_itemsets.update(Lk)
        current_L = Lk
        print(f"  Frequent {k}-itemsets found: {len(Lk)}")

    return all_itemsets


def generate_rules(frequent_itemsets, min_confidence):
    """Generate association rules X -> Y with given minimum confidence.
    Returns a list of dicts with antecedent, consequent, support, confidence, lift."""

    rules = []
    for itemset, support in frequent_itemsets.items():
        if len(itemset) < 2:
            continue
        for i in range(1, len(itemset)):
            for antecedent in combinations(itemset, i):
                antecedent = frozenset(antecedent)
                consequent = itemset - antecedent

                antecedent_support = frequent_itemsets.get(antecedent)
                consequent_support = frequent_itemsets.get(consequent)
                if antecedent_support is None or consequent_support is None:
                    continue

                confidence = support / antecedent_support
                if confidence < min_confidence:
                    continue

                lift = confidence / consequent_support

                rules.append({
                    "antecedent": ", ".join(sorted(antecedent)),
                    "consequent": ", ".join(sorted(consequent)),
                    "support": round(support, 4),
                    "confidence": round(confidence, 4),
                    "lift": round(lift, 4),
                })

    return rules


def run_apriori(df, min_support=0.05, min_confidence=0.65):
    """Run the full Apriori pipeline and save the top rules."""
    print("\n" + "=" * 60)
    print("APRIORI ASSOCIATION RULE MINING")
    print("=" * 60)
    print(f"Minimum support:    {min_support}")
    print(f"Minimum confidence: {min_confidence}")

    print("\nDiscretizing numeric variables and building transactions...")
    transactions = discretize_for_apriori(df)
    print(f"Total transactions: {len(transactions)}")

    print("\nFinding frequent itemsets...")
    frequent_itemsets = get_frequent_itemsets(transactions, min_support)
    print(f"Total frequent itemsets: {len(frequent_itemsets)}")

    print("\nGenerating association rules...")
    rules = generate_rules(frequent_itemsets, min_confidence)
    print(f"Total rules generated: {len(rules)}")

    rules_df = pd.DataFrame(rules)

    # Keep rules whose consequent is exactly the conversion outcome
    if not rules_df.empty:
        rules_to_conv = rules_df[rules_df["consequent"] == "Converted"].copy()
        rules_to_conv = rules_to_conv.sort_values("lift", ascending=False)

        out_path = os.path.join(RESULTS_DIR, "apriori_rules_to_converted.csv")
        rules_to_conv.to_csv(out_path, index=False)
        print(f"\nRules predicting Converted saved to {out_path}")

        print("\nTop 10 rules predicting Converted (by lift):")
        print(rules_to_conv.head(10).to_string(index=False))

        # Plot top rules
        if len(rules_to_conv) > 0:
            top = rules_to_conv.head(15)
            fig, axes = plt.subplots(1, 2, figsize=(16, 6))

            axes[0].scatter(rules_df["support"], rules_df["confidence"],
                            c=rules_df["lift"], cmap="YlOrRd",
                            alpha=0.7, edgecolor="black")
            axes[0].set_xlabel("Support")
            axes[0].set_ylabel("Confidence")
            axes[0].set_title("Association Rules: Support vs Confidence (color = Lift)")
            cb = plt.colorbar(axes[0].collections[0], ax=axes[0])
            cb.set_label("Lift")

            top_plot = top.iloc[::-1]
            axes[1].barh(range(len(top_plot)), top_plot["lift"], color="darkred")
            axes[1].set_yticks(range(len(top_plot)))
            axes[1].set_yticklabels([a[:60] + "..." if len(a) > 60 else a
                                      for a in top_plot["antecedent"]], fontsize=8)
            axes[1].axvline(1.0, color="red", linestyle="--", label="Lift = 1")
            axes[1].set_xlabel("Lift")
            axes[1].set_title("Top 15 Rules Predicting Conversion")
            axes[1].legend()

            plt.tight_layout()
            out_path = os.path.join(FIGURES_DIR, "10_apriori_rules.png")
            plt.savefig(out_path)
            plt.close()
            print(f"Saved: {out_path}")

        return rules_to_conv
    return pd.DataFrame()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    df = load_data(DATA_PATH)
    basic_data_checks(df)

    descriptive_statistics(df)
    plot_numeric_distributions(df)
    plot_categorical_distributions(df)

    conversion_overview(df)
    numeric_vs_conversion(df)
    correlation_heatmap(df)

    run_predictive_models(df)
    run_apriori(df, min_support=0.05, min_confidence=0.65)

    print("\n" + "=" * 60)
    print("ANALYSIS COMPLETE")
    print("=" * 60)
    print(f"Figures saved to: {FIGURES_DIR}/")
    print(f"Results saved to: {RESULTS_DIR}/")


if __name__ == "__main__":
    main()
