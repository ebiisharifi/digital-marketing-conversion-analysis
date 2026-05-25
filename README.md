# Data Mining Analysis of Conversion Patterns in Digital Marketing Campaigns

End-to-end data mining project on an 8,000-record digital marketing dataset. The goal is to identify the customer behaviors and campaign characteristics that drive conversions, using a mix of exploratory analysis, predictive modeling, and association rule mining.

## A Note on Development

This project reflects my background in business analytics and data mining. The methodology, choice of techniques, evaluation criteria, and  interpretation of findings are my own. I used Claude Code to assist with the Python implementation, and visualization details, which let me focus on the analytical reasoning rather than syntax.

## Project Overview

The dataset contains demographic data, marketing campaign metrics, customer engagement signals, and historical purchase information. The target variable is binary (Conversion: 1 = converted, 0 = not converted). The class is imbalanced (about 87.6% converted), so the analysis emphasizes lift and ROC-AUC alongside raw accuracy.

The project follows the Knowledge Discovery in Databases (KDD) framework:

1. Data loading, cleaning, and inspection
2. Descriptive statistics and exploratory data analysis (EDA)
3. Target variable analysis
4. Predictive modeling (Logistic Regression, Decision Tree, Random Forest)
5. Apriori association rule mining

## Dataset

- 8,000 customer records, 20 attributes
- Source: Kaggle, "Predict Conversion in Digital Marketing Dataset"
- Demographic: Age, Gender, Income
- Marketing: CampaignChannel (Email, Social Media, SEO, PPC, Referral), CampaignType (Awareness, Consideration, Conversion, Retention), AdSpend, ClickThroughRate, ConversionRate
- Engagement: WebsiteVisits, PagesPerVisit, TimeOnSite, SocialShares, EmailOpens, EmailClicks
- Historical: PreviousPurchases, LoyaltyPoints
- Target: Conversion (binary)

CustomerID, AdvertisingPlatform, and AdvertisingTool are dropped (identifier and confidential single-value columns).

## How to Run

Clone the repo and install dependencies:

```bash
git clone <your-repo-url>
cd <repo-name>
pip install -r requirements.txt
```

Either open the notebook:

```bash
jupyter notebook notebooks/digital_marketing_conversion_analysis.ipynb
```

Or run the script (produces the same figures and CSV outputs):

```bash
python analysis.py
```

## Key Findings

**Engagement beats demographics.** The strongest predictors of conversion are behavioral: time on site, pages per visit, click-through rate, and ad spend. Age and income have almost no predictive power on their own.

**Campaign type matters more than channel.** "Conversion"-type campaigns convert about 93% of customers, while Awareness, Consideration, and Retention sit near 86%. The channel (Email, SEO, Social Media, PPC, Referral) has essentially no effect on conversion rate. Apriori confirmed this: channel did not appear in any of the top rules.

**Strongest customer profile.** Rules combining high ad spend, female gender, and Conversion-type campaigns reached 95-96% confidence with the highest lift values.

**Model comparison on a 25% test set:**

| Model               | Accuracy | ROC-AUC |
|---------------------|----------|---------|
| Logistic Regression | 0.75     | 0.78    |
| Decision Tree       | 0.76     | 0.70    |
| Random Forest       | 0.90     | 0.81    |

Random Forest is the strongest model. The top features driving its predictions are TimeOnSite, PagesPerVisit, ClickThroughRate, AdSpend, and ConversionRate, all behavioral signals.

## Apriori Results Summary

- Minimum support: 5%
- Minimum confidence: 65%
- Frequent itemsets found: 809 (sizes 1 to 4)
- Association rules generated: 374

Top rules predicting conversion (sorted by lift):

| Antecedent                                          | Support | Confidence | Lift  |
|-----------------------------------------------------|---------|------------|-------|
| Type=Conversion, Purchase=Regular                   | 7.4%    | 96.1%      | 1.097 |
| AdSpend=High, Gender=Female, Type=Conversion        | 5.2%    | 95.9%      | 1.094 |
| AdSpend=High, Gender=Female, Purchase=Loyal         | 7.5%    | 95.6%      | 1.090 |
| Loyalty=Gold, Type=Conversion                       | 8.0%    | 95.5%      | 1.090 |
| AdSpend=High, Type=Conversion                       | 8.3%    | 95.1%      | 1.085 |

The full rule set is in `results/apriori_rules_to_converted.csv`.

## Business Recommendations

1. **Lead scoring.** Flag users in real time once they cross thresholds around 8 minutes time on site, 4-5 email clicks, or 25 website visits.
2. **Reallocate budget toward campaign messaging.** Channel selection has little impact on conversion in this dataset; campaign type does. Test more "Conversion"-style messaging.
3. **Loyalty program.** High-volume historical buyers convert at much higher rates. A VIP or early-access tier could amplify this.
4. **Site optimization.** Run A/B tests on UI changes designed to close the engagement-duration gap between converters and non-converters.

## Tools

- Python 3.10+
- pandas, numpy for data manipulation
- matplotlib, seaborn for visualization
- scikit-learn for predictive modeling
- Apriori implemented from scratch (no external dependency needed)
