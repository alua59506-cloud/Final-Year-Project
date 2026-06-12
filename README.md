# Final-Year-Project
# US Airline Sentiment Analysis Pipeline

## Project Overview
This project focuses on the development of a robust **Natural Language Processing (NLP) pipeline** designed to automate the sentiment analysis of Twitter data for major US airlines. The system transitions from raw data ingestion and exploratory analysis to a high-performing classification model capable of generating business-critical insights such as **Net Brand Sentiment Scores** and **Temporal Sentiment Trends**.

## Dataset Description
The analysis utilizes the `Tweets.csv` dataset, which contains **14,640 entries** and 15 initial features.
*   **Target Classes:** Negative (62.69%), Neutral (21.17%), and Positive (16.14%).
*   **Key Attributes:** Tweet text, airline name, tweet timestamp, and negative reason.

## Technical Workflow

### 1. Data Cleaning & Engineering
To ensure data integrity, a rigorous cleaning process was implemented:
*   **High-Missingness Pruning:** Columns such as `negativereason_gold` (>99% missing) and `tweet_coord` (>93% missing) were dropped.
*   **Deduplication:** A three-tier deduplication process removed 39 full row duplicates, 116 duplicate `tweet_id` entries, and 58 duplicate `text` entries to prevent model bias.
*   **Feature Extraction:** New metadata features, such as `tweet_length`, were engineered to capture structural patterns in consumer feedback.

### 2. Sophisticated NLP Preprocessing
A custom preprocessing function was developed to normalize text while preserving sentiment context:
*   **Normalization:** Lowercasing, contraction expansion (e.g., "don't" to "do not"), and removal of URLs/Twitter handles.
*   **Contextual Stopwords:** Standard NLTK stopwords were used, but **negation words** (e.g., "not", "never") were explicitly retained to prevent misclassifying negative sentiment.
*   **Tokenization & Lemmatization:** Utilized `TweetTokenizer` for social media-specific text and `WordNetLemmatizer` to reduce words to their base forms.

### 3. Model Development & Evaluation
The project compared two primary architectures using a **stratified 80/20 train-test split** to maintain class distribution:
*   **Multinomial Naive Bayes (Baseline):** Achieved 75.69% accuracy but struggled with minority class recall.
*   **Linear SVM (Champion):** Implemented with **TF-IDF Vectorization** (unigrams/bigrams) and **`class_weight='balanced'`** to handle the heavy class imbalance.

#### Final Performance Metrics:
| Model | Accuracy | Macro F1-Score | ROC-AUC |
| :--- | :--- | :--- | :--- |
| Naive Bayes | 75.69% | 0.6356 | 0.8837 |
| **Linear SVM** | **79.09%** | **0.7232** | **0.8946** |

*Note: ROC-AUC for the SVM was calculated using a calibrated classifier to obtain valid probability scores.*

## Key Business Insights
The pipeline generates several visualizations for stakeholder decision-making:
*   **Net Brand Sentiment Score:** A normalized metric showing US Airways and American Airlines with the lowest perceived brand health.
*   **Negative Reason Breakdown:** Identifies "Customer Service Issues" and "Late Flights" as the primary drivers of negative feedback across all brands.
*   **Temporal Sentiment Trends:** Line charts mapping sentiment spikes over time, enabling airlines to correlate service failures with specific dates.

## Deployment & Future Work
*   **Productionization:** The next phase involves wrapping the `clean_text` function and the saved `champion_model.pkl` into a **FastAPI** application supported by a **Streamlit** dashboard for live tweet monitoring.
*   **Advanced Features:** Integration of temporal metadata (e.g., `hour_of_day`) into a `ColumnTransformer` architecture to improve context-aware predictions.
*   **Unit Testing:** The system currently includes preprocessing and performance regression unit tests to ensure long-term stability.

## Installation
The project requires the following libraries:
```bash
pip install nltk scikit-learn pandas numpy matplotlib seaborn wordcloud vaderSentiment contractions
```

## Author
Philip Alumeta 
*Final Year Project*
[JKUAT]
