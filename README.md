# Spam Message Detector (NLP)

Classifies a text message as **spam** or **ham** (genuine) using TF-IDF and machine learning.
Built during the YBI Foundation Project Workshop (internship at YantraByte) and extended with my own experiments.

## Dataset
- 1,400 messages (1,050 ham, 350 spam), from the YBI Foundation workshop dataset
- No missing values; 912 unique messages after cleaning (the rest are repeats)

## Pipeline
Message → text cleaning (lowercase, links and numbers replaced) → TF-IDF (unigrams + bigrams) → Multinomial Naive Bayes → spam/ham

## Results
- Test accuracy: 100% (280 held-out messages)
- 5-fold cross-validation: 1.0 on every fold
- Naive Bayes and Logistic Regression, with 3 feature sizes and 2 n-gram settings (12 configurations): all scored 100%
- Duplicate check: [write your result from Step 2 here]

## Own tests
I wrote 6 messages to test the model on realistic inputs: it got **2 of 6** correct. It flagged every message containing words like "free", "reward" or "offer" as spam, including genuine-sounding ones. [Add your second round here if you do one.]

## Limitations
- The dataset appears small and repetitive, so 100% accuracy is probably inflated and cannot separate good models from bad ones
- The model seems to rely on trigger words rather than context, which can cause false positives
- My own test set is small (6 messages), so these results are indicative, not conclusive

## Files
- `Project_Workshop_Student.ipynb`: full notebook
- `screenshots/`: results

## Tools
Python, pandas, scikit-learn, matplotlib
