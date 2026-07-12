# Sentiment analysis approach comparison

Golden dataset: 100 real App Store reviews (app id 1459969523,
"Nebula", US store), manually labeled positive/negative/neutral by reading
each review's title + content (see
`scripts/sentiment_eval/build_golden_dataset.py` for the label mapping and
labeling rules). Labels reflect the sentiment expressed in the text, not the
star rating -- so they can also be compared against a naive
rating-derived label (4-5★ -> positive, 3★ -> neutral, 1-2★ -> negative).

Naive rating-derived label vs. manual text label agreement: **92.0%**
(this is the ceiling a "just use the star rating" approach would hit on this
set, and the baseline every real approach below is compared against).

## VADER

Accuracy vs. manual golden labels: **86.0%**
Agreement with star-rating-derived label: **87.0%**

```
              precision    recall  f1-score   support

    positive       0.90      0.95      0.93        80
     neutral       0.50      0.22      0.31         9
    negative       0.67      0.73      0.70        11

    accuracy                           0.86       100
   macro avg       0.69      0.63      0.64       100
weighted avg       0.84      0.86      0.85       100
```

Confusion matrix (rows = manual label, columns = predicted):

| actual \ predicted | positive | neutral | negative |
|---|---|---|---|
| positive | 76 | 1 | 3 |
| neutral | 6 | 2 | 1 |
| negative | 2 | 1 | 8 |

Example disagreements (predicted vs. manual):

- "good but could use improvements" (manual: neutral, predicted: positive, rating: 3★): 'I really liked someone the psychic I met on here there’s like two or three that stand out to me the most but my problem is mostly on the app'
- "Clarity" (manual: neutral, predicted: positive, rating: 5★): 'Site'
- "Accurate, sometimes slow" (manual: neutral, predicted: positive, rating: 4★): 'I think that the readings are pretty spot on, but sometimes they take a little longer to respond, so your time is running, but you’re not ge'
- "used to be the best app ever☹️" (manual: negative, predicted: positive, rating: 2★): 'ive had nebula for years, it used to be so personalized and so amazing and accurate and now everything is kind if generic and things you use'
- "Its chill" (manual: neutral, predicted: negative, rating: 4★): 'No clue if its accurate or not we will see august'

## TextBlob

Accuracy vs. manual golden labels: **80.0%**
Agreement with star-rating-derived label: **76.0%**

```
              precision    recall  f1-score   support

    positive       0.89      0.90      0.89        80
     neutral       0.33      0.56      0.42         9
    negative       0.75      0.27      0.40        11

    accuracy                           0.80       100
   macro avg       0.66      0.58      0.57       100
weighted avg       0.82      0.80      0.80       100
```

Confusion matrix (rows = manual label, columns = predicted):

| actual \ predicted | positive | neutral | negative |
|---|---|---|---|
| positive | 72 | 7 | 1 |
| neutral | 4 | 5 | 0 |
| negative | 5 | 3 | 3 |

Example disagreements (predicted vs. manual):

- "The Promise of" (manual: positive, predicted: neutral, rating: 5★): 'Hope'
- "good but could use improvements" (manual: neutral, predicted: positive, rating: 3★): 'I really liked someone the psychic I met on here there’s like two or three that stand out to me the most but my problem is mostly on the app'
- "Poor customer support" (manual: negative, predicted: neutral, rating: 2★): 'I signed up for a trial  period  but changed my mind immediately and canceled. A week later my credit card was charged. Emailed customer ser'
- "Accurate, sometimes slow" (manual: neutral, predicted: positive, rating: 4★): 'I think that the readings are pretty spot on, but sometimes they take a little longer to respond, so your time is running, but you’re not ge'
- "used to be the best app ever☹️" (manual: negative, predicted: positive, rating: 2★): 'ive had nebula for years, it used to be so personalized and so amazing and accurate and now everything is kind if generic and things you use'

## Gemini (gemini-flash-latest)

Accuracy vs. manual golden labels: **99.0%**
Agreement with star-rating-derived label: **91.0%**

```
              precision    recall  f1-score   support

    positive       1.00      0.99      0.99        80
     neutral       0.90      1.00      0.95         9
    negative       1.00      1.00      1.00        11

    accuracy                           0.99       100
   macro avg       0.97      1.00      0.98       100
weighted avg       0.99      0.99      0.99       100
```

Confusion matrix (rows = manual label, columns = predicted):

| actual \ predicted | positive | neutral | negative |
|---|---|---|---|
| positive | 79 | 1 | 0 |
| neutral | 0 | 9 | 0 |
| negative | 0 | 0 | 11 |

Example disagreements (predicted vs. manual):

- "The Promise of" (manual: positive, predicted: neutral, rating: 5★): 'Hope'


## Discussion

- **VADER / TextBlob** are lexicon-based, deterministic, free, and score all
  100 reviews in well under a second with no network call -- but they only
  see word-level polarity, so they struggle with reviews that are short,
  sarcastic, or where sentiment is implied rather than stated in charged
  words (e.g. "the app used to be free" carries clear negative sentiment to
  a human reader but no negative-polarity words to a lexicon scorer).
- **Gemini** reads the review the way a human annotator does, so it should
  handle implied sentiment and mixed/neutral cases better, at the cost of
  needing an API key, network calls, and (outside the free tier) per-request
  billing. Batching ~20 reviews per request keeps this to ~5 requests for
  the whole set.
- Both lexicon approaches and the LLM approach are being measured against
  labels I (an LLM) assigned by hand -- see the caveat in
  `scripts/sentiment_eval/build_golden_dataset.py`. Treat this as a
  reasonable proxy for human judgment, not as an unimpeachable ground truth.
