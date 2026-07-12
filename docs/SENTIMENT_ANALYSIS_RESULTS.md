# Sentiment analysis: method comparison

## TL;DR

**Use Gemini (`gemini-flash-lite-latest`).** It has the highest macro F1
(0.87) of the methods evaluated and, critically, the best recall on the
negative and neutral minority classes — the reviews a review-analysis
product most needs to surface. VADER is the better fallback of the two free
lexicon options if API cost/quota/latency rules Gemini out.

| Method | Accuracy | Macro F1 | Recall: positive | Recall: neutral | Recall: negative |
|---|---|---|---|---|---|
| **Gemini (gemini-flash-lite-latest)** | 92.3% | **0.87** | 97.2% | 63.0% | 98.1% |
| Rating-derived (naive baseline) | 83.5% | 0.69 | 98.9% | 18.9% | 87.1% |
| VADER | 69.5% | 0.55 | 89.9% | 11.9% | 59.8% |
| TextBlob | 60.9% | 0.49 | 84.7% | 30.8% | 29.7% |

Re-run `scripts/sentiment_eval/evaluate.py` and re-check this table after any
change to the app mix or model version — don't assume the ranking holds.

## Why accuracy alone doesn't work here

The golden set's class distribution is **56% positive / 29% negative / 15%
neutral** — imbalanced by design (see [Dataset](#dataset) below). A
classifier that always guessed "positive" would score 56% accuracy while
having zero ability to detect a negative or neutral review, so accuracy
alone rewards approaches that lean on the majority class and hides how they
do on the classes that actually matter for surfacing unhappy users.

The summary table instead ranks by **macro F1** (the unweighted average of
each class's F1 score, so a method can't hide a bad minority-class score
behind a large majority-class one) and breaks out **recall per class**, so
it's visible at a glance which method actually catches negative and neutral
reviews rather than just being "usually right because most reviews are
positive."

## Dataset

- **1500 real App Store reviews** across 3 apps, US store, 500 most-recent
  reviews per app (Apple's RSS review feed hard-caps at 500 reviews/app):
  - Nebula: Spiritual Guidance (id `1459969523`) — astrology/lifestyle, skews heavily positive
  - Robinhood: Trade Anything (id `938003185`) — finance, skews negative/complaint-heavy
  - Duolingo: Language Lessons (id `570060128`) — education, broad mix
- Mixing three apps from different categories gives a more realistic class
  distribution than any single app would (a single-app set skews however
  that app's user base skews).
- Each review's title + content was read and manually labeled
  positive/negative/neutral based on the sentiment expressed in the text,
  **independent of the star rating** — see
  `scripts/sentiment_eval/build_golden_dataset.py` for the full label
  mapping and the labeling rules used.
- This independence is also what lets a **naive rating-derived baseline**
  be compared on equal footing: 4-5★ → positive, 3★ → neutral, 1-2★ →
  negative. That baseline agrees with the manual text labels 83.5% of the
  time — the ceiling a "just use the star rating, skip sentiment analysis
  entirely" approach would hit on this set.

## Per-method results

### Gemini (gemini-flash-lite-latest)

Accuracy vs. manual golden labels: **92.3%** · Agreement with star-rating-derived label: **84.9%**

```
              precision    recall  f1-score   support

    positive       0.96      0.97      0.97       845
     neutral       0.82      0.63      0.71       227
    negative       0.89      0.98      0.93       428

    accuracy                           0.92      1500
   macro avg       0.89      0.86      0.87      1500
weighted avg       0.92      0.92      0.92      1500
```

Confusion matrix (rows = manual label, columns = predicted):

| actual \ predicted | positive | neutral | negative |
|---|---|---|---|
| positive | 821 | 23 | 1 |
| neutral | 34 | 143 | 50 |
| negative | 0 | 8 | 420 |

Reads the review the way a human annotator does, so it handles implied
sentiment and mixed/neutral cases far better than a lexicon score — its one
soft spot is neutral recall (63%): mixed reviews get pushed toward negative
more often than toward the correct neutral bucket. Example disagreements:

- "good but could use improvements" (manual: neutral, predicted: negative, 3★): *"I really liked someone the psychic I met on here there's like two or three that stand out to me the most but my problem is mostly on the app"*
- "Accurate, sometimes slow" (manual: neutral, predicted: negative, 4★): *"I think that the readings are pretty spot on, but sometimes they take a little longer to respond, so your time is running, but you're not ge[tting a full session]"*
- "age?" (manual: neutral, predicted: negative, 3★): *"The app says 16+ on the app store, although it won't let me click my age because I am not over 18."*

Cost/latency: batching 150 reviews per request and firing all requests
concurrently keeps this to ~10 requests for the whole 1500-review set. Needs
an API key and network access; outside the free tier, per-request billing.

### VADER

Accuracy vs. manual golden labels: **69.5%** · Agreement with star-rating-derived label: **75.0%**

```
              precision    recall  f1-score   support

    positive       0.71      0.90      0.80       845
     neutral       0.21      0.12      0.15       227
    negative       0.83      0.60      0.70       428

    accuracy                           0.70      1500
   macro avg       0.59      0.54      0.55      1500
weighted avg       0.67      0.70      0.67      1500
```

Confusion matrix (rows = manual label, columns = predicted):

| actual \ predicted | positive | neutral | negative |
|---|---|---|---|
| positive | 760 | 67 | 18 |
| neutral | 167 | 27 | 33 |
| negative | 137 | 35 | 256 |

Lexicon-based, deterministic, free, and scores all 1500 reviews in well
under a second with no network call — but it only sees word-level polarity,
so it struggles with reviews that are short, sarcastic, or imply sentiment
without using charged words (e.g. "the app used to be free" reads as clearly
negative to a human but has no negative-polarity words for a lexicon
scorer). Recall on neutral is particularly weak (12%): VADER pulls most
mixed/vague reviews toward positive instead. Example disagreements:

- "The Promise of" (manual: neutral, predicted: positive, 5★): *"Hope"*
- "Clarity" (manual: neutral, predicted: positive, 5★): *"Site"*
- "used to be the best app ever☹️" (manual: negative, predicted: positive, 2★): *"ive had nebula for years, it used to be so personalized and so amazing and accurate and now everything is kind if generic..."*

### TextBlob

Accuracy vs. manual golden labels: **60.9%** · Agreement with star-rating-derived label: **64.6%**

```
              precision    recall  f1-score   support

    positive       0.74      0.85      0.79       845
     neutral       0.18      0.31      0.23       227
    negative       0.85      0.30      0.44       428

    accuracy                           0.61      1500
   macro avg       0.59      0.48      0.49      1500
weighted avg       0.69      0.61      0.61      1500
```

Confusion matrix (rows = manual label, columns = predicted):

| actual \ predicted | positive | neutral | negative |
|---|---|---|---|
| positive | 716 | 121 | 8 |
| neutral | 143 | 70 | 14 |
| negative | 110 | 191 | 127 |

Same profile as VADER (free, deterministic, word-level) but noticeably
weaker on negative recall (30%) — it frequently reads a negative review as
neutral rather than negative, which is the worst kind of miss for a
review-triage tool (an unhappy user gets bucketed as "nothing to see here").
Example disagreements:

- "Poor customer support" (manual: negative, predicted: neutral, 2★): *"I signed up for a trial period but changed my mind immediately and canceled. A week later my credit card was charged..."*
- "Its chill" (manual: neutral, predicted: positive, 4★): *"No clue if its accurate or not we will see august"*

## Caveats

- **Golden labels were assigned by an LLM (Claude) reading each review's
  title + content**, not by an independent human annotator. Treat this as a
  reasonable proxy for human judgment, not an unimpeachable ground truth. In
  particular, this makes the comparison partly LLM-vs-LLM for the Gemini
  row, which likely flatters Gemini's score somewhat relative to what a
  fully independent human-labeled set would show.
- No leakage: `manual_sentiment` (the gold label) is only ever read as the
  ground-truth column in `evaluate.py` — VADER, TextBlob, and Gemini are all
  scored from `title` + `content` alone, never from the label or the star
  rating.
