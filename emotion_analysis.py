#Step 5D-NRC

import pandas as pd
import re
import urllib.request
from collections import defaultdict
from pathlib import Path


# ============================================================
# STEP 5D - NRC WORD-LIST EMOTION ANALYSIS
# ============================================================

INPUT_FILE = "/Users/nisajovel/step5_emotion_results.csv"

NRC_FILE = "/Users/nisajovel/NRC-Emotion-Lexicon-Wordlevel-v0.92.txt"

OUTPUT_FILE = "/Users/nisajovel/step5_final_results.csv"


NRC_URL = (
    "https://raw.githubusercontent.com/"
    "Franck-Dernoncourt/NRC_Emotion_Lexicon/master/"
    "NRC-emotion-lexicon-wordlevel-alphabetized-v0.92.txt"
)


EMOTIONS = [
    "anger",
    "anticipation",
    "disgust",
    "fear",
    "joy",
    "sadness",
    "surprise",
    "trust"
]


# ============================================================
# DOWNLOAD NRC LEXICON IF NEEDED
# ============================================================

if not Path(NRC_FILE).exists():

    print("NRC lexicon not found.")
    print("Downloading NRC Emotion Lexicon...")

    urllib.request.urlretrieve(
        NRC_URL,
        NRC_FILE
    )

    print("NRC lexicon downloaded successfully.")

else:

    print("NRC lexicon already exists.")


# ============================================================
# LOAD NRC LEXICON
# ============================================================

def load_nrc_lexicon(filename):

    lexicon = defaultdict(set)

    with open(
        filename,
        "r",
        encoding="utf-8"
    ) as file:

        for line in file:

            parts = line.strip().split("\t")

            if len(parts) != 3:
                continue

            word, emotion, association = parts

            if (
                emotion in EMOTIONS
                and association == "1"
            ):
                lexicon[word.lower()].add(emotion)

    return lexicon


print("\nLoading NRC emotion lexicon...")

lexicon = load_nrc_lexicon(NRC_FILE)

print(
    f"NRC words loaded: {len(lexicon):,}"
)


# ============================================================
# NRC EMOTION SCORING FUNCTION
# ============================================================

def get_nrc_emotion(text):

    scores = {
        emotion: 0
        for emotion in EMOTIONS
    }

    words = re.findall(
        r"[A-Za-z']+",
        str(text).lower()
    )

    for word in words:

        if word in lexicon:

            for emotion in lexicon[word]:
                scores[emotion] += 1

    highest_score = max(scores.values())

    # No emotion words were found
    if highest_score == 0:
        return "none"

    winners = [
        emotion
        for emotion, score in scores.items()
        if score == highest_score
    ]

    # Fixed tie-breaking for reproducibility
    return sorted(winners)[0]


# ============================================================
# LOAD STEP 5 LLM RESULTS
# ============================================================

reviews = pd.read_csv(INPUT_FILE)

reviews["combined_review"] = (
    reviews["title"].fillna("")
    + " "
    + reviews["text"].fillna("")
)


# ============================================================
# CALCULATE NRC EMOTIONS
# ============================================================

print("\nCalculating NRC emotions...\n")

reviews["nrc_emotion"] = (
    reviews["combined_review"]
    .apply(get_nrc_emotion)
)


# ============================================================
# COMPARE LLM VS NRC
# ============================================================

reviews["emotion_agreement"] = (
    reviews["llm_emotion"]
    == reviews["nrc_emotion"]
)

agreement_rate = (
    reviews["emotion_agreement"].mean()
    * 100
)


print("=" * 60)
print("STEP 5 EMOTION RESULTS")
print("=" * 60)

print(
    f"\nLLM / NRC agreement: "
    f"{agreement_rate:.1f}%"
)


print("\nLLM emotion counts:")

print(
    reviews[
        "llm_emotion"
    ].value_counts()
)


print("\nNRC emotion counts:")

print(
    reviews[
        "nrc_emotion"
    ].value_counts()
)


print("\nLLM vs NRC comparison:")

comparison = pd.crosstab(
    reviews["llm_emotion"],
    reviews["nrc_emotion"],
    rownames=["LLM Emotion"],
    colnames=["NRC Emotion"]
)

print(comparison)


# ============================================================
# SHOW DISAGREEMENTS
# ============================================================

disagreements = reviews[
    reviews["emotion_agreement"] == False
]

print(
    f"\nNumber of disagreements: "
    f"{len(disagreements)}"
)


print("\nExample disagreements:")

for _, row in disagreements.head(10).iterrows():

    print("\n" + "-" * 60)

    print(
        "Title:",
        row["title"]
    )

    print(
        "LLM Emotion:",
        row["llm_emotion"]
    )

    print(
        "NRC Emotion:",
        row["nrc_emotion"]
    )


# ============================================================
# SAVE STEP 5 FINAL RESULTS
# ============================================================

reviews.to_csv(
    OUTPUT_FILE,
    index=False
)

print(
    f"\nStep 5 final results saved to:"
    f"\n{OUTPUT_FILE}"
)

#Step #6

import pandas as pd
import json
import time
from openai import OpenAI

# ============================================================
# STEP 6 - BALANCED THREE-CLASS SCORING
# ============================================================

DATA_URL = (
    "https://mcauleylab.ucsd.edu/public_datasets/data/"
    "amazon_2023/raw/review_categories/Gift_Cards.jsonl.gz"
)

INPUT_FILE = "/Users/nisajovel/balanced_results.csv"
OUTPUT_FILE = "/Users/nisajovel/step5_final_results.csv"
RANDOM_SEED = 6418
SAMPLE_PER_CLASS = 50

client = OpenAI(
    base_url="http://dobolyi.com:9001/v1",
    api_key="6418"
)

MODEL_NAME = "cyankiwi/Qwen3.6-35B-A3B-AWQ-4bit"


def rating_to_class(rating):
    if rating >= 4:
        return "POSITIVE"
    elif rating == 3:
        return "NEUTRAL"
    else:
        return "NEGATIVE"


def build_prompt(title, text):
    return f"""
You are analyzing an Amazon Gift Card review.

Classify the review into exactly ONE sentiment:

POSITIVE
NEUTRAL
NEGATIVE

Definitions:

POSITIVE:
The reviewer is generally satisfied, pleased, approving,
or recommends the product.

NEUTRAL:
The reviewer is mixed, moderate, uncertain, or neither
clearly positive nor clearly negative.

NEGATIVE:
The reviewer is generally dissatisfied, disappointed,
frustrated, critical, or warns against the product.

Also identify exactly ONE primary emotion from:

anger
anticipation
disgust
fear
joy
sadness
surprise
trust

Important:
- Use only the review title and text.
- Do not use or infer the star rating.
- If title and body conflict, consider the review as a whole.

Return ONLY valid JSON like this:

{{
    "sentiment": "POSITIVE",
    "emotion": "joy"
}}

Review Title:
{title}

Review Text:
{text}
"""


def classify_review(title, text):

    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {
                "role": "user",
                "content": build_prompt(title, text)
            }
        ],
        temperature=0
    )

    raw = response.choices[0].message.content.strip()

    # Remove accidental markdown formatting if model adds it
    raw = raw.replace("```json", "").replace("```", "").strip()

    result = json.loads(raw)

    return (
        result["sentiment"].upper(),
        result["emotion"].lower()
    )


print("Loading Amazon Gift Cards dataset...")

df = pd.read_json(
    DATA_URL,
    lines=True,
    compression="gzip"
)

df["actual_sentiment"] = df["rating"].apply(rating_to_class)

print("\nFull dataset class distribution:")
print(df["actual_sentiment"].value_counts())


# ============================================================
# BALANCED SAMPLE: 50 FROM EACH CLASS
# ============================================================

balanced = (
    df.groupby("actual_sentiment", group_keys=False)
    .sample(
        n=SAMPLE_PER_CLASS,
        random_state=RANDOM_SEED
    )
    .reset_index(drop=True)
)

# Shuffle while remaining reproducible
balanced = balanced.sample(
    frac=1,
    random_state=RANDOM_SEED
).reset_index(drop=True)

print("\nBalanced sample:")
print(balanced["actual_sentiment"].value_counts())


# ============================================================
# MODEL PREDICTIONS
# ============================================================

predictions = []
emotions = []

print("\nStarting balanced classification...\n")

for index, row in balanced.iterrows():

    title = "" if pd.isna(row["title"]) else str(row["title"])
    text = "" if pd.isna(row["text"]) else str(row["text"])

    try:
        prediction, emotion = classify_review(title, text)

    except Exception as error:
        print(f"Error on review {index + 1}: {error}")
        prediction = "ERROR"
        emotion = "ERROR"

    predictions.append(prediction)
    emotions.append(emotion)

    print(
        f"Review {index + 1:3}/150 | "
        f"Actual: {row['actual_sentiment']:<8} | "
        f"Predicted: {prediction:<8} | "
        f"Emotion: {emotion}"
    )

    time.sleep(0.1)


balanced["predicted_sentiment"] = predictions
balanced["llm_emotion"] = emotions

balanced["correct"] = (
    balanced["actual_sentiment"]
    == balanced["predicted_sentiment"]
)


# ============================================================
# RESULTS
# ============================================================

valid = balanced[
    balanced["predicted_sentiment"].isin(
        ["POSITIVE", "NEUTRAL", "NEGATIVE"]
    )
].copy()

accuracy = valid["correct"].mean() * 100

print("\n" + "=" * 60)
print("STEP 6 - BALANCED THREE-CLASS RESULTS")
print("=" * 60)

print(f"\nValid predictions: {len(valid)} / 150")
print(f"Overall accuracy: {accuracy:.1f}%")

print("\nAccuracy by class:")

class_accuracy = (
    valid.groupby("actual_sentiment")["correct"]
    .mean()
    .mul(100)
    .round(1)
)

print(class_accuracy)

print("\nConfusion matrix:")

confusion = pd.crosstab(
    valid["actual_sentiment"],
    valid["predicted_sentiment"],
    rownames=["Actual"],
    colnames=["Predicted"]
)

print(confusion)


# ============================================================
# SAVE
# ============================================================

balanced.to_csv(
    OUTPUT_FILE,
    index=False
)

print(f"\nSaved: {OUTPUT_FILE}")

