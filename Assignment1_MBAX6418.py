#Step #1

from openai import OpenAI

# Connect to the class LLM endpoint
client = OpenAI(
    base_url="http://dobolyi.com:9001/v1",
    api_key="6418"
)

MODEL_NAME = "cyankiwi/Qwen3.6-35B-A3B-AWQ-4bit"


def build_sentiment_prompt(title, text):
    return f"""
You are a sentiment classifier analyzing Amazon Gift Card reviews.

Classify the overall sentiment of the review as either:
POSITIVE
NEGATIVE

Rules:
- POSITIVE means the reviewer is generally satisfied, pleased,
  approving, or recommends the product.
- NEGATIVE means the reviewer is generally dissatisfied,
  disappointed, frustrated, critical, or warns against the product.
- If the title and review text conflict, consider the entire review
  and give more weight to the detailed review text.
- If the review contains both positive and negative comments,
  classify it according to its overall sentiment.
- For very short reviews, classify only from the sentiment expressed.
- Do not assume information that is not present.
- Do not use or infer a star rating.

Review Title:
{title}

Review Text:
{text}

Return exactly one word:
POSITIVE or NEGATIVE
"""


def classify_sentiment(title, text):
    prompt = build_sentiment_prompt(title, text)

    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {"role": "user", "content": prompt}
        ],
        temperature=0
    )

    return response.choices[0].message.content.strip().upper()


# Step 1 spot checks
positive_result = classify_sentiment(
    "Perfect gift!",
    "This gift card worked perfectly and was very easy to use. "
    "I would definitely buy it again."
)

negative_result = classify_sentiment(
    "Terrible experience",
    "The gift card did not work and customer service was no help. "
    "I am very disappointed."
)

print("Positive test prediction:", positive_result)
print("Negative test prediction:", negative_result)





#Step 2A Loading first 100 reviews

import pandas as pd
import time

DATA_URL = (
    "https://mcauleylab.ucsd.edu/public_datasets/data/"
    "amazon_2023/raw/review_categories/Gift_Cards.jsonl.gz"
)

# Load the first 100 Amazon Gift Card reviews
reviews = pd.read_json(
    DATA_URL,
    lines=True,
    compression="gzip",
    nrows=100
)

print("Reviews loaded:", len(reviews))
print("\nColumns:")
print(reviews.columns.tolist())

print("\nFirst 5 reviews:")
print(reviews[["rating", "title", "text"]].head())


#Step 2B Create the Correct Answers

# Create the rating-based correct answer.
# IMPORTANT: This is used only AFTER prediction.
# The LLM never receives this value.

reviews["actual_sentiment"] = reviews["rating"].apply(
    lambda rating: "POSITIVE" if rating >= 4 else "NEGATIVE"
)

print("\nRating-based classes:")
print(reviews["actual_sentiment"].value_counts())




#Step 2C Classifying all 100 reviews

predictions = []

print("\nStarting classification of 100 reviews...\n")

for index, row in reviews.iterrows():

    title = "" if pd.isna(row["title"]) else str(row["title"])
    text = "" if pd.isna(row["text"]) else str(row["text"])

    try:
        prediction = classify_sentiment(title, text)

    except Exception as error:
        print(f"Error on review {index + 1}: {error}")
        prediction = "ERROR"

    predictions.append(prediction)

    print(
        f"Review {index + 1:3}/100 | "
        f"Predicted: {prediction}"
    )

    # Small pause so we don't hammer the class endpoint
    time.sleep(0.1)

reviews["predicted_sentiment"] = predictions


#Step 2D Correct Predictions


reviews["correct"] = (
    reviews["predicted_sentiment"]
    == reviews["actual_sentiment"]
)

valid_reviews = reviews[
    reviews["predicted_sentiment"].isin(["POSITIVE", "NEGATIVE"])
].copy()

accuracy = valid_reviews["correct"].mean() * 100

print("\n" + "=" * 50)
print("STEP 2 RESULTS")
print("=" * 50)

print(f"Valid predictions: {len(valid_reviews)} / 100")
print(f"Overall accuracy: {accuracy:.1f}%")





#Step 2E Performance by class

print("\nActual class distribution:")
print(valid_reviews["actual_sentiment"].value_counts())

print("\nAccuracy by actual class:")

class_accuracy = (
    valid_reviews.groupby("actual_sentiment")["correct"]
    .mean()
    .mul(100)
)

print(class_accuracy.round(1))


print("\nPrediction confusion matrix:")

confusion_matrix = pd.crosstab(
    valid_reviews["actual_sentiment"],
    valid_reviews["predicted_sentiment"],
    rownames=["Actual"],
    colnames=["Predicted"]
)

print(confusion_matrix)

#Step 2F Find mistakes 

mistakes = valid_reviews[
    valid_reviews["correct"] == False
]

print("\nNumber of incorrect predictions:", len(mistakes))

print("\nMisclassified reviews:")

for index, row in mistakes.iterrows():
    print("\n" + "-" * 60)
    print("Rating:", row["rating"])
    print("Title:", row["title"])
    print("Text:", row["text"])
    print("Expected:", row["actual_sentiment"])
    print("Predicted:", row["predicted_sentiment"])



#Step 2G Save Results 

OUTPUT_FILE = "initial_100_results.csv"

reviews.to_csv(
    OUTPUT_FILE,
    index=False
)

print(f"\nResults saved to: {OUTPUT_FILE}")


#Step 4

