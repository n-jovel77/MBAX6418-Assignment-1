import pandas as pd
from pathlib import Path
import html

# ============================================================
# STEP 7 - FINAL BALANCED DASHBOARD
# MBAX 6418 Assignment 1
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

INPUT_FILE = "/Users/nisajovel/step5_final_results.csv"
OUTPUT_FILE = "/Users/nisajovel/dashboard.html"

# ============================================================
# LOAD FINAL BALANCED RESULTS
# ============================================================

df = pd.read_csv(INPUT_FILE)

CLASSES = ["POSITIVE", "NEUTRAL", "NEGATIVE"]

# Keep only valid three-class predictions
valid = df[
    df["predicted_sentiment"].isin(CLASSES)
].copy()

# Recalculate correctness so it is always reliable
valid["correct"] = (
    valid["actual_sentiment"]
    == valid["predicted_sentiment"]
)


# ============================================================
# OVERALL METRICS
# ============================================================

total_reviews = len(valid)
correct_reviews = int(valid["correct"].sum())
incorrect_reviews = total_reviews - correct_reviews

accuracy = (
    correct_reviews / total_reviews * 100
    if total_reviews > 0
    else 0
)


# ============================================================
# CLASS ACCURACY
# ============================================================

class_accuracy = (
    valid.groupby("actual_sentiment")["correct"]
    .mean()
    .mul(100)
)

positive_accuracy = float(
    class_accuracy.get("POSITIVE", 0)
)

neutral_accuracy = float(
    class_accuracy.get("NEUTRAL", 0)
)

negative_accuracy = float(
    class_accuracy.get("NEGATIVE", 0)
)


# ============================================================
# ACTUAL VS PREDICTED COUNTS
# ============================================================

actual_counts = valid[
    "actual_sentiment"
].value_counts()

predicted_counts = valid[
    "predicted_sentiment"
].value_counts()


# ============================================================
# STAR DISTRIBUTION
# ============================================================

star_counts = (
    valid["rating"]
    .value_counts()
    .sort_index()
)


# ============================================================
# CONFUSION MATRIX
# ============================================================

confusion = pd.crosstab(
    valid["actual_sentiment"],
    valid["predicted_sentiment"],
    rownames=["Actual"],
    colnames=["Predicted"]
)

# Ensure all classes appear
confusion = confusion.reindex(
    index=CLASSES,
    columns=CLASSES,
    fill_value=0
)


# ============================================================
# EMOTION RESULTS
# ============================================================

if (
    "llm_emotion" in valid.columns
    and "nrc_emotion" in valid.columns
):

    valid["emotion_agreement"] = (
        valid["llm_emotion"].astype(str).str.lower()
        ==
        valid["nrc_emotion"].astype(str).str.lower()
    )

    emotion_agreement = (
        valid["emotion_agreement"].mean()
        * 100
    )

else:
    emotion_agreement = 0


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def safe_text(value):
    if pd.isna(value):
        return ""
    return html.escape(str(value))


def sentiment_badge(sentiment):

    sentiment = str(sentiment).upper()

    if sentiment == "POSITIVE":
        css_class = "positive"

    elif sentiment == "NEUTRAL":
        css_class = "neutral"

    else:
        css_class = "negative"

    return (
        f'<span class="badge {css_class}">'
        f'{safe_text(sentiment)}'
        f'</span>'
    )


def emotion_badge(emotion):

    if pd.isna(emotion):
        return ""

    return (
        f'<span class="badge emotion">'
        f'{safe_text(str(emotion).title())}'
        f'</span>'
    )


def result_badge(correct):

    if bool(correct):
        return (
            '<span class="badge correct">'
            'CORRECT'
            '</span>'
        )

    return (
        '<span class="badge incorrect">'
        'MISMATCH'
        '</span>'
    )


# ============================================================
# REVIEW TABLE
# ============================================================

review_rows = ""

for _, row in valid.iterrows():

    title = safe_text(
        row.get("title", "")
    )

    text = safe_text(
        row.get("text", "")
    )

    if len(text) > 250:
        text = text[:250] + "..."

    result_type = (
        "correct"
        if row["correct"]
        else "mismatch"
    )

    llm_emotion = row.get(
        "llm_emotion",
        ""
    )

    nrc_emotion = row.get(
        "nrc_emotion",
        ""
    )

    review_rows += f"""
    <tr data-result="{result_type}">
        <td>{int(row["rating"])}</td>

        <td>
            <strong>{title}</strong>
            <div class="review-text">
                {text}
            </div>
        </td>

        <td>
            {sentiment_badge(row["actual_sentiment"])}
        </td>

        <td>
            {sentiment_badge(row["predicted_sentiment"])}
        </td>

        <td>
            {emotion_badge(llm_emotion)}
        </td>

        <td>
            {emotion_badge(nrc_emotion)}
        </td>

        <td>
            {result_badge(row["correct"])}
        </td>
    </tr>
    """


# ============================================================
# CLASS ACCURACY BARS
# ============================================================

class_accuracy_html = ""

for sentiment in CLASSES:

    value = float(
        class_accuracy.get(
            sentiment,
            0
        )
    )

    class_accuracy_html += f"""
    <div class="bar-row">

        <div class="bar-label">
            <span>{sentiment}</span>
            <strong>{value:.1f}%</strong>
        </div>

        <div class="bar-background">
            <div
                class="bar accuracy-bar"
                style="width:{value}%">
            </div>
        </div>

    </div>
    """


# ============================================================
# ACTUAL VS PREDICTED COUNTS
# ============================================================

comparison_html = ""

max_count = max(
    list(actual_counts.values)
    + list(predicted_counts.values)
    + [1]
)

for sentiment in CLASSES:

    actual = int(
        actual_counts.get(
            sentiment,
            0
        )
    )

    predicted = int(
        predicted_counts.get(
            sentiment,
            0
        )
    )

    actual_width = (
        actual / max_count * 100
    )

    predicted_width = (
        predicted / max_count * 100
    )

    comparison_html += f"""
    <div class="comparison-group">

        <h4>{sentiment}</h4>

        <div class="bar-label">
            <span>Actual</span>
            <strong>{actual}</strong>
        </div>

        <div class="bar-background">
            <div
                class="bar actual-bar"
                style="width:{actual_width}%">
            </div>
        </div>

        <div class="bar-label second-label">
            <span>Predicted</span>
            <strong>{predicted}</strong>
        </div>

        <div class="bar-background">
            <div
                class="bar predicted-bar"
                style="width:{predicted_width}%">
            </div>
        </div>

    </div>
    """


# ============================================================
# STAR-RATING DISTRIBUTION
# ============================================================

star_html = ""

max_star_count = max(
    star_counts.max(),
    1
)

for rating, count in star_counts.items():

    width = (
        count / max_star_count
        * 100
    )

    star_html += f"""
    <div class="bar-row">

        <div class="bar-label">
            <span>{int(rating)} Star</span>
            <strong>{count}</strong>
        </div>

        <div class="bar-background">
            <div
                class="bar star-bar"
                style="width:{width}%">
            </div>
        </div>

    </div>
    """


# ============================================================
# CONFUSION MATRIX HTML
# ============================================================

confusion_html = f"""
<table class="matrix">

    <thead>
        <tr>
            <th>Actual ↓ / Predicted →</th>
            <th>NEGATIVE</th>
            <th>NEUTRAL</th>
            <th>POSITIVE</th>
        </tr>
    </thead>

    <tbody>

        <tr>
            <th>NEGATIVE</th>
            <td>{confusion.loc["NEGATIVE", "NEGATIVE"]}</td>
            <td>{confusion.loc["NEGATIVE", "NEUTRAL"]}</td>
            <td>{confusion.loc["NEGATIVE", "POSITIVE"]}</td>
        </tr>

        <tr>
            <th>NEUTRAL</th>
            <td>{confusion.loc["NEUTRAL", "NEGATIVE"]}</td>
            <td>{confusion.loc["NEUTRAL", "NEUTRAL"]}</td>
            <td>{confusion.loc["NEUTRAL", "POSITIVE"]}</td>
        </tr>

        <tr>
            <th>POSITIVE</th>
            <td>{confusion.loc["POSITIVE", "NEGATIVE"]}</td>
            <td>{confusion.loc["POSITIVE", "NEUTRAL"]}</td>
            <td>{confusion.loc["POSITIVE", "POSITIVE"]}</td>
        </tr>

    </tbody>

</table>
"""


# ============================================================
# BUILD FINAL HTML
# ============================================================

html_output = f"""
<!DOCTYPE html>

<html lang="en">

<head>

<meta charset="UTF-8">

<meta
    name="viewport"
    content="width=device-width, initial-scale=1.0"
>

<title>
Amazon Gift Card Sentiment Dashboard
</title>

<style>

* {{
    box-sizing: border-box;
}}

body {{
    margin: 0;
    font-family:
        Arial,
        Helvetica,
        sans-serif;
    background: #f5f7fb;
    color: #18212f;
}}

.header {{
    background:
        linear-gradient(
            135deg,
            #18212f,
            #334155
        );
    color: white;
    padding: 45px 7%;
}}

.header h1 {{
    margin: 0;
    font-size: 35px;
}}

.header p {{
    color: #cbd5e1;
    margin-top: 10px;
}}

.container {{
    width: 88%;
    max-width: 1400px;
    margin: 30px auto 60px auto;
}}

.section-title {{
    margin-top: 42px;
    margin-bottom: 18px;
}}

.metrics {{
    display: grid;
    grid-template-columns:
        repeat(5, 1fr);
    gap: 16px;
}}

.card {{
    background: white;
    border-radius: 14px;
    padding: 22px;
    box-shadow:
        0 4px 16px
        rgba(0,0,0,0.06);
}}

.card-label {{
    color: #64748b;
    font-size: 13px;
}}

.card-value {{
    margin-top: 8px;
    font-size: 29px;
    font-weight: 700;
}}

.dashboard-grid {{
    display: grid;
    grid-template-columns:
        1fr 1fr;
    gap: 20px;
}}

.panel {{
    background: white;
    border-radius: 14px;
    padding: 25px;
    box-shadow:
        0 4px 16px
        rgba(0,0,0,0.06);
}}

.bar-row {{
    margin-bottom: 20px;
}}

.bar-label {{
    display: flex;
    justify-content: space-between;
    margin-bottom: 6px;
    font-size: 14px;
}}

.second-label {{
    margin-top: 10px;
}}

.bar-background {{
    background: #e8edf4;
    height: 18px;
    border-radius: 9px;
    overflow: hidden;
}}

.bar {{
    height: 18px;
    border-radius: 9px;
}}

.accuracy-bar {{
    background: #10b981;
}}

.actual-bar {{
    background: #3b82f6;
}}

.predicted-bar {{
    background: #8b5cf6;
}}

.star-bar {{
    background: #f59e0b;
}}

.comparison-group {{
    margin-bottom: 25px;
}}

.matrix {{
    width: 100%;
    border-collapse: collapse;
}}

.matrix th,
.matrix td {{
    border:
        1px solid #e2e8f0;
    padding: 13px;
    text-align: center;
}}

.matrix th {{
    background: #eef2f7;
}}

.filter-area {{
    display: flex;
    align-items: center;
    gap: 10px;
    flex-wrap: wrap;
    margin-bottom: 18px;
}}

.filter-button {{
    border: none;
    padding: 10px 16px;
    border-radius: 9px;
    background: #e8edf4;
    color: #334155;
    font-weight: 600;
    cursor: pointer;
}}

.filter-button:hover {{
    background: #dbe3ee;
}}

.filter-button.active {{
    background: #18212f;
    color: white;
}}

#visible-count {{
    margin-left: auto;
    color: #64748b;
}}

.table-container {{
    background: white;
    border-radius: 14px;
    padding: 20px;
    overflow-x: auto;
    box-shadow:
        0 4px 16px
        rgba(0,0,0,0.06);
}}

table {{
    width: 100%;
    border-collapse: collapse;
}}

th {{
    text-align: left;
    background: #eef2f7;
    padding: 13px;
    font-size: 13px;
}}

td {{
    padding: 13px;
    border-bottom:
        1px solid #edf0f5;
    vertical-align: top;
    font-size: 13px;
}}

.review-text {{
    color: #64748b;
    margin-top: 7px;
    max-width: 550px;
    line-height: 1.4;
}}

.badge {{
    display: inline-block;
    padding: 6px 9px;
    border-radius: 8px;
    font-size: 11px;
    font-weight: bold;
}}

.positive {{
    background: #dcfce7;
    color: #166534;
}}

.neutral {{
    background: #e0e7ff;
    color: #3730a3;
}}

.negative {{
    background: #fee2e2;
    color: #991b1b;
}}

.correct {{
    background: #dbeafe;
    color: #1d4ed8;
}}

.incorrect {{
    background: #fef3c7;
    color: #92400e;
}}

.emotion {{
    background: #f1f5f9;
    color: #334155;
}}

.insight {{
    margin-top: 14px;
    padding: 14px;
    background: #f8fafc;
    border-radius: 10px;
    color: #475569;
    line-height: 1.5;
}}

@media(max-width: 1000px) {{

    .metrics {{
        grid-template-columns:
            repeat(2, 1fr);
    }}

    .dashboard-grid {{
        grid-template-columns: 1fr;
    }}

}}

</style>

</head>


<body>

<div class="header">

    <h1>
        Amazon Gift Card Review Analysis
    </h1>

    <p>
        MBAX 6418 — Balanced Sentiment &
        Emotion Classification
    </p>

</div>


<div class="container">


<h2 class="section-title">
    Model Performance
</h2>

<div class="metrics">

    <div class="card">
        <div class="card-label">
            Reviews Scored
        </div>

        <div class="card-value">
            {total_reviews}
        </div>
    </div>


    <div class="card">
        <div class="card-label">
            Overall Accuracy
        </div>

        <div class="card-value">
            {accuracy:.1f}%
        </div>
    </div>


    <div class="card">
        <div class="card-label">
            Positive Accuracy
        </div>

        <div class="card-value">
            {positive_accuracy:.1f}%
        </div>
    </div>


    <div class="card">
        <div class="card-label">
            Neutral Accuracy
        </div>

        <div class="card-value">
            {neutral_accuracy:.1f}%
        </div>
    </div>


    <div class="card">
        <div class="card-label">
            Negative Accuracy
        </div>

        <div class="card-value">
            {negative_accuracy:.1f}%
        </div>
    </div>

</div>


<h2 class="section-title">
    Class Performance
</h2>

<div class="dashboard-grid">

    <div class="panel">

        <h3>
            Accuracy by Actual Class
        </h3>

        {class_accuracy_html}

        <div class="insight">
            Balanced sampling makes differences
            between classes easier to see.
            Neutral reviews are especially important
            because three-star reviews are treated as
            their own class.
        </div>

    </div>


    <div class="panel">

        <h3>
            Actual vs Predicted
        </h3>

        {comparison_html}

    </div>

</div>


<h2 class="section-title">
    Descriptive Results
</h2>

<div class="dashboard-grid">

    <div class="panel">

        <h3>
            Star Rating Distribution
        </h3>

        {star_html}

    </div>


    <div class="panel">

        <h3>
            LLM vs NRC Emotion Agreement
        </h3>

        <div class="card-value">
            {emotion_agreement:.1f}%
        </div>

        <div class="insight">
            The LLM evaluates overall context,
            while the NRC approach scores individual
            emotion-associated words.
        </div>

    </div>

</div>


<h2 class="section-title">
    Confusion Matrix
</h2>

<div class="panel">

    {confusion_html}

</div>


<h2 class="section-title">
    Review-Level Results
</h2>


<div class="filter-area">

    <button
        class="filter-button active"
        onclick="filterReviews('all', this)"
    >
        All Reviews
    </button>

    <button
        class="filter-button"
        onclick="filterReviews('correct', this)"
    >
        Correct Only
    </button>

    <button
        class="filter-button"
        onclick="filterReviews('mismatch', this)"
    >
        Mismatches Only
    </button>

    <span id="visible-count">
        Showing {total_reviews} reviews
    </span>

</div>


<div class="table-container">

<table>

<thead>

<tr>
    <th>Rating</th>
    <th>Review</th>
    <th>Actual</th>
    <th>Predicted</th>
    <th>LLM Emotion</th>
    <th>NRC Emotion</th>
    <th>Result</th>
</tr>

</thead>

<tbody>

{review_rows}

</tbody>

</table>

</div>


</div>


<script>

function filterReviews(filter, button) {{

    const rows =
        document.querySelectorAll(
            "tbody tr"
        );

    let visible = 0;

    rows.forEach(row => {{

        const result =
            row.dataset.result;

        if (
            filter === "all"
            ||
            result === filter
        ) {{

            row.style.display = "";
            visible++;

        }} else {{

            row.style.display = "none";

        }}

    }});

    document.getElementById(
        "visible-count"
    ).textContent =
        `Showing ${{visible}} reviews`;

    document.querySelectorAll(
        ".filter-button"
    ).forEach(btn => {{

        btn.classList.remove(
            "active"
        );

    }});

    button.classList.add(
        "active"
    );
}}

</script>


</body>

</html>
"""


# ============================================================
# SAVE FINAL DASHBOARD
# ============================================================

Path(OUTPUT_FILE).write_text(
    html_output,
    encoding="utf-8"
)


# ============================================================
# TERMINAL SUMMARY
# ============================================================

print("=" * 60)
print("STEP 7 FINAL DASHBOARD CREATED")
print("=" * 60)

print(
    f"Reviews displayed: "
    f"{total_reviews}"
)

print(
    f"Overall accuracy: "
    f"{accuracy:.1f}%"
)

print(
    f"Positive accuracy: "
    f"{positive_accuracy:.1f}%"
)

print(
    f"Neutral accuracy: "
    f"{neutral_accuracy:.1f}%"
)

print(
    f"Negative accuracy: "
    f"{negative_accuracy:.1f}%"
)

print(
    f"Emotion agreement: "
    f"{emotion_agreement:.1f}%"
)

print(
    f"Correct predictions: "
    f"{correct_reviews}"
)

print(
    f"Mismatches: "
    f"{incorrect_reviews}"
)

print(
    f"Dashboard saved as:"
    f"\n{OUTPUT_FILE}"
)

print("\nConfusion Matrix:")
print(confusion)


