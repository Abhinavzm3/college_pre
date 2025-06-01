from flask import Flask, request, jsonify, render_template
import pandas as pd

app = Flask(__name__)

# 1) Load CSV & clean up column names
csv_path = 'AKTU_Counselling.csv'
df = pd.read_csv(csv_path)

# Strip off trailing " ▲▼" from all columns
df.columns = [col.replace('\xa0▲▼', '').strip() for col in df.columns]

# Convert Opening/Closing Rank columns to float
df['Opening Rank'] = pd.to_numeric(df['Opening Rank'], errors='coerce')
df['Closing Rank'] = pd.to_numeric(df['Closing Rank'], errors='coerce')

# 2) Helper to compute admission probability
def compute_probability(rank: int, opening: float, closing: float) -> float:
    """
    - If rank <= opening → 100%
    - If rank >= closing → 0%
    - Otherwise, linear interpolation between opening and closing
    """
    if pd.isna(opening) or pd.isna(closing):
        return 0.0
    if opening == closing:
        return 100.0 if rank <= opening else 0.0
    if rank <= opening:
        return 100.0
    if rank >= closing:
        return 0.0
    # Linear interpolation
    prob = (closing - rank) / (closing - opening)
    return max(0.0, min(1.0, prob)) * 100.0

# 3) Route: serve the main HTML page
@app.route("/")
def index():
    return render_template("index.html")

# 4) POST /predict (includes "round" and rank-extremes logic)
@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()
    if not data:
        return jsonify({"error": "JSON body required"}), 400

    # Parse inputs
    try:
        user_rank  = int(data.get("user_rank", -1))
        quota      = data.get("quota", "").strip()
        category   = data.get("category", "").strip()
        gender     = data.get("gender", "").strip()
        round_name = data.get("round", "").strip()
        top_n      = int(data.get("top_n", 10))
    except Exception as e:
        return jsonify({"error": f"Invalid input fields: {e}"}), 400

    # Validate required fields
    if user_rank < 0 or not quota or not category or not gender or not round_name:
        return jsonify({"error": "Missing or invalid field(s)"}), 400

    # Filter by quota, category, gender, and round
    subset = df[
        (df["Quota"] == quota) &
        (df["Category"] == category) &
        (df["Seat Gender"] == gender) &
        (df["Round"] == round_name)
    ].copy()

    # If no rows match those filters, return empty list
    if subset.empty:
        return jsonify({"predictions": []})

    # Extract the numeric arrays of opening/closing ranks
    openings = subset["Opening Rank"].dropna().values
    closings = subset["Closing Rank"].dropna().values

    # If user_rank is lower (better) than the minimum opening rank,
    # return *all* rows with 100% probability
    min_open = float(openings.min()) if len(openings) > 0 else None
    max_close = float(closings.max()) if len(closings) > 0 else None

    matches = []

    if min_open is not None and user_rank <= min_open:
        # Lower (better) than every opening: include all rows
        for _, row in subset.iterrows():
            op = row["Opening Rank"]
            cl = row["Closing Rank"]
            if pd.isna(op) or pd.isna(cl):
                continue
            matches.append({
                "Institute": row["Institute"].strip(),
                "Program": row["Program"].strip(),
                "Opening Rank": float(op),
                "Closing Rank": float(cl),
                "Probability": 100.0
            })
    else:
        # If user_rank is worse than (>) the maximum closing, return no matches
        if max_close is not None and user_rank > max_close:
            return jsonify({"predictions": []})

        # Otherwise, find all rows where Opening Rank ≤ user_rank ≤ Closing Rank
        for _, row in subset.iterrows():
            op = row["Opening Rank"]
            cl = row["Closing Rank"]

            if pd.isna(op) or pd.isna(cl):
                continue

            if (user_rank <= cl) and (user_rank >= op):
                prob = compute_probability(user_rank, op, cl)
                matches.append({
                    "Institute": row["Institute"].strip(),
                    "Program": row["Program"].strip(),
                    "Opening Rank": float(op),
                    "Closing Rank": float(cl),
                    "Probability": round(prob, 2)
                })

    # Sort matches by **Probability ascending**, then by Closing Rank ascending
    matches.sort(key=lambda x: (x["Probability"], x["Closing Rank"]))

    # Return up to top_n results
    top_matches = matches[:top_n]
    return jsonify({"predictions": top_matches})


# 5) GET /college-info (unchanged)
@app.route("/college-info", methods=["GET"])
def college_info():
    name_param    = request.args.get("name", "").strip()
    program_param = request.args.get("program", "").strip()
    if not name_param:
        return jsonify({"error": "Query parameter 'name' is required."}), 400

    mask = df["Institute"].str.contains(name_param, case=False, na=False)
    if program_param:
        mask &= df["Program"].str.contains(program_param, case=False, na=False)

    filtered = df[mask].copy()
    results = []
    for _, row in filtered.iterrows():
        results.append({
            "Institute": row["Institute"].strip(),
            "Program": row["Program"].strip(),
            "Quota": row["Quota"].strip(),
            "Category": row["Category"].strip(),
            "Opening Rank": float(row["Opening Rank"]),
            "Closing Rank": float(row["Closing Rank"])
        })

    return jsonify({"count": len(results), "results": results})


if __name__ == "__main__":
    # Debug mode for local testing
    app.run(host="0.0.0.0", port=5000, debug=True)
