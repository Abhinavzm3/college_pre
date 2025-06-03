from flask import Flask, request, jsonify, render_template
import pandas as pd
import numpy as np

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
    return max(0.0, min(100.0, prob * 100))

# 3) Route: serve the main HTML page
@app.route("/")
def index():
    return render_template("index.html")

# 4) POST /predict (fixed probability calculation)
@app.route("/predict", methods=["POST"])
def predict():
    # Validate JSON exists
    if not request.is_json:
        return jsonify({"error": "JSON body required"}), 400
    
    data = request.get_json()
    
    # Validate required fields
    required_fields = ["user_rank", "quota", "category", "gender", "round"]
    for field in required_fields:
        if field not in data or not str(data[field]).strip():
            return jsonify({"error": f"Missing required field: {field}"}), 400

    # Parse inputs with validation
    try:
        user_rank = int(data["user_rank"])
        if user_rank <= 0:
            raise ValueError("Rank must be positive integer")
            
        quota = data["quota"].strip()
        category = data["category"].strip()
        gender = data["gender"].strip()
        round_name = data["round"].strip()
        top_n = int(data.get("top_n", 10))
    except Exception as e:
        return jsonify({"error": f"Invalid input: {str(e)}"}), 400

    # Filter by quota, category, gender, and round
    subset = df[
        (df["Quota"] == quota) &
        (df["Category"] == category) &
        (df["Seat Gender"] == gender) &
        (df["Round"] == round_name)
    ].copy()

    # If no rows match those filters, return empty list
    if subset.empty:
        return jsonify({
            "predictions": [],
            "message": "No colleges found for the selected filters"
        })

    # Calculate probabilities for all valid colleges
    matches = []
    for _, row in subset.iterrows():
        op = row["Opening Rank"]
        cl = row["Closing Rank"]
        
        # Skip rows with invalid rank data
        if pd.isna(op) or pd.isna(cl):
            continue
            
        prob = compute_probability(user_rank, op, cl)
        
        # Only include colleges with non-zero probability
        if prob > 0:
            matches.append({
                "Institute": row["Institute"].strip(),
                "Program": row["Program"].strip(),
                "Opening Rank": float(op),
                "Closing Rank": float(cl),
                "Probability": round(prob, 2)
            })

    # Sort by highest probability first, then by closing rank (most selective first)
    matches.sort(key=lambda x: (x["Probability"], x["Closing Rank"]))

    # Return up to top_n results
    top_matches = matches[:top_n]
    
    # Add warning if no matches but data exists
    if not top_matches:
        min_close = subset["Closing Rank"].min()
        max_open = subset["Opening Rank"].max()
        return jsonify({
            "predictions": [],
            "message": f"No colleges found. Your rank {user_rank} is outside range. " +
                      f"Valid range for filters: Opening ≤ {max_open}, Closing ≥ {min_close}"
        })
        
    return jsonify({"predictions": top_matches})

# 5) GET /college-info (improved error handling)
@app.route("/college-info", methods=["GET"])
def college_info():
    name_param = request.args.get("name", "").strip()
    program_param = request.args.get("program", "").strip()
    
    if not name_param:
        return jsonify({"error": "Query parameter 'name' is required."}), 400

    try:
        mask = df["Institute"].str.contains(name_param, case=False, na=False)
        if program_param:
            mask &= df["Program"].str.contains(program_param, case=False, na=False)

        filtered = df[mask].copy()
        results = []
        for _, row in filtered.iterrows():
            # Handle NaN values in rank columns
            opening = row["Opening Rank"]
            closing = row["Closing Rank"]
            
            results.append({
                "Institute": row["Institute"].strip(),
                "Program": row["Program"].strip(),
                "Quota": row["Quota"].strip(),
                "Category": row["Category"].strip(),
                "Opening Rank": float(opening) if not pd.isna(opening) else "N/A",
                "Closing Rank": float(closing) if not pd.isna(closing) else "N/A"
            })

        return jsonify({
            "count": len(results),
            "results": results,
            "message": f"Found {len(results)} matches" if results else "No colleges found"
        })
        
    except Exception as e:
        return jsonify({"error": f"Processing error: {str(e)}"}), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)