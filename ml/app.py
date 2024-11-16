from flask import Flask, request, jsonify
import joblib
import numpy as np
from datetime import datetime
from flask_cors import CORS

# Load the trained model
model = joblib.load("random_forest_interest_rate.pkl")

# Initialize Flask app
app = Flask(__name__)
# Enable CORS for all routes
CORS(app)



@app.route("/predict_savings", methods=["POST"])
def predict_savings():
    try:
        # Parse the JSON input
        data = request.json
        year = data.get("year")
        principal = data.get("principal")

        # Validate input
        if year is None or not isinstance(year, int):
            return jsonify({"error": "Invalid year. Please provide a valid year."}), 400
        if not principal:
            return jsonify({"error": "Missing principal."}), 400

        # Calculate the time dynamically based on the current year
        current_year = datetime.now().year
        time = year - current_year

        if time <= 0:
            return jsonify({"error": "The specified year must be in the future."}), 400

        # Predict the interest rate for the given year
        predicted_rate = model.predict(np.array([[year]]))[0]
        # if predicted_rate < 0:
        #     return jsonify({"error": "Predicted interest rate is negative. Check the model or input."}), 400

        # Convert the predicted rate to decimal
        rate_decimal = predicted_rate / 100

        # Calculate savings using Simple Interest formula
        total_amount = principal * (1 + rate_decimal * time)
        interest_earned = total_amount - principal

        # Return the results as JSON
        return jsonify(
            {
                "year": year,
                "current_year": current_year,
                "predicted_interest_rate": round(predicted_rate, 2),
                "principal": principal,
                "time": time,
                "total_amount": round(total_amount, 2),
            }
        )

    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(debug=True)
