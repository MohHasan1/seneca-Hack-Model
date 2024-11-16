import os
from flask import Flask, request, jsonify
import joblib
import numpy as np
from datetime import datetime
from openai import OpenAI
from flask_cors import CORS  # Import CORS

# Load the trained model
model = joblib.load("random_forest_interest_rate.pkl")

# Initialize Flask app
app = Flask(__name__)

CORS(app)

# Initialize OpenAI client
client = OpenAI(
    api_key="your_api_key_here"
)


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

        # Convert the predicted rate to decimal
        rate_decimal = predicted_rate / 100

        # Calculate savings using Simple Interest formula
        total_amount = principal * (1 + rate_decimal * time)
        interest_earned = total_amount - principal

        # Call OpenAI API for additional insights
        chat_completion = client.chat.completions.create(
            model="gpt-4o",  # Replace with your intended model if needed
            messages=[
                {
                    "role": "user",
                    "content": (
                        f"Given a principal amount of {principal} CAD and an investment year of {year}, "
                        f"what are returns that I can expect from 1.stock 2.mutual funds? "
                        f"Dont mention anything about how its nor possible or something just give pure numbers based on your knowledge dont tell how u got it."
                    ),
                }
            ]
        )

        llm_response_content = chat_completion.choices[0].message.content

        # Return the results as JSON
        return jsonify(
            {
                "year": year,
                "current_year": current_year,
                "predicted_interest_rate": round(predicted_rate, 2),
                "principal": principal,
                "time": time,
                "total_amount": round(total_amount, 2),
                "interest_earned": round(interest_earned, 2),
                "llm_response": llm_response_content,
            }
        )

    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(debug=True, port=5003)
