from flask import Flask, request, jsonify
import numpy as np
from sklearn.linear_model import LinearRegression
from flask_cors import CORS

app = Flask(__name__)
CORS(app)
# Hardcoded data (replace with a database or dynamic data)
years = np.array([2011, 2012, 2013, 2014, 2015, 2016, 2017,
                 2018, 2019, 2020, 2021, 2022]).reshape(-1, 1)
house_worth = np.array([500000, 540000, 550000, 575000, 600000,
                       700000, 800000, 750000, 775000, 825000, 900000, 975000])

# Create and train the linear regression model
model = LinearRegression()
model.fit(years, house_worth)


@app.route('/predict', methods=['POST'])
def predict():
    # Get data from the client (POST request)
    data = request.json
    user_savings = data.get('user_savings')
    target_year = data.get('target_year')

    if not user_savings or not target_year:
        return jsonify({"error": "Missing user_savings or target_year"}), 400

    # Predict house worth for the target year
    target_year_array = np.array([[target_year]])
    predicted_house_worth_target_year = model.predict(target_year_array)[0]

    # Calculate the percentage of the user's savings relative to the predicted house worth
    percentage_of_goal = (
        user_savings / predicted_house_worth_target_year) * 100

    result = {
        "predicted_house_worth": predicted_house_worth_target_year,
        "user_savings": user_savings,
        "percentage_of_goal": percentage_of_goal,
        "target_year": target_year,
        "message": "",
        "difference_needed": 0.0
    }

    # Message based on the user's savings
    if user_savings >= predicted_house_worth_target_year:
        result["message"] = f"Congratulations! You can afford the house in {target_year}!"
    else:
        result["message"] = f"You still need {predicted_house_worth_target_year - user_savings:,.2f} CAD more to buy the house in {target_year}."
        result["difference_needed"] = predicted_house_worth_target_year - user_savings

    # Return the result as a JSON response
    return jsonify(result)


if __name__ == '__main__':
    app.run(debug=True, port=5001)
