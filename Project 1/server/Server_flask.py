from flask import Flask, request, jsonify, render_template
import Utility

app = Flask(__name__)


@app.route("/get_area_types")
def get_area_types():
    response = jsonify({
        'area_types': Utility.get_area_types()
    })
    response.headers.add('Access-Control-Allow-Origin', '*')
    return response


@app.route("/get_locations")  # to not get 404 # Creating a routine / route
def get_locations():
    response = jsonify({
        'locations': Utility.get_locations()
    })
    response.headers.add('Access-Control-Allow-Origin', '*')
    return response


@app.route("/predict_home_price", methods=['POST'])
def predict_home_price():
    area_type = request.form['area_type']
    location = request.form['location']
    total_sqft = float(request.form['total_sqft'])
    balcony = int(request.form['balcony'])
    bathroom = int(request.form['bathroom'])
    BHK = int(request.form['BHK'])

    response = jsonify({
        'predicted_price': Utility.get_predicted_price(area_type, location, total_sqft, balcony, bathroom, BHK)
    })
    response.headers.add('Access-Control-Allow-Origin', '*')
    return response


if __name__ == "__main__":
    print("Bangalore House Price Prediction")
    Utility.load_artifacts()
    app.run(debug=True)
