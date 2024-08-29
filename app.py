from flask import Flask, send_from_directory, jsonify
import pandas as pd
import os

app = Flask(__name__)

@app.route('/get_sentiment_data')
def get_sentiment_data():
    try:
        df = pd.read_csv('sentiment_results.csv')
        data = df.to_dict(orient='records')
        return jsonify(data)
    except FileNotFoundError:
        return jsonify([])

@app.route('/charts/<path:filename>')
def serve_chart(filename):
    return send_from_directory('static/charts', filename)

if __name__ == '__main__':
    app.run(debug=True)
