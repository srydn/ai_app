from flask import Flask, request, jsonify
import joblib
import numpy as np
from flask_cors import CORS

app = Flask(__name__)
CORS(app)
model = joblib.load('../student_performance_model.pkl')


@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.json
        input_data = {
            'okumaSkoru': data['readingScore'],
            'yazmaSkoru': data['writingScore'],
            'cinsiyet_male': 1 if data['gender'] == 'Erkek' else 0,
            'ebeveynEgitim_bachelor\'s degree': 1 if data['parentEdu'] == 'Lisans' else 0,
            'ebeveynEgitim_high school': 1 if data['parentEdu'] == 'Lise Mezunu' else 0,
            'ebeveynEgitim_master\'s degree': 1 if data['parentEdu'] == 'Yüksek Lisans' else 0,
            'ebeveynEgitim_some college': 1 if data['parentEdu'] == 'Lisans (Mezun Değil)' else 0,
            'ebeveynEgitim_some high school': 1 if data['parentEdu'] == 'Lise (Mezun Değil)' else 0,
            'ogleYemegi_standard': 1 if data["lunch"] == 'Standart' else 0,
            'testHazirlik_none': 1 if data['testPrep'] == "Yok" else 0
        }
        feature_order = ['okumaSkoru','yazmaSkoru', 'cinsiyet_male', 'ebeveynEgitim_bachelor\'s degree',
                         'ebeveynEgitim_high school', 'ebeveynEgitim_master\'s degree',
                         'ebeveynEgitim_some college', 'ebeveynEgitim_some high school',
                         'ogleYemegi_standard', 'testHazirlik_none']
        model_input = [input_data[col] for col in feature_order]
        print(model_input)
        raw_predict = model.predict([model_input])
        prediction = max(0,min(100,raw_predict))
        return jsonify({'prediction': float(prediction[0])})

    except Exception as e:
        return jsonify({'Error oocured: ': str(e)}), 400
# try:
#     features [[data[]]]

if __name__ == '__main__':
    app.run(debug=True)
