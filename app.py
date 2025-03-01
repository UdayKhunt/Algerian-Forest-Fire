from flask import Flask,render_template, request
import pickle


app = Flask(__name__)

scaler = pickle.load(open('models/scaler.pkl','rb'))
ridge = pickle.load(open('models/ridge.pkl','rb'))

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict' , methods = ['GET','POST'])
def predict():
    if request.method == 'POST':
        Temperature = float(request.form.get('temperature'))
        RH = float(request.form.get('rh'))
        Ws = float(request.form.get('ws'))
        Rain = float(request.form.get('rain'))
        FFMC = float(request.form.get('ffmc'))
        DMC = float(request.form.get('dmc'))
        ISI = float(request.form.get('isi'))
        Classes = float(request.form.get('classes'))
        Region = float(request.form.get('region'))
        scaled_data = scaler.transform([[Temperature,RH,Ws,Rain,FFMC,DMC,ISI,Classes,Region]])
        result = ridge.predict(scaled_data)[0]

        return render_template('form.html',x = result)

    else:
        return render_template('form.html',x=None)


if __name__=='__main__':
    app.run(host='0.0.0.0')