from flask import Flask

app = Flask(__name__)

@app.route('/')
def hello_world():
    return 'Hello from my CI/CD powered Flask app! V1'

@app.route('/test')
def test_func():
    return "One more test from Flask"


app.run(host='localhost',port=5001)