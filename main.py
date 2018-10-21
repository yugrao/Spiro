from flask import Flask, render_template, request

app = Flask(__name__)

@app.route("/")
def index():
    return "hello world"#render_template('templates/index.html', word = "platinum")

@app.route("/req", methods=['POST'])
def run_model():
    req_data = request.get_json()
    pic = req_data["pic"]
    message = req_data["message"]
    return jsonify({"i got":"your request", "req":str(req_data)})#"I GOT YOUR REQUEST OF"+str(req_data)

if __name__ == '__main__':
    app.run(host='127.0.0.1', port=8080, debug=True)
