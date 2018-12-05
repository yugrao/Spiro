#!flask/bin/python
from flask import Flask, jsonify
from flask import abort
from flask import request

app = Flask(__name__)

data = [
    {
        'id': 1,
        'data': [1,2,3,4,5],
        'output':3
    },
    {
        'id': 2,
        'data': [6,7,8,9,10],
        'output':8
    }
]

@app.route('/spiro/data', methods=['GET'])
def get_spiro_data():
    return jsonify({'data': data})

@app.route('/spiro/data/<int:data_id>', methods=['GET'])
def get_spiro_data2(data_id):
    d = [d for d in data if data['id'] == int(data_id)]
    if len(d) == 0:
        abort(404)
    return jsonify({'data': d[0]})

def predict_data(a):
    model = load_model('Spiral_Model-84.38-1020053228.h5')
    x, y = np.array(a).T
    x = list(map(lambda x:(x-480)*2/3, x))
    y = list(map(lambda y:(y-480)*2/3, y))
    plt.plot(x, y, "ko-", linewidth=5, markersize=0)
    plt.axis([-300, 300, 300, -300])
    plt.axis("off")
    fig = plt.gcf()
    fig.set_size_inches(5,5)
    fig.savefig("thisisarequestimg.png", dpi=25)
    plt.clf()
    img_a = np.asarray(Image.open("thisisarequestimg.png"))
    img_a = [[list(map(lambda rgbx: (rgbx[0] * 299.0/1000 + rgbx[1] * 587.0/1000 + rgbx[2] * 114.0/1000)/255, row)) for row in a]]
    img_a = np.array(img_a)
    return model.predict(np.array(img_a))

@app.route('/spiro/data', methods=['POST'])
def add_data():
    if not request.json or not 'data' in request.json:
        abort(400)
    new_data = {
        'id': data[-1]['id'] + 1,
        'data': [list(point) for point in request.json['data']]
    }
    new_data['output'] = predict_data(new_data['data'])
    data.append(new_data)
    return jsonify({'data': new_data}), 201


if __name__ == '__main__':
    app.run(debug=True)
