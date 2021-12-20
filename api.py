import flask
import script
app = flask.Flask(__name__)
app.config["DEBUG"] = True

@app.route('/process', methods=['GET'])
def home():
    input = flask.request.args.get('input')
    return script.processInput(input)

app.run()