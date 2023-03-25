from flask import Flask, render_template, request, redirect, url_for, jsonify
from utils import helpers, shared_state
from run import execute
import shutil

app = Flask(__name__, template_folder='.')

@app.route('/', methods=['GET'])
def index():
    return render_template(r"web_ui.html")


def add_word(prompt, token):
    if len(prompt) == 0 or prompt[-1] == ' ':
        prompt += token
    else:
        prompt += ' ' + token
    return prompt

@app.route('/execute_function', methods=['POST'])
def execute_function():
    meta_prompt = request.json['variable1']
    shared_state.config.meta_prompt = meta_prompt
    image_path = execute(shared_state.config)
    shutil.copyfile(str(image_path), "static/output.png")
    response = {
        'result': str(image_path)
    }

    
    return jsonify(response)


@app.route('/post', methods=['POST'])
def post():
    return "recived: {}".format(request.form)

def run():
    app.run(debug=True)

if __name__ == "__main__":
    run()