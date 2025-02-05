import os
import psutil
from flask import Flask, render_template, request
from sentiment_pipeline import SentimentPipeline

app = Flask(__name__)
MODELS_DIR = "./models"

def modelo_unificado():
    model_path = os.path.join(MODELS_DIR, "sentiment_x.pkl")
    if os.path.exists(model_path):
        return SentimentPipeline.load(model_path)
    return None

def load_ensemble_models():
    model_files = sorted([f for f in os.listdir(MODELS_DIR)
                           if f.startswith("sentiment_batch_") and f.endswith(".pkl")])
    if model_files:
        model_files = model_files[:-1]
    models = []
    for file in model_files:
        filepath = os.path.join(MODELS_DIR, file)
        sp = SentimentPipeline.load(filepath)
        models.append(sp)
    return models

def ensemble_predict(models, sample):
    votes = []
    for model in models:
        prediction = model.predict([sample])
        votes.append(prediction[0])
    tally = {}
    for vote in votes:
        tally[vote] = tally.get(vote, 0) + 1
    return max(tally, key=tally.get)

def medir_memoria():
    """Retorna o uso atual de memória (RSS) do processo em bytes."""
    process = psutil.Process()
    return process.memory_info().rss

@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None
    memory_used = None
    if request.method == "POST":
        input_text = request.form.get("text", "")
        if input_text:
            unified_model = modelo_unificado()
            mem_before = medir_memoria()
            if unified_model is not None:
                prediction = unified_model.predict([input_text])[0]
            else:
                models = load_ensemble_models()
                prediction = ensemble_predict(models, input_text)
            mem_after = medir_memoria()
            memory_used = mem_after - mem_before
    return render_template("index.html", prediction=prediction, memory_used=memory_used)

if __name__ == "__main__":
    app.run(debug=True)
