import os
import math
import pickle
import pandas as pd
import psutil
from sentiment_pipeline import SentimentPipeline

def medir_memoria():
    process = psutil.Process()
    mem_info = process.memory_info()
    return mem_info.rss

def ensemble_predict(models, sample):
    votes = []
    for model in models:
        prediction = model.predict([sample])
        votes.append(prediction[0])
   
    tally = {}
    for vote in votes:
        tally[vote] = tally.get(vote, 0) + 1
    return max(tally, key=tally.get)

def main():
    models_dir = "./models"
    model_files = sorted([f for f in os.listdir(models_dir) if f.startswith("sentiment_batch_") and f.endswith(".pkl")])
    if not model_files:
        print("Nenhum modelo encontrado. Execute o treinamento por lotes primeiro.")
        return

    model_files = model_files[:-1]
    
    models = []
    for file in model_files:
        filepath = os.path.join(models_dir, file)
        sp = SentimentPipeline.load(filepath)
        models.append(sp)
    print(f"Carregados {len(models)} submodelos (excluindo o 10º lote).")
    
    mem_before = medir_memoria()
    
    df = pd.read_csv("./datasets/clean_tweets.csv")
    samples = df['text'].tolist()
    labels = df['sentiment'].tolist()
    
    total_samples = len(samples)
    num_batches = 10
    batch_size = math.ceil(total_samples / num_batches)
    
    # Divide os dados em 10 partes
    samples_batches = [samples[i * batch_size:(i + 1) * batch_size] for i in range(num_batches)]
    labels_batches = [labels[i * batch_size:(i + 1) * batch_size] for i in range(num_batches)]
    
    test_samples = samples_batches[-1]
    test_labels = labels_batches[-1]
    
    print(f"Número de amostras do conjunto de teste (10ª parte): {len(test_samples)}")
    
    total = len(test_samples)
    correct = 0
    incorrect = 0

    for sample, true_label in zip(test_samples, test_labels):
        predicted = ensemble_predict(models, sample)
        if predicted == true_label:
            correct += 1
        else:
            incorrect += 1

    accuracy = (correct / total * 100) if total > 0 else 0
    print(f"Total de amostras testadas: {total}")
    print(f"Acertos: {correct}")
    print(f"Erros: {incorrect}")
    print(f"Acurácia: {accuracy:.2f}%")
    
    mem_after = medir_memoria()
    memory_used = mem_after - mem_before
    print(f"\nMemória utilizada na predição: {memory_used} bytes ({memory_used / 1048576:.2f} MB)")

if __name__ == "__main__":
    main()
