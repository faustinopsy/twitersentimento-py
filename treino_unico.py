import os
import pickle
import pandas as pd
import psutil
from sentiment_pipeline import SentimentPipeline
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

def medir_memoria():
    process = psutil.Process()
    mem_info = process.memory_info()
    return mem_info.rss

def main():
    dataset_path = "./datasets/clean_tweets.csv"
    models_dir = "./models"
    os.makedirs(models_dir, exist_ok=True)

    mem_before = medir_memoria()

    df = pd.read_csv(dataset_path)
    samples = df['text'].tolist()
    labels = df['sentiment'].tolist()
    
    total_samples = len(samples)
    print(f"Total de amostras: {total_samples}")
    
    train_size = int(total_samples * 0.9)
    
    train_samples = samples[:train_size]
    train_labels = labels[:train_size]
    test_samples = samples[train_size:]
    test_labels = labels[train_size:]
    
    print(f"Utilizando {len(train_samples)} amostras para treinamento.")
    print(f"Utilizando {len(test_samples)} amostras para validação.")
    
    sp = SentimentPipeline()
    sp.train(train_samples, train_labels)
    
    model_filepath = os.path.join(models_dir, "sentiment_model.pkl")
    sp.save(model_filepath)
    print(f"Modelo treinado e salvo em: {model_filepath}")
    
    predictions = sp.predict(test_samples)
    acc = accuracy_score(test_labels, predictions) * 100
    print(f"Acurácia no conjunto de validação: {acc:.2f}%")
    
    cm = confusion_matrix(test_labels, predictions, labels=["positive", "negative", "neutral"])
    print("Matriz de Confusão:")
    print(cm)
    print("\nRelatório de Classificação:")
    print(classification_report(test_labels, predictions))
    
    mem_after = medir_memoria()
    memory_used = mem_after - mem_before
    print(f"Memória utilizada durante o treinamento: {memory_used} bytes ({memory_used / 1048576:.2f} MB)")

if __name__ == "__main__":
    main()
