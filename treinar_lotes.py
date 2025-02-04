import os
import math
import pickle
import pandas as pd
import psutil
import time
from sentiment_pipeline import SentimentPipeline

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
    num_batches = 10
    batch_size = math.ceil(total_samples / num_batches)
    
    print(f"Total de amostras: {total_samples}")
    print(f"Dividindo em {num_batches} lotes (aprox. {batch_size} amostras cada)...")

    samples_batches = [samples[i * batch_size:(i + 1) * batch_size] for i in range(num_batches)]
    labels_batches = [labels[i * batch_size:(i + 1) * batch_size] for i in range(num_batches)]
    
    for i, (s_batch, l_batch) in enumerate(zip(samples_batches, labels_batches)):
        print(f"Treinando lote {i+1} de {num_batches}...")
        sp = SentimentPipeline()
        sp.train(s_batch, l_batch)
        
        model_filepath = os.path.join(models_dir, f"sentiment_batch_{i}.pkl")
        sp.save(model_filepath)
        print(f"Lote {i+1} treinado e salvo em: {model_filepath}")
    
    mem_after = medir_memoria()
    memory_used = mem_after - mem_before

    print("Treinamento por lotes concluído.")
    print(f"Memória utilizada durante o treinamento: {memory_used} bytes ({memory_used / 1048576:.2f} MB)")

if __name__ == "__main__":
    main()
