import pandas as pd

def limpar_dados(source_filepath, destination_filepath):
    df = pd.read_csv(source_filepath)
    # colunas 'text' (coluna 10) e 'airline_sentiment' (coluna 1)
    df_clean = df.iloc[:, [10, 1]]
    
    df_clean.columns = ['text', 'sentiment']
    
    df_clean.to_csv(destination_filepath, index=False)
    print(f"Dados limpos gravados em: {destination_filepath}")

if __name__ == "__main__":
    source = "./datasets/Tweets.csv" 
    destination = "./datasets/clean_tweets.csv"
    limpar_dados(source, destination)
