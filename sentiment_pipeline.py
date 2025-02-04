import pickle
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.naive_bayes import MultinomialNB

class SentimentPipeline:
    def __init__(self):
        self.pipeline = Pipeline([
            ('vect', CountVectorizer()),
            ('tfidf', TfidfTransformer()),
            ('clf', MultinomialNB()),
        ])

    def train(self, samples, labels):
        """Treina o pipeline com os dados fornecidos."""
        self.pipeline.fit(samples, labels)

    def predict(self, samples):
        """Retorna as predições para os dados fornecidos."""
        return self.pipeline.predict(samples)

    def save(self, filepath):
        """Serializa o pipeline para um arquivo."""
        with open(filepath, 'wb') as f:
            pickle.dump(self.pipeline, f)

    @classmethod
    def load(cls, filepath):
        """Carrega um pipeline serializado a partir de um arquivo."""
        with open(filepath, 'rb') as f:
            pipeline = pickle.load(f)
        obj = cls()
        obj.pipeline = pipeline
        return obj

if __name__ == "__main__":
    samples = ["I love this airline", "I hate delays", "The flight was average"]
    labels = ["positive", "negative", "neutral"]

    sp = SentimentPipeline()
    sp.train(samples, labels)
    preds = sp.predict(samples)
    print("Previsões:", preds)
