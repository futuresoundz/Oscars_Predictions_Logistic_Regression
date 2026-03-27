import math
import pandas as pd

class BestActorPredictor:
    def __init__(self, num_features):

        self.weights = [0.0] * num_features
        self.bias = 0.0

    def sigmoid(self, z):
        return 1 / (1 + math.exp(-z))

    def predict(self, features):
        z = sum(w * f for w, f in zip(self.weights, features)) + self.bias
        return self.sigmoid(z)
    
    def train(self, training_data, labels, epochs=2000, lr=0.001):
        for _ in range(epochs):
            for features, actual_outcome in zip(training_data, labels):
                prediction = self.predict(features)
                error = actual_outcome - prediction
                for i in range(len(self.weights)):
                    self.weights[i] += lr * error * features[i]
                self.bias += lr * error

    def get_precision(self, test_data, labels, threshold = 0.5):
        true_positives = 0
        false_positives = 0

        for features, actual in zip(test_data, labels):
            probabilty = self.predict(features)
            prediction = 1 if probabilty >= threshold else 0

            if prediction == 1:
                if actual == 1:
                    true_positives +=1
                else: 
                    false_positives += 1

        if (true_positives + false_positives) == 0:
            return 0.0
        
        precision_score = (true_positives / (true_positives + false_positives)) * 100

        return precision_score

candidates = {
    "Michael B. Jordan":   [1, 0, 0, 0], 
    "Timothee Chalamet": [0, 0, 1, 1], 
    "Leonardo DiCaprio": [0, 0, 0, 0], 
    "Wagner Moura":    [0, 0, 1, 0], 
    "Ethan Hawke":    [0, 0, 0, 0]  
}

model = BestActorPredictor(4)

df = pd.read_csv("oscars_data_2026.csv")
x = df[['sag', 'bafta', 'gg', 'cca']].values.tolist()
y = df['won_oscar'].tolist()
model.train(x, y)
results = [(name, model.predict(features)) for name, features in candidates.items()]

def get_probability(item):
        return item[1]
results.sort(key = get_probability, reverse = True)

for name, probabilty in results:
    print(f"{name:<10}: {probabilty*100:.2f}%")

winner_name, winner_prob = results[0]

print(f"\nThe oscar goes to... : {winner_name}!")
precision = model.get_precision(x, y)
print(f"Precision score: {precision:.3f}%")
