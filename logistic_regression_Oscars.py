import math

class LogisticActorPicker:
    def __init__(self):

        self.weights = [4.5, 2.1, -3.0, 1.5]
        self.bias = -2.0

    def sigmoid(self, z):
        return 1 / (1 + math.exp(-z))

    def predict(self, features):
        z = sum(w * f for w, f in zip(self.weights, features)) + self.bias
        return self.sigmoid(z)

candidates = {
    "Chalamet": [0, 0.92, 0, 0.8],
    "Jordan":   [1, 0.97, 0, 1.0],  
    "Moura":    [0, 0.88, 0, 0.9],
    "DiCaprio": [0, 0.90, 0, 0.7],
    "Hawke":    [0, 0.94, 0, 0.6]
}

model = LogisticActorPicker()

print("2026 RESULTS")
results = []
for name, features in candidates.items():
    prob = model.predict(features)
    results.append((name, prob))

results.sort(key=lambda x: x[1], reverse=True)
oldprob = -1.0
winner_name = ""

for name, prob in results:
    print(f"{name:<10}: {prob*100:.2f}%")
    if prob > oldprob:
       oldprob = prob 
       winner_name = name

print(f"\nThe oscar goes to... : {winner_name}")