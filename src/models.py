
import logging
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

class Model:
    def __init__(self):
        self.model = RandomForestClassifier()

    def train(self, data):
        # Split data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(data[:, :-1], data[:, -1], test_size=0.2, random_state=42)

        # Train model
        self.model.fit(X_train, y_train)

        # Evaluate model
        y_pred = self.model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        logging.info(f'Model accuracy: {accuracy}')

    def predict(self, data):
        # Make predictions
        predictions = self.model.predict(data[:, :-1])
        return predictions
