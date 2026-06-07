
import logging
from src.models import Model
from src.utils import calculate_drift

class Pipeline:
    def __init__(self, data, drift_threshold):
        self.data = data
        self.drift_threshold = drift_threshold
        self.model = Model()

    def run(self):
        # Train model
        self.model.train(self.data)

        # Make predictions
        predictions = self.model.predict(self.data)

        # Calculate drift
        drift = calculate_drift(self.data, predictions)

        # Check for drift
        if drift > self.drift_threshold:
            logging.warning(f'Drift detected: {drift}')
        else:
            logging.info(f'No drift detected: {drift}')
