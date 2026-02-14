from schemas.training import TrainData

class Trainer:
    def __init__(self, series_id: str, data: TrainData):
        self.series_id = series_id
        self.data = data

    def train(self):
        print(f"Training model for series {self.series_id} with data: {self.data}")