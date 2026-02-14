from abc import ABC, abstractmethod
from typing import Dict, Any

class BaseModel(ABC):

    @abstractmethod
    def fit(self, data):
        """@brief Fit the model on training data.

        @param data Training dataset.
        @return None.
        """
        pass

    @abstractmethod
    def predict(self, data_point):
        """@brief Predict on a single data point.

        @param data_point Input data point.
        @return Prediction result.
        """
        pass

    @abstractmethod
    def save(self) -> Dict[str, Any]:
        """@brief Return serializable model state (params + metrics).

        @return Dictionary with model parameters and metrics.
        """
        pass

    @classmethod
    @abstractmethod
    def load(cls, state: Dict[str, Any]) -> "BaseModel":
        """@brief Restore model from serialized state.
        
        @param state Dictionary with model parameters and metrics.
        @return Restored model instance.
        """
        pass
