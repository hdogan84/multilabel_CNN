# custom handler file
import torch

# model_handler.py

"""
ModelHandler defines a custom model handler.
"""

from ts.torch_handler.base_handler import BaseHandler


class ModelHandler(BaseHandler):
    """
    A custom model handler implementation.
    """

    def preprocess(self, data):
        """
        Transform raw input into one Data Loader.
        :param batch: list of raw requests, should match batch size
        :return: list of preprocessed model input data
        """
        # Take the input data and make it inference ready
        # Has to be implement in child class

    def inference(self, data, *args, **kwargs):
        """
        The Inference Function is used to make a prediction call on the given input request.
        The user needs to override the inference function to customize it.

        Args:
            data (DataLoader): torch DataLoader  is passed to make the Inference Request of all batches.
            The shape should match the model input shape.

        Returns:
            Torch Tensor : The Predicted Torch Tensor of all Batches of data loader is returned in this function.
        """

        with torch.no_grad():
            outputs = []
            for samples, _, sample_ids in data:
                marshalled_data = samples.to(self.device)
                outputs.append(self.model(marshalled_data, *args, **kwargs))

        return torch.stack(outputs).to(self.device)
