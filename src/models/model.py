import numpy as np

class Model:
    """Wrapper class for the Keras models"""
    def __init__(self, kmodel, preprocess_fn):
        self.kmodel = kmodel
        self.preprocess_fn = preprocess_fn
        # Avoid Keras lazy predict function construction
        self.kmodel._make_predict_function()

    def predict(self, data):
        """Computes the wrapped model's output for an input
        
        Args:
        data: array of shape (height, width, channels) to compute features on

        Returns: the model's output of shape (1, out_height, out_width, 
        channels)
        """
        data = np.expand_dims(data, axis=0)
        data = self.preprocess_fn(data)
        output = self.kmodel.predict(data)
        return output

    @property
    def output_shape(self):
        """Returns the shape of the model representation

        Note that this only works with the Tensorflow backend
        Note that the height and width components are going to be unspecified. 
        """
        # @Todo: Consider to dynamically compute the real shape for a specific 
        # input shape
        return self.kmodel.layers[-1].output_shape
