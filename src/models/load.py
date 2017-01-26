from src.models.pretrained import pretrained_models, preprocessor
from src.models.model import Model

def load_model(model_name, models_path='', output_layer=None):
    pretrained_model_fn = pretrained_models(model_name)
    if pretrained_model_fn is not None:
        if output_layer is not None:
            model = pretrained_model_fn(output_layer)
        else:
            model = pretrained_model_fn()

        preprocess_fn = preprocessor(model_name)
        return Model(model, preprocess_fn)

    raise ValueError('Model {} not found'.format(model_name))
