from keras.models import Model


def _vgg_16(output_layer='block5_conv3'):
    from keras.applications.vgg16 import VGG16
    model = VGG16(weights='imagenet', include_top=False)
    wrapped_model = Model(input=model.input, 
                          output=model.get_layer(output_layer).output)
    return wrapped_model


def _res_net_50(output_layer='activation_49'):
    from keras.applications.resnet50 import ResNet50
    
    model = ResNet50(weights='imagenet', include_top=False)
    wrapped_model = Model(input=model.input, 
                          output=model.get_layer(output_layer).output)
    return wrapped_model


def pretrained_models(model_name):
    """Returns a function constructing the specified pretrained model"""
    models = {
        'ResNet50': _res_net_50,
        'VGG16': _vgg_16
    }
    return models.get(model_name)


def preprocessor(model_name):
    """Returns a function preprocessing an image before passing it to the net"""
    if model_name == 'ResNet50':
        from keras.applications.resnet50 import preprocess_input
        return preprocess_input
    elif model_name == 'VGG16':
        from keras.applications.vgg16 import preprocess_input
        return preprocess_input
