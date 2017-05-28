from keras.models import Model

def _wrap_model(model, output_layer):
    return Model(input=model.input, 
                 output=model.get_layer(output_layer).output)


def _vgg_16(output_layer='block5_conv3'):
    from keras.applications.vgg16 import VGG16
    model = VGG16(weights='imagenet', include_top=False)
    return _wrap_model(model, output_layer)


def _vgg_19(output_layer='block4_conv4'):
    from keras.applications.vgg19 import VGG19
    model = VGG19(weights='imagenet', include_top=False)
    return _wrap_model(model, output_layer)


def _res_net_50(output_layer='activation_49'):
    from keras.applications.resnet50 import ResNet50
    model = ResNet50(weights='imagenet', include_top=False)
    return _wrap_model(model, output_layer)


def _inception_v3(output_layer='mixed10'):
    from keras.applications.inception_v3 import InceptionV3
    model = InceptionV3(weights='imagenet', include_top=False)
    return _wrap_model(model, output_layer)


def _xception(output_layer='block14_sepconv2_act'):
    from keras.applications.xception import Xception
    model = Xception(weights='imagenet', include_top=False)
    return _wrap_model(model, output_layer)


def pretrained_models(model_name):
    """Returns a function constructing the specified pretrained model"""
    models = {
        'VGG16': _vgg_16,
        'VGG19': _vgg_19,
        'ResNet50': _res_net_50,
        'Inception-v3': _inception_v3,
        'Xception': _xception
    }
    return models.get(model_name)


def preprocessor(model_name):
    """Returns a function preprocessing an image before passing it to the net"""
    if model_name == 'VGG16':
        from keras.applications.vgg16 import preprocess_input
    elif model_name == 'VGG19':
        from keras.applications.vgg19 import preprocess_input
    elif model_name == 'ResNet50':
        from keras.applications.resnet50 import preprocess_input
    elif model_name == 'Inception-v3':
        from keras.applications.inception_v3 import preprocess_input
    elif model_name == 'Xception':
        from keras.applications.xception import preprocess_input
    return preprocess_input
