# Keras Implementation of Mish Activation Function.

"""Update: 11/17/2019- Keras implementation is now depreciated, please move to TFKeras directory for the Tensorflow 2.0 Keras based implementation"""

# Import Necessary Modules.
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from keras.layers import Activation
from keras.utils.generic_utils import get_custom_objects
import keras.backend as K

class Mish(Activation):
    '''
    Mish Activation Function.
    .. math::
        mish(x) = x * tanh(softplus(x)) = x * tanh(ln(1 + e^{x}))
    Shape:
        - Input: Arbitrary. Use the keyword argument `input_shape`
        (tuple of integers, does not include the samples axis)
        when using this layer as the first layer in a model.
        - Output: Same shape as the input.
    Examples:
        >>> X = Activation('Mish', name="conv1_act")(X_input)
    '''

    def __init__(self, activation, **kwargs):
        super(Mish, self).__init__(activation, **kwargs)
        self.__name__ = 'Mish'

def mish(x):
    return x*K.tanh(K.softplus(x))

get_custom_objects().update({'Mish': Mish(mish)})

###############################################################################
class Relu6(Activation):

    def __init__(self, activation, **kwargs):
        super(Relu6, self).__init__(activation, **kwargs)
        self.__name__ = 'Relu6'

def relu6(x):
    return K.relu(x, max_value=6.0)

get_custom_objects().update({'Relu6': Relu6(relu6)})

###############################################################################
class Hswish(Activation):

    def __init__(self, activation, **kwargs):
        super(Hswish, self).__init__(activation, **kwargs)
        self.__name__ = 'Hswish'

def hard_swish(x):
    """Hard swish
    """
    return x * K.relu(x + 3.0, max_value=6.0) / 6.0

get_custom_objects().update({'Hswish': Hswish(hard_swish)})