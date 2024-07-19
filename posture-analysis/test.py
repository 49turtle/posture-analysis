# from tensorflow.compiler.tf2tensorrt.wrap_py_utils import get_linked_tensorrt_version
# from tensorflow.compiler.tf2tensorrt.wrap_py_utils import get_loaded_tensorrt_version

# compiled_version = get_linked_tensorrt_version()
# loaded_version = get_loaded_tensorrt_version()

# print("Linked TensorRT version: %s" % str(compiled_version))
# print("Loaded TensorRT version: %s" % str(loaded_version))


# from tensorflow.python.compiler.tensorrt import trt_convert as trt


import tensorrt
# print(tensorrt.__file__)
print(tensorrt.__version__)

import tensorflow as tf
print(tf.__version__)
# print(tf.config.list_physical_devices())

# import tensorflow as tf
# from tensorflow.python.compiler.tensorrt import trt_convert as trt

# print("TensorFlow version:", tf.__version__)
# try:
#     trt_version = trt.trt_utils.trt_version()
#     print("TensorRT version:", trt_version)
# except AttributeError as e:
#     print("Error fetching TensorRT version:", e)
