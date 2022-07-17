## Object classification using VGG16
#### Determine object from image through convolution neural network

**VGG16 -** VGG16 is object detection and classification algorithm which is able to classify 1000 images of 1000 different categories with 92.7% accuracy. It is one of the popular algorithms for image classification and is easy to use with transfer learning.

![VGG16](https://github.com/Abhishek-k-git/Object_classification/blob/main/images/VGG16.jpg)
     
> Libraries used

 1. Tensorflow : TensorFlow is an open source software library for high performance numerical computation. Its flexible architecture allows easy deployment of computation across a variety of platforms (CPUs, GPUs, TPUs), and from desktops to clusters of servers to mobile and edge devices.
 Originally developed by researchers and engineers from the Google Brain team within Google's AI organization, it comes with strong support for     machine learning and deep learning and the flexible numerical computation core is used across many other scientific domains.
 2. Keras : Keras is a high-level neural networks API for Python.Keras is compatible with Python 3.6+
 <span style="color:green">It comes preinstalled with tensorflow </span>

> Installation

 ```python
 pip install tensorflow-gpu==2.0.0-rc0
 ```

> Model Building
```python
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input, decode_predictions
from tensorflow.keras.preprocessing.image import load_img, img_to_array

model = VGG16()
model.summary()
```

**Getting images from local directory and applying to model for prediction
```python
#import os

for file in os.listdir('Objects'):
    print(file)
    full_path = 'Objects/'+file
    image = load_img(full_path, target_size = (224, 224))
    image = img_to_array(image)
    image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))
    image = preprocess_input(image)
    y_pred = model.predict(image)
    label = decode_predictions(y_pred, top=1)
    print(label)
```

**Final prediction**
```
alarm clock.jpg
[[('n02708093', 'analog_clock', 0.5168703)]]
alarm clock2.jpg
[[('n04328186', 'stopwatch', 0.8668445)]]
camera.jpg
[[('n04069434', 'reflex_camera', 0.35195756)]]
camera2.jpg
[[('n04069434', 'reflex_camera', 0.95981157)]]
chair.jpg
[[('n04099969', 'rocking_chair', 0.9997309)]]
chair2.jpg
[[('n03376595', 'folding_chair', 0.29405418)]]
coffee cup.jpg
[[('n07930864', 'cup', 0.7228752)]]
cycle.jpg
[[('n02835271', 'bicycle-built-for-two', 0.74734783)]]
head phone.jpg
[[('n04317175', 'stethoscope', 0.63425094)]]
lens.jpg
[[('n03657121', 'lens_cap', 0.9950363)]]
mask.jpg
[[('n02909870', 'bucket', 0.14818962)]]
table.jpg
[[('n03179701', 'desk', 0.17381892)]]
```
