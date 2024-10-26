# KITTI-ObjectDetection
This repository contains code for training and evaluating object detection models using the KITTI dataset with TensorFlow.

## Dataset
The KITTI dataset is used for various vision tasks such as stereo, optical flow, and visual odometry. This repository focuses on the object detection dataset, which includes monocular images and 3D bounding boxes.

- **Training Images**: 6347
- **Validation Images**: 423
- **Testing Images**: 711

## Setup

Install the required packages:
```bash
pip install torch torchvision torchaudio diffusers transformers tensorflow_datasets
```

## Usage
1. Load the Dataset:
```
import tensorflow_datasets as tfds
dataset, info = tfds.load('kitti', with_info=True)
```

2. Preprocess the Data:
```
def preprocess(data):
    image = data['image']
    image = tf.image.resize_with_pad(image, 128, 128)
    image = tf.cast(image, tf.float32) / 255.0
    labels = tf.reduce_sum(tf.one_hot(data['objects']['type'], depth=8), axis=0)
    return image, labels
```

3. Train the Model:
```
from tensorflow.keras import layers, models
model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(128, 128, 3)),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dense(8, activation='sigmoid')
])
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
```

4. Evaluate the Model:
```
test_loss, test_acc = model.evaluate(test_dataset)
print(f'Test accuracy: {test_acc}')
```

### References

https://www.cvlibs.net/datasets/kitti/

### Citation
@inproceedings{Geiger2012CVPR,
  author = {Andreas Geiger and Philip Lenz and Raquel Urtasun},
  title = {Are we ready for Autonomous Driving? The KITTI Vision Benchmark Suite},
  booktitle = {Conference on Computer Vision and Pattern Recognition (CVPR)},
  year = {2012}
}


