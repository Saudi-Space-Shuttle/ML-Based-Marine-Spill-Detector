# ML-Based Marine Spill Detector
- to locate oil spills using open data

## Proposal 

The proposal of the project is published in the NASA Space Apps challenge website and is summurized [here](Proposal.md).


## Fitting a Baseline Model

### 1. Prerequisites
I. [Python/Conda Distribution](https://docs.conda.io/en/latest/miniconda.html)
    
II. [Tensorflow](https://docs.anaconda.com/anaconda/user-guide/tasks/tensorflow/)
```
conda install -c anaconda tensorflow-gpu
```

III. [SSS_02 Extracted Samples (SSS_02_delta.zip)](/data)

If you're using Linux, open the Terminal and enter:
 ```terminal
 wget "https://www.dropbox.com/s/t0diyq5y8onun77/SSS_02_delta.zip?dl=1"
 ```

### 2. Prepare Data Generator using Tensorflow

Data augmentation is a standard practice in imagery and big data and is available in the [ImageDataGenerator](https://www.tensorflow.org/api_docs/python/tf/keras/preprocessing/image/ImageDataGenerator) class.


```Python
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator

dataset_directory = '/home/username/SSS_02_delta'
image_size = (100, 100)
batch_size = 50

gen = ImageDataGenerator(
    horizontal_flip = True,
    vertical_flip = True,
    rotation_range = 5,
    height_shift_range = 2,
    width_shift_range = 2,
)

train_generator = gen.flow_from_directory(
    dataset_directory+'/train',
    target_size=image_size,
    batch_size=batch_size,
    class_mode='binary',
    color_mode = 'grayscale',
)

valid_generator = gen.flow_from_directory(
    dataset_directory+'/valid',
    target_size=image_size,
    batch_size=batch_size,
    class_mode='binary',
    color_mode = 'grayscale',
)
```




