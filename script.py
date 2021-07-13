import tensorflow as tf
import tensorflow_datasets as tfds

print("Hello World ML!")

(train_data, test_data), data_info = tfds.load(
    'mnist', # using that handwritten number data set
    split = ['train', 'test'],
    shuffle_files = True,
    as_supervised = True,
    with_info = True
)

def normalize_img(image, label):
    """
        required step because in the machine learning model
        all uint8's are turned into float32's anyways. specifying them
        in the beginning and making all numbers the same datatype throughout is
        good practice - stackoverflow (idk lol)
    """

    # convert to tf.float32

    return tf.cast(image, tf.float32) / 255., label

# normalize all the images to be float32's - think javascript Array.map. this is something like that.
train_data = train_data.map(
    normalize_img, num_parallel_calls=tf.data.experimental.AUTOTUNE
)

"""
cache transformation can cache a dataset, 
either in memory or on local storage. 
This will save some operations (like file opening and data reading) 
from being executed during each epoch. The next epochs will 
reuse the data cached by the cache transformation. - stackoverflow

basically makes it so you don't have to read the data for every epoch while training
"""

train_data = train_data.cache()

train_data = train_data.shuffle(data_info.splits['train'].num_examples)
train_data = train_data.batch(128)

# makes the processing and training overlap to save time
train_data = train_data.prefetch(tf.data.experimental.AUTOTUNE)

test_data = test_data.map(
    normalize_img, num_parallel_calls=tf.data.experimental.AUTOTUNE)
test_data = test_data.batch(128)
test_data = test_data.cache()
test_data = test_data.prefetch(tf.data.experimental.AUTOTUNE)

"""
Flatten = convert the data into a one dimensional array
1D Array ex: [1, 2, 3, 4]

Dense = Layer with x number of nodes, all of them are interconnected hence the name Dense
"meaning all the neurons in a layer are connected to those in the next layer."

"An activation function in a neural network defines how the weighted sum of the input is transformed into an output from a node or nodes in a layer of the network"

relu = Rectified Linear Unit
The rectified linear activation function or ReLU for short is a piecewise 
linear function that will output the input directly if it is positive, 
otherwise, it will output zero. ... The rectified linear activation function 
overcomes the vanishing gradient problem, allowing models to learn faster and perform better.

after the first layer you do not need to specify the input shape

Sequential is a model type where layers have exactly one input and output tensor
"""

model = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28)),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(10)
])


"""
optimizer = the function that edits the weights of the network until its perfect
loss = the function that calculates how far off the prediction was
metrics = the stats the model logs out
"""



model.compile(
    optimizer=tf.keras.optimizers.Adam(0.001),
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=["accuracy"]
)

"""
train model
epochs = number of cycles
validation_data = testing data
"""

model.fit(
    train_data,
    epochs=6,
    validation_data=test_data
)
