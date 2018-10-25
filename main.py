import tensorflow as tf
import matplotlib.pyplot as plt

mnist = tf.keras.datasets.mnist  # 28x28 images of hand-written digits 0-9
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# normalize data
x_train = tf.keras.utils.normalize(x_train, axis=0)
x_test = tf.keras.utils.normalize(y_train, axis=0)

# show first image 
# plt.imshow(x_train[0], cmap = plt.cm.binary)
# plt.show()
# print(x_train[0])

# build the 'sequential model'
model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dense(128, activation=tf.nn.relu))
model.add(tf.keras.layers.Dense(128, activation=tf.nn.relu))
model.add(tf.keras.layers.Dense(10, activation=tf.nn.softmax))

# training settings 
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# train the model
model.fit(x_train, y_train, epochs=3)

# save the model
model.save('epic_num_reader.model')
new_model = tf.keras.models.load_model('epic_num_reader.model')
predictions = new_model.predict([x_test])