
import tensorflow as tf
import os
import numpy as np
# Remove dodgy images
import cv2
import imghdr
from matplotlib import pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout


os.path.join('data', 'bald')
os.listdir('data')
data_dir = 'data'

image_exts = ['jpeg','jpg','bmp','png']


for image_class in os.listdir(data_dir):
    for image in os.listdir(os.path.join(data_dir, image_class)):
        image_path = os.path.join(data_dir, image_class, image)
        try:
            img = cv2.imread(image_path)
            tip = imghdr.what(image_path)
            if tip not in image_exts:
                print('Image not in ext list {}'.format(image_path))
                os.remove(image_path)
        except Exception as e:
            print('Issue with image {}'.format(image_path))
            # os.remove(image_path)

data = tf.keras.utils.image_dataset_from_directory('data')
data_iterator = data.as_numpy_iterator()
data_iterator
# Get another batch from the iterator
batch = data_iterator.next()

# Images represented as numpy arrays
batch[0].shape

# Class 1 = Not Bald People
# Class 0 = Bald People 
batch[1]





fig, ax = plt.subplots(ncols=4, figsize=(20,20))
for idx, img in enumerate(batch[0][:4]):
    ax[idx].imshow(img.astype(int))
    ax[idx].title.set_text(batch[1][idx])





# Preprocess Data
# Scale data
data = data.map(lambda x, y: (x/225, y))




scaled_iterator = data.as_numpy_iterator()




scaled_iterator





batch = scaled_iterator.next()





batch[0].min()





fig, ax = plt.subplots(ncols=4, figsize=(20,20))
for idx, img in enumerate(batch[0][:4]):
    ax[idx].imshow(img)
    ax[idx].title.set_text(batch[1][idx])




# Split data
len(data)





train_size = int(len(data)*.6)
val_size = int(len(data)*.1)+1
test_size = int(len(data)*.1)+1





train_size + val_size + test_size




train = data.take(train_size)
val = data.skip(train_size).take(val_size)
test = data.skip(train_size+val_size).take(test_size)




# Build Deep learning model
train


model = Sequential()




model.add(Conv2D(16, (3,3), 1, activation='relu', input_shape=(256,256,3)))
model.add(MaxPooling2D())
model.add(Conv2D(32, (3,3), 1, activation='relu'))
model.add(MaxPooling2D())
model.add(Conv2D(16, (3,3), 1, activation='relu'))
model.add(MaxPooling2D())
model.add(Flatten())
model.add(Dense(256, activation='relu'))
model.add(Dense(1, activation='sigmoid'))





model.compile('adam', loss=tf.losses.BinaryCrossentropy(), metrics=['accuracy'])





model.summary()





# Train

logdir='logs'





tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=logdir)




hist = model.fit(train, epochs=20, validation_data=val)





# Plot Performance

fig = plt.figure()
plt.plot(hist.history['loss'], color='teal', label='loss')
plt.plot(hist.history['val_loss'], color='orange', label='val_loss')
fig.suptitle('Loss', fontsize=20)
plt.legend(loc="upper left")
plt.show()




fig = plt.figure()
plt.plot(hist.history['accuracy'], color='teal', label='accuracy')
plt.plot(hist.history['val_accuracy'], color='orange', label='val_accuracy')
fig.suptitle('Accuracy', fontsize=20)
plt.legend(loc="upper left")
plt.show()





# Evaluate
from tensorflow.keras.metrics import Precision, Recall, BinaryAccuracy





pre = Precision()
re = Recall()
acc = BinaryAccuracy()




for batch in test.as_numpy_iterator(): 
    X, y = batch
    yhat = model.predict(X)
    pre.update_state(y, yhat)
    re.update_state(y, yhat)
    acc.update_state(y, yhat)




print(pre.result(), re.result(), acc.result())





#Test
import cv2


img = cv2.imread('wotjr.jpeg')
plt.imshow(img)
plt.show()


resize = tf.image.resize(img, (256,256))
plt.imshow(resize.numpy().astype(int))
plt.show()



yhat = model.predict(np.expand_dims(resize/255, 0))

yhat

if yhat > 0.5: 
    print(f'Predicted class is Not Bald')
else:
    print(f'Predicted class is Bald')

