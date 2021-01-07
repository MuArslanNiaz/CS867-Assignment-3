# Downlaod and place data on Google Drive
# Mount with Drive and Get Dataset
from google.colab import drive
drive.mount('/content/drive')
!cp "/content/drive/MyDrive/cv/intel-image-classification.zip" /content/intel-image-classification.zip
!unzip -q intel-image-classification.zip # -q for quiet
!cp "/content/drive/MyDrive/cv/Test_data.zip" /content/Test_data.zip
!unzip -q Test_data.zip # -q for quiet 

# Load data and Apply Data Augmentation
path_train = "/content/seg_train/seg_train"
path_test = "/content/seg_test/seg_test"
path_pred = "/content/seg_pred/seg_pred"

from keras.preprocessing.image import ImageDataGenerator
batch_size=32
datagen_args = dict(rotation_range=20,width_shift_range=0.2,
    height_shift_range=0.2,rescale=1./255)
datagen_train = ImageDataGenerator(**datagen_args)
datagenerator_train = datagen_train.flow_from_directory(path_train,class_mode='categorical',
	target_size=(150,150),interpolation="lanczos",shuffle=True)datagen_test = ImageDataGenerator(**datagen_args)
datagenerator_test = datagen_test.flow_from_directory(path_test,class_mode='categorical',
	target_size=(150,150),interpolation="lanczos",shuffle=True)
datagen_pred = ImageDataGenerator(**datagen_args)
datagenerator_pred = datagen_pred.flow_from_directory(path_pred,class_mode='categorical',
	target_size=(150,150),interpolation="lanczos",shuffle=True)

# Imports for VGG model
from keras.applications.vgg16 import VGG16
from keras.models import Model
from keras.layers import Dense
from keras.layers import Flatten

# Load VGG16 model without classification layers
model = VGG16(include_top=False, input_shape=(150, 150, 3))

# Add new classification layers
flat1 = Flatten()(model.layers[-1].output) # flatten last layer
class1 = Dense(1024, activation='sigmoid')(flat1) # add FC layer on previous layer
output = Dense(6, activation='softmax')(class1) # add softmax layer

# Define the new model
model = Model(inputs=model.inputs, outputs=output)
model.summary()

# Display and save Network Diagram
from keras.utils.vis_utils import plot_model
plot_model(model, to_file='model_plot.png', show_shapes=True, show_layer_names=True)

# Complie Model
from keras.optimizers import SGD
sgd = SGD(lr=0.001, decay=1e-7, momentum=.9)
model.compile(loss='categorical_crossentropy',optimizer=sgd,metrics=['accuracy'])

# Train model
H = model.fit(datagenerator_train, batch_size=128,epochs=10,validation_data=(datagenerator_test))

# Draw Learning Curve
import numpy as np
import matplotlib.pyplot as plt
N = np.arange(0, 10)
plt.style.use('ggplot')
plt.figure()
plt.plot(N, H.history['loss'], label='train_loss')
plt.plot(N, H.history['val_loss'], label='val_loss')
plt.plot(N, H.history['accuracy'], label='train_accuracy')
plt.plot(N, H.history['val_accuracy'], label='val_accuracy')
plt.title('Loss/Accuracy Per Epoch Case 3')
plt.xlabel('Epoch #')
plt.ylabel('Loss/Accuracy')
plt.legend()
plt.show()

# Save the model's trained weights
model.save_weights('vgg_transfer_trained_wts_case_3.h5')
!cp "/content/vgg_transfer_trained_wts_case_3.h5" "/content/drive/MyDrive/cv/vgg_transfer_trained_wts_case_3.h5"

# Validate model
score = model.evaluate(datagenerator_test, batch_size=64)
print('Test Loss Case 3 = ', score[0])
print('Test Accuracy Case 3 = ', score[1])

# Draw Confusion Matrix
# Making prediction
y_pred = model.predict(datagenerator_pred)
y_true = np.argmax(y_pred, axis=-1)

from sklearn.metrics import confusion_matrix
confusion_mtx = confusion_matrix(y_true, np.argmax(y_pred, axis=1))
confusion_mtx