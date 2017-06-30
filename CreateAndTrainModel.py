# Türkçe olan kısımları ben ekledim, diğer her şey kaynaktan alınmıştır.
from __future__ import print_function

#import os
#os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   # see issue #152
#os.environ["CUDA_VISIBLE_DEVICES"] = ""

import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K
import numpy as np


# Tek seferde ne kadar verimizin ağdan geçeğini belirtir
# RAMimiz yüksekse yüksek tutmakta fayda var, daha hızlı öğrenme sağlar
batch_size = 128

# Verimizin kaç sınıfa ayrıldığı 
# Mnist için sıfırdan dokuza kadar 10 veri sınıfımız var
num_classes = 10

# Verimiz yapay sinir ağı üzerinden kaç kez geçecek.
# Kaynakta 12 epoch olarak verilmiş fakat hızlı olması için 2 epochda bitiriyorum
epochs = 5

# input image dimensions
img_rows, img_cols = 28, 28

# Veri 4 parçaya bölünmüş olarak ve 
# Her bir görsel için 28x28 boyutunda bir matris olarak verimizi alabiliriz
# the data, shuffled and split between train and test sets
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Beyaz Arkplanları Siyaha çevirip tekrar ekledik
x_train = np.concatenate((x_train,255 - x_train), axis=0)
y_train = np.concatenate((y_train, y_train), axis=0)
x_test = np.concatenate((x_test,255 - x_test), axis=0)
y_test = np.concatenate((y_test,y_test), axis=0)

# Veriyi channel bilgisine göre tekrar düzenliyoruz.
# Tensorflow ve Theano farklı veri şekilleri ile çalışıyorlar 
if K.image_data_format() == 'channels_first':
    x_train = x_train.reshape(x_train.shape[0], 1, img_rows, img_cols)
    x_test = x_test.reshape(x_test.shape[0], 1, img_rows, img_cols)
    input_shape = (1, img_rows, img_cols)
else:
    x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
    x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
    input_shape = (img_rows, img_cols, 1)

x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255
print('x_train shape:', x_train.shape)
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')

# y değerlerimiz x verimizin hangi rakama yani hangi sınıfa ait olduğunu tutan rakamlardı
# Yani y verilerimizin birinci boyutu x içerisindeki indexe karşılık gelirken
# ikinci boyutu 0 indexi alabiliyor ve y, 0-9 aralığında tek değer alabiliyordu
# Burada to_categorical metodu ile verimizin ikinci boyutunu sıfır ve birlerden oluşan
# 10 değer alabilecek şekilde genişletiyoruz.
# Örneğin bir verimizin y değeri 8 ise, yeni şekliyle 9 adet sıfır ve 
# yedinci(indexler sıfırdan başladığı için sekizinci değil) indexi 1 olacak hale getiriyoruz

# convert class vectors to binary class matrices
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

# Keras Sequential sınıfı Yapay sinir ağımızın temelidir
# Bu sınıfa istediğimiz kadar katman ekleyerek tüm ağı oluşturabiliriz
# Hangi katmanı ne zaman eklenmesi gerektiği ile ilgili, 
# en az kayıp vereni bulana kadar deneme-yanılma yapılır
model = Sequential()

# Buradaki adımları aşağıda açıklamaya çalıştım
model.add(Conv2D(32, kernel_size=(3, 3),
                 activation='relu',
                 input_shape=input_shape))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(num_classes, activation='softmax'))

model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.Adadelta(),
              metrics=['accuracy'])

model.fit(x_train, y_train,
          batch_size=batch_size,
          epochs=epochs,
          verbose=1,
          validation_data=(x_test, y_test))
score = model.evaluate(x_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])

model_json = model.to_json()
with open("MyModel.json","w") as json_file:
    json_file.write(model_json)
model.save_weights("MyModel.h5")
