from keras.preprocessing import image
import numpy as np
from keras.models import load_model
model = load_model('E:/keras_data/data1/5class_model.h5')
# model.load_weights('my_model_weights.h5')
img_path='600.jpg'
img=image.load_img(img_path,target_size=(150,150))
x=image.img_to_array(img)
x=np.expand_dims(x,axis=0)
preds=model.predict(x)
print('Predicted:',preds[0])


