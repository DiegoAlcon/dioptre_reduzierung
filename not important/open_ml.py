# Opening a machine learning model

#import pickle
import tensorflow as tf

#filename = r'C:\Users\SANCHDI2\OneDrive - Alcon\Desktop\Saved Models\UNet_bubbles.sav'
#filename = 'UNet_bubbles.sav'
#model = pickle.load(open(filename, 'rb'))

#with open("my_file.pkl", "rb") as f:
#    loaded_object = pickle.load(f)

#with open("UNet_bubbles.sav", "rb") as f:
#    loaded_object = pickle.load(f)

model = tf.keras.models.load_model('UNet_bubbles.keras')

print('Hello world')