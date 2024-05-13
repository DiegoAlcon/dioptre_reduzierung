# Opening a machine learning model

import pickle
#filename = r'C:\Users\SANCHDI2\OneDrive - Alcon\Desktop\Saved Models\UNet_bubbles.sav'
filename = 'UNet_bubbles.sav'
model = pickle.load(open(filename, 'rb'))

print('Hello world')