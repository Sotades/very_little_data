from keras.models import model_from_json, Sequential
import cv2

# load json and create model
json_file = open('conv3_relu1_maxpool1.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
# load weights into new model
loaded_model.load_weights("conv3_relu1_maxpool1.h5")
print("Loaded model from disk")

def prepare(filepath):
    IMG_W = 210
    IMG_H = 1280
    img_array = cv2.imread(filepath, cv2.IMREAD_COLOR)

    return img_array.reshape(-1, IMG_W, IMG_H, 3)


prediction = loaded_model.predict_classes([prepare('data/new_images/NOK/copied_label_defect1.jpg')])

print(prediction)






