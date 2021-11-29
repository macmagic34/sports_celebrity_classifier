from typing import final
import cv2
import numpy as np
import base64
import json
import joblib
from wavelet import w2d

np.set_printoptions(suppress=True)

__model = None
__celebrity_to_number = {}
__number_to_celebrity = {}

def get_image_from_b64_string(b64_image_string):
    encoded_data = b64_image_string.split(',')[1]
    nparr = np.frombuffer(base64.b64decode(encoded_data), np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    return img


def cropped_images_with_face_and_eye(image_path, b64_image_string):

    if image_path:
        color_img = cv2.imread(image_path)
    else:
        color_img = get_image_from_b64_string(b64_image_string)
    gray_img = cv2.cvtColor(color_img, cv2.COLOR_BGR2GRAY)
    face_cascade = cv2.CascadeClassifier(r'C:\Users\TR3X\anaconda3\envs\sportceleb\Library\etc\haarcascades\haarcascade_frontalface_default.xml')
    eye_cascade = cv2.CascadeClassifier(r'C:\Users\TR3X\anaconda3\envs\sportceleb\Library\etc\haarcascades\haarcascade_eye.xml')
    faces_detected = face_cascade.detectMultiScale(gray_img, 1.2, 5)
    cropped_faces = []
    for (x, y, w, h) in faces_detected:
        cropped_gray_face = gray_img[y:y+h, x:x+w]
        cropped_color_face = color_img[y:y+h, x:x+w]
        eyes_detected = eye_cascade.detectMultiScale(cropped_gray_face, 1.2, 5)
        if len(eyes_detected) >= 2:
            cropped_faces.append(cropped_color_face)

    return cropped_faces

def classify_image(b64_image, file_path=None):

    imgs = cropped_images_with_face_and_eye(file_path, b64_image)
    result = []
    for img in imgs:
        scaled_raw_img = cv2.resize(img, (32, 32))
        wavelet_transformed_img = w2d(img, 'db1', 5)
        scaled_wavelet_transformed_img = cv2.resize(wavelet_transformed_img, (32, 32))
        combined_image = np.vstack((scaled_raw_img.reshape(32*32*3, 1), scaled_wavelet_transformed_img.reshape(32*32, 1)))
        len_image_array = 32*32*3 + 32*32
        X = np.array(combined_image).reshape(1, len_image_array).astype(float)
        result.append({
            'class' : __number_to_celebrity[__model.predict(X)[0]],
            'class_probability' : np.round(__model.predict_proba(X)*100, 2).tolist()[0],
            'class_dictionary' : __celebrity_to_number
            })
            
    return result



def load_artifacts():
    print('Loading artifacts.... Started')
    global __celebrity_to_number
    global __number_to_celebrity
    with open(r'D:\My Projects\Sports celebrity classification\Server\artifacts\class_dict.json') as f:
        __celebrity_to_number = json.load(f)
        __number_to_celebrity = {x:y for y,x in __celebrity_to_number.items()}
    global __model
    if __model is None:
        __model = joblib.load(r'D:\My Projects\Sports celebrity classification\Server\artifacts\face_recog_model_svm.pkl')
    print("Artifacts loaded successfully")


def get_b64_image_for_virat():
    with open(r'D:\My Projects\Sports celebrity classification\Server\b64.txt') as f:
        return f.read()


if __name__ == '__main__':
    load_artifacts()
    #print(classify_image(get_b64_image_for_virat(), None))
    print(classify_image(None, r'D:\My Projects\Sports celebrity classification\Server\test_images\Virat_Kohli_Anushka_Sharma_4.jpg'))
    #print(classify_image(None, None))
    #print(classify_image(None, None))
