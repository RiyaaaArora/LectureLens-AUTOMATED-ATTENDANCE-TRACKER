import cv2
import face_recognition
import os

#--- Load Known Faces ---
def load_known_faces():
    images = []
    classnames = []
    directory = "Photos"
import cv2
import face_recognition
import os

# --- Load Known Faces ---
def load_known_faces():
    images = []
    classnames = []
    directory = "Photos"

    # 🛡️ SAFETY FIX: Create folder if it doesn't exist
    if not os.path.exists(directory):
        os.makedirs(directory)

    for cls in os.listdir(directory):
        if os.path.splitext(cls)[1] in [".jpg", ".jpeg", ".png"]:
            img_path = os.path.join(directory, cls)
            curImg = cv2.imread(img_path)
            if curImg is not None:  # Check if image loaded correctly
                images.append(curImg)
                classnames.append(os.path.splitext(cls)[0])

    return images, classnames

# --- Encode Known Faces ---
def find_encodings(images):
    encode_list = []
    for img in images:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        encodings = face_recognition.face_encodings(img)
        if encodings:
            encode = encodings[0]
            encode_list.append(encode)
    return encode_list

# 🛡️ SAFETY FIX: Initialize these as empty lists first
# This prevents the app from crashing if no photos exist yet.
Images, classnames = load_known_faces()
    if Images:
        encodeListKnown = find_encodings(Images)
    else:
        encodeListKnown = []
    for cls in os.listdir(directory):
        if os.path.splitext(cls)[1] in [".jpg", ".jpeg", ".png"]:
            img_path = os.path.join(directory, cls)
            curImg = cv2.imread(img_path)
            images.append(curImg)
            classnames.append(os.path.splitext(cls)[0])

    return images, classnames

#--- Encode Known Faces ---

def find_encodings(images):
    encode_list = []
    for img in images:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        encodings = face_recognition.face_encodings(img)
        if encodings:  # Check if any faces were found
            encode = encodings[0]  # Take the first face (assuming one per image)
            encode_list.append(encode)
        else:
            print("Warning: No faces detected in an image. Skipping this image.")  # Optional: Log or handle
    return encode_list

Images, classnames = load_known_faces()
if Images:
    encodeListKnown = find_encodings(Images)
else:
    encodeListKnown = []