import cv2
from sklearn.datasets import fetch_olivetti_faces

# Images and model path
vid = cv2.VideoCapture(0)
cascPath = r"models/haarcascade_frontalface_default.xml"

# Create the haar cascade
faceCascade = cv2.CascadeClassifier(cascPath)


def face_detect():
    print("pressione 'Q' para sair")
    faces_num = 0

    while(True):
        # Pressione "Q" para sair
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        #frame = cv2.imread("data/fiaputas_img.png")
        ret, frame = vid.read()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        faces = faceCascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30),
            flags = cv2.CASCADE_SCALE_IMAGE
        )

        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

        cv2.imshow('Orwell_Product', frame)
        cv2.imshow('Orwell_Gray', gray)

        faces_num = len(faces)

    print(f"Found {faces_num} faces!")
    vid.release()
    cv2.destroyAllWindows()

def get_contourns():
    print("pressione 'Q' para sair")
    
    while(True):
        # Pressione "Q" para sair
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        #ret, frame = vid.read()
        frame = cv2.imread("data/fiaputas_img.png")
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blur = cv2.blur(gray, (3,3))

        cv2.imshow('Orwell_Product', frame)
        cv2.imshow('Orwell_Gray', gray)
        cv2.imshow('Orwell_Blur', gray)

if __name__ == "__main__":
    face_detect()