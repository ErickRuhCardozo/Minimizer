import re
import cv2
import time
import pickle
import webbrowser
import pygetwindow
from threading import Thread, Event


class Minimizer(Thread):
    def __init__(self, event: Event, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.event = event


    def run(self):
        regex = re.compile('Nota Paraná')

        while not self.event.is_set():
            windows = pygetwindow.getAllWindows()

            for window in windows:
                if regex.search(window.title) and not window.isMinimized:
                    try:
                        window.minimize()
                        break
                    except Exception:
                        pass

            time.sleep(1)
    

    def maximize(self):
        regex = re.compile('Nota Paraná')
        windows = pygetwindow.getAllWindows()

        for window in windows:
            if regex.search(window.title):
                window.maximize()
                break


def main():
    classifier = cv2.CascadeClassifier('haarcascades/haarcascade_frontalface_alt2.xml')
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.read('trained.yml')
    
    with open("labels.pkl", 'rb') as f:
        pkl = pickle.load(f)
        labels = {v: k for k, v in pkl.items()}
    
    last_label = None
    conf_frame_count = 0
    minimizer_event = Event()
    minimizer = Minimizer(minimizer_event, daemon=True)
    minimizer.start()
    cap = cv2.VideoCapture(0)

    while True:
        _, frame = cap.read()
        gray  = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = classifier.detectMultiScale(gray, scaleFactor=1.5, minNeighbors=4)

        for i, (x, y, w, h) in enumerate(faces):
            if i > 0:
                break
            
            roi_gray = gray[y:y+h, x:x+w]
            label_id, conf = recognizer.predict(roi_gray)

            if conf < 50:
                conf_frame_count += 1
                name = labels[label_id]

                if last_label != name:
                    conf_frame_count = 0

                last_label = name
                cv2.putText(frame, name, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)

            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)

        cv2.imshow('frame',frame)
        cv2.waitKey(1)

        if conf_frame_count > 10:
            minimizer_event.set()
            break

    minimizer.join() # Is this really necessary?
    minimizer.maximize()
    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()