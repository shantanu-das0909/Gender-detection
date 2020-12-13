import cv2
import numpy as np
from tensorflow.keras.models import load_model


model = load_model("gender_detector.model")
cap = cv2.VideoCapture(0)

CLASS_MAPPER = {
    0: "MALE",
    1: "FEMALE"
}


def mapper(val):
    return CLASS_MAPPER[val]


while True:
    success, frame = cap.read()
    # cv2.rectangle(frame, (100, 100), (500, 500), (255, 255, 255), 2)

    image = frame
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (224, 224))
    image = np.array([image])

    prediction = model.predict(image)
    class_index = np.argmax(prediction)
    class_name = mapper(class_index)

    cv2.putText(frame, class_name, (150, 50), cv2.FONT_HERSHEY_COMPLEX, 0.7, (0, 0, 255), 2, cv2.LINE_AA)

    cv2.imshow("video", frame)

    key = cv2.waitKey(10)
    if key == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()