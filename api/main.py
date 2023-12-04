import os

from flask import Flask, request, jsonify, make_response, send_from_directory

import cv2
import imutils
import numpy as np
from imutils.contours import sort_contours

from tf_keras.models import load_model

import base64

def get_labeled_image(src):
    gray = cv2.cvtColor(src.copy(), cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 75, 200)

    contours = cv2.findContours(edges.copy(), cv2.RETR_EXTERNAL,
                                cv2.CHAIN_APPROX_SIMPLE)
    contours = imutils.grab_contours(contours)
    contours = sort_contours(contours, method="left-to-right")[0]

    res = []

    for c in contours:
        (x, y, w, h) = cv2.boundingRect(c)

        if (10 <= w <= 300) and (40 <= h <= 300):
            roi = gray[y:y + h, x:x + w]
            thresh = cv2.threshold(roi, 150, 255,
                                   cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
            (tH, tW) = thresh.shape

            if tW > tH:
                thresh = imutils.resize(thresh, width=32)

            # otherwise, resize along the height
            else:
                thresh = imutils.resize(thresh, height=32)

            (tH, tW) = thresh.shape
            dX = int(max(0, 32 - tW) / 2.0)
            dY = int(max(0, 32 - tH) / 2.0)

            padded = cv2.copyMakeBorder(thresh, top=dY, bottom=dY,
                                        left=dX, right=dX, borderType=cv2.BORDER_CONSTANT,
                                        value=(0, 0, 0))
            padded = cv2.resize(padded, (32, 32))

            padded = padded.astype("float32") / 255.0
            padded = np.expand_dims(padded, axis=-1)

            res.append((padded, (x, y, w, h)))

    boxes = [b[1] for b in res]
    res = np.array([c[0] for c in res], dtype="float32")

    chars = model.predict(res)

    labelNames = "0123456789"
    labelNames += "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    labelNames = [l for l in labelNames]

    final = src.copy()
    for (char, (x, y, w, h)) in zip(chars, boxes):
        i = np.argmax(char)
        label = labelNames[i]
        cv2.rectangle(final, (x, y), (x + w, y + h), (0, 0, 255), 2)
        cv2.putText(final, label, (x - 10, y - 10),
                    cv2.FONT_HERSHEY_COMPLEX, 2, (0, 0, 255), 2)

    return final


app = Flask(__name__, static_folder="../gatornum/build")


# Serve React App
@app.route('/', defaults={'path': ''})
@app.route('/<path:path>')
def serve(path):
    if path != "" and os.path.exists(app.static_folder + '/' + path):
        return send_from_directory(app.static_folder, path)
    else:
        return send_from_directory(app.static_folder, 'index.html')


@app.route('/scan', methods=['POST'])
def scan():
    request_json = request.get_json()

    if "image" not in request_json:
        return jsonify({"error", "no image provided"})

    blob64 = request_json["image"]

    encoded_data = blob64.split(',')[1]
    buff = np.frombuffer(base64.b64decode(encoded_data), np.uint8)
    image = cv2.imdecode(buff, cv2.IMREAD_COLOR)
    image = imutils.resize(image, width=800, height=600)
    labeledImage = get_labeled_image(image)
    cv2.imwrite("output.png", labeledImage)
    retval, buffer = cv2.imencode('.png', labeledImage)

    label64 = base64.b64encode(buffer).decode("utf-8")
    return jsonify({"labeled": label64})


if __name__ == '__main__':
    model = load_model("../model")
    app.run(debug=True)
