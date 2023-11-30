import cv2

image = cv2.imread('Test.jpg')

gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
ret, thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY_INV)
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (150, 2))
mask = cv2.morphologyEx(thresh, cv2.MORPH_DILATE, kernel)

mask = cv2.resize(mask, None, fx=0.25, fy=0.25)

cv2.imshow("Mask Image", mask)
cv2.waitKey()

bboxes = []
bboxes_img = image.copy()
bboxes_img = cv2.resize(bboxes_img, None, fx=0.25, fy=0.25)
contours = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
contours = contours[0] if len(contours) == 2 else contours[1]
for cntr in contours:
    x, y, w, h = cv2.boundingRect(cntr)
    cv2.rectangle(bboxes_img, (x, y), (x + w, y + h), (0, 0, 255), 1)
    bboxes.append((x, y, w, h))


cv2.imshow("Boxes", bboxes_img)
cv2.waitKey()
