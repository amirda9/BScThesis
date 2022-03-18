import cv2
import Detector
import dlib
import numpy as np
import detectPupil


cascade = cv2.CascadeClassifier("Da9_itis/haarcascade_eye.xml")
detector_cas = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(
    'Da9_itis/shape_predictor_68_face_landmarks.dat')


def createEyeMask(eyeLandmarks, im):
    leftEyePoints = eyeLandmarks
    eyeMask = np.zeros_like(im)
    cv2.fillConvexPoly(eyeMask, np.int32(leftEyePoints), (255, 255, 255))
    for i in range(eyeMask.shape[0]):
        for j in range(eyeMask.shape[1]):
            if eyeMask[i, j,0] == 255:
                eyeMask[i, j] = frame[i,j]
    eyeMask = np.uint8(eyeMask)
    return eyeMask


def findIris(eyeMask, im, thresh):
    r = im[:, :, 2]
    _, binaryIm = cv2.threshold(r, thresh, 255, cv2.THRESH_BINARY_INV)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (4, 4))
    morph = cv2.dilate(binaryIm, kernel, 1)
    morph = cv2.merge((morph, morph, morph))
    morph = morph.astype(float)/255
    eyeMask = eyeMask.astype(float)/255
    iris = cv2.multiply(eyeMask, morph)
    return iris


def findCentroid(iris):
    M = cv2.moments(iris[:, :, 0])
    if M['m00'] != 0:
        cX = int(M["m10"] / M["m00"])
        cY = int(M["m01"] / M["m00"])
        centroid = (cX, cY)
        return centroid
    else:
        print('No centroid found')
    


def createIrisMask(iris, centroid):
    cv2.imshow('create iris mask',iris[:, :, 0])
    cnts, _ = cv2.findContours(np.uint8(iris[:, :, 0]), cv2.RETR_EXTERNAL,
                               cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(iris, cnts, -1, (255, 255, 255), 1)
    cv2.circle(iris, centroid, 10, (255, 255, 255), -1)
    return iris,iris
    print(cnts)
    flag = 10000
    final_cnt = None
    try:
        for cnt in cnts:
            (x, y), radius = cv2.minEnclosingCircle(cnt)
            distance = abs(centroid[0]-x)+abs(centroid[1]-y)
            if distance < flag:
                flag = distance
                final_cnt = cnt
            else:
                continue
        (x, y), radius = cv2.minEnclosingCircle(final_cnt)
        center = (int(x), int(y))
        radius = int(radius) - 2

        irisMask = np.zeros_like(iris)
        inverseIrisMask = np.ones_like(iris)*255
        cv2.circle(irisMask, center, radius, (255, 255, 255), -1)
        cv2.circle(inverseIrisMask, center, radius, (0, 0, 0), -1)
        irisMask = cv2.GaussianBlur(irisMask, (5, 5), cv2.BORDER_DEFAULT)
        inverseIrisMask = cv2.GaussianBlur(
            inverseIrisMask, (5, 5), cv2.BORDER_DEFAULT)

        return irisMask, inverseIrisMask
    except:
        print('shit happened')
        return None,None


def changeEyeColor(im, irisMask, inverseIrisMask):
    imCopy = cv2.applyColorMap(im, cv2.COLORMAP_TWILIGHT_SHIFTED)
    imCopy = imCopy.astype(float)/255
    irisMask = irisMask.astype(float)/255
    inverseIrisMask = inverseIrisMask.astype(float)/255
    im = im.astype(float)/255
    faceWithoutEye = cv2.multiply(inverseIrisMask, im)
    newIris = cv2.multiply(irisMask, imCopy)
    result = faceWithoutEye + newIris
    return result


def float642Uint8(im):
    im2Convert = im.astype(np.float64) / np.amax(im)
    im2Convert = 255 * im2Convert
    convertedIm = im2Convert.astype(np.uint8)
    return convertedIm


if __name__ == '__main__':
    # print('im here')
    cap = cv2.VideoCapture(0)
    windowName = "EyePaint"
    trackbarValue = "Threshold"
    cv2.namedWindow(windowName, cv2.WINDOW_FULLSCREEN)
    detector = Detector.CascadeDetector()
    cv2.createTrackbar(trackbarValue, windowName, 0, 255, detector.on_trackbar)
    while True:
        _, frame = cap.read()
        frame = cv2.flip(frame, 1)
        # frame = detector.find_eyes(frame)
        dets = detector_cas(frame, 1)
        landmarks=[]
        for k, d in enumerate(dets):
            shape = predictor(frame, d)
            left_eye_Pts = np.array([[shape.part(36).x, shape.part(36).y],
                                         [shape.part(37).x, shape.part(37).y],
                                         [shape.part(38).x, shape.part(38).y],
                                         [shape.part(39).x, shape.part(39).y],
                                         [shape.part(40).x, shape.part(40).y],
                                         [shape.part(41).x, shape.part(41).y]], dtype=np.float32)
            right_eye_Pts = np.array([[shape.part(42).x, shape.part(42).y],
                                          [shape.part(43).x, shape.part(43).y],
                                          [shape.part(44).x, shape.part(44).y],
                                          [shape.part(45).x, shape.part(45).y],
                                          [shape.part(46).x, shape.part(46).y],
                                          [shape.part(47).x, shape.part(47).y]], dtype=np.float32)
            landmarks.append(left_eye_Pts)
            landmarks.append(right_eye_Pts)
        if len(landmarks) == 2:
            leftEyeLandmarks = landmarks[0]
            rightEyeLandmarks = landmarks[1]
            leftEyeMask = createEyeMask(leftEyeLandmarks, frame)
            rightEyeMask = createEyeMask(rightEyeLandmarks, frame)
            leftEyeIris = findIris(leftEyeMask, frame, cv2.getTrackbarPos(trackbarValue, windowName))
            rightEyeIris = findIris(rightEyeMask, frame, cv2.getTrackbarPos(trackbarValue, windowName))
            eyes = leftEyeMask + rightEyeMask
            leftEyeCentroid = findCentroid(leftEyeIris)
            rightEyeCentroid = findCentroid(rightEyeIris)
            leftEyeIrisMask, leftEyeInverseIrisMask = createIrisMask(leftEyeIris, leftEyeCentroid)
            rightEyeIrisMask, rightEyeInverseIrisMask = createIrisMask(rightEyeIris, rightEyeCentroid)
            res = leftEyeIris + rightEyeIris
            res = cv2.circle(res, leftEyeCentroid, 5, (0, 255, 0),-1)
            res = cv2.circle(res, rightEyeCentroid, 5, (0, 255, 0), -1)
            for i in range(68):
                res = cv2.circle(res, (shape.part(i).x, shape.part(
                    i).y), 1, (0, 255, 0), thickness=3)
        cv2.imshow('frame', frame)
        cv2.imshow('leftEyeIris', res)
		# print(frame.shape)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

		# im = cv2.copyTo(frame,None)
		# # Create eye mask using eye landmarks from facial landmark detection
		# leftEyeMask = createEyeMask(landmarks[36:42], im)
		# rightEyeMask = createEyeMask(landmarks[43:49], im)

		# # Find the iris by thresholding the red channel of the image within the boundaries of the eye mask
		# leftIris = findIris(leftEyeMask, im, 38)
		# rightIris = findIris(rightEyeMask, im, 45)

		# # Find the centroid of the binary image of the eye
		# leftIrisCentroid = findCentroid(leftIris)
		# rightIrisCentroid = findCentroid(rightIris)

		# # Generate the iris mask and its inverse mask
		# leftIrisMask, leftInverseIrisMask = createIrisMask(leftIris, leftIrisCentroid)
		# rightIrisMask, rightInverseIrisMask = createIrisMask(rightIris, rightIrisCentroid)

		# # Change the eye color and merge it to the original image
		# coloredEyesLady = changeEyeColor(im, rightIrisMask, rightInverseIrisMask)
		# coloredEyesLady = float642Uint8(coloredEyesLady)
		# coloredEyesLady = changeEyeColor(coloredEyesLady, leftIrisMask, leftInverseIrisMask)

		# # Present results
		# cv2.imshow("", coloredEyesLady)
    cap.release()
    cv2.destroyAllWindows()


