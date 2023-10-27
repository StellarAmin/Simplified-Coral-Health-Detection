import cv2
import numpy as np
import library

# cap = cv2.VideoCapture('rtsp://192.168.1.20:554/user=admin_password=tlJwpbo6_channel=1_stream=0.sdp?real_stream')
cap = cv2.VideoCapture("cameraCapture.mp4")
allow = 0
prespectiveDims = np.float32([[0, 0], [600, 0], [0, 600], [600, 600]])
matrixFO = np.array(
    [
        [1.09289617e00, 2.37310172e-15, -3.82513661e01],
        [7.77156117e-16, 1.02214651e00, -1.02214651e01],
        [3.93023288e-18, 6.47810798e-18, 1.00000000e00],
    ]
)


fullOriginalIMG = cv2.imread("full_original.png")
fullPinkMask = cv2.cvtColor(cv2.imread("mask_pink.png"), cv2.COLOR_BGR2GRAY)
fullWhiteMask = cv2.cvtColor(cv2.imread("mask_white.png"), cv2.COLOR_BGR2GRAY)
fullOriginalMask = cv2.cvtColor(cv2.imread("maskFO.png"), cv2.COLOR_BGR2GRAY)
croppedOriginalMask = cv2.cvtColor(cv2.imread("maskO.png"), cv2.COLOR_BGR2GRAY)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.GaussianBlur(frame, (5, 5), 0)
    frameCopy = frame.copy()
    pinkMask, fullMask, whiteMask = library.getMask(frame)

    boundaries = library.detectCoralBoards(fullMask)
    y, yp, x, xp = 0, 30, 0, 30
    for i in boundaries[4]:
        size = cv2.contourArea(i)
    if boundaries is not None:
        x1, y1, x2, y2, area = boundaries
        cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 255, 0), 2)
        pts1 = np.float32([[x1, y1], [x2, y1], [x1, y2], [x2, y2]])
        matrix = cv2.getPerspectiveTransform(pts1, prespectiveDims)
        pinkCoral = cv2.warpPerspective(pinkMask, matrix, (600, 600))
        whiteCoral = cv2.warpPerspective(whiteMask, matrix, (600, 600))
        fullCoral = cv2.warpPerspective(fullMask, matrix, (600, 600))
        coloredCoral = cv2.warpPerspective(frame, matrix, (600, 600))

    if cv2.waitKey(1) & 0xFF == ord("s"):
        fullOriginalIMG = cv2.imread("full_original.png")
        fullResult = fullOriginalIMG.copy()
        fullCoralS = fullCoral.copy()
        fullCoral, fullOriginalMask, heightDiff, fullOriginalIMG = library.setSize(
            fullCoral, fullOriginalMask, fullOriginalIMG
        )

        library.getGrowth(
            fullCoral, fullOriginalMask, heightDiff, fullOriginalIMG
        )  # green
        library.getWhite(whiteCoral, fullWhiteMask, heightDiff, fullOriginalIMG)  # red
        library.getPink(pinkCoral, fullPinkMask, heightDiff, fullOriginalIMG)  # blue
        library.getDecay(
            fullCoralS, croppedOriginalMask, fullResult, heightDiff, fullOriginalIMG
        )  # yellow

    cv2.imshow("Past Coral", fullOriginalIMG)
    cv2.imshow("Cropped Coral", coloredCoral)
    cv2.imshow("Present Coral", frame)
    # cv2.imshow("Mask", fullOriginalMask)
    # cv2.imshow("Transformation", fullCoral)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break
