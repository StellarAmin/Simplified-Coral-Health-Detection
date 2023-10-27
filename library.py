import cv2
import numpy as np
import time

y, yp, x, xp = 0, 30, 0, 30
axis = [
    (100, 50, 170, 190),
    (160, 140, 250, 230),
    (230, 60, 275, 200),
    (280, 150, 340, 230),
    (350, 0, 400, 170),
    (460, 50, 520, 175),
    (350, 170, 420, 315),
    (420, 160, 460, 290),
    (520, 220, 600, 290),
    (20, 60, 100, 280),
    (40, 280, 90, 400),
    (140, 220, 160, 380),
    (160, 330, 250, 450),
    (220, 240, 280, 440),
    (165, 330, 220, 440),
    (280, 300, 360, 460),
    (350, 320, 400, 450),
    (460, 300, 530, 445),
    (550, 320, 600, 435),
    (10, 430, 90, 570),
    (100, 440, 160, 570),
    (210, 440, 280, 570),
    (335, 470, 410, 570),
    (460, 450, 525, 560),
]  # x1 y1 x2 y2 n=24
br = [[0 for x in range(20)] for y in range(20)]
br1 = [[0 for x in range(20)] for y in range(20)]
br3 = [[0 for x in range(20)] for y in range(20)]
growth = [[0 for x in range(100)] for y in range(100)]
diffWhite = [0 for x in range(24)]
diffPink = [0 for x in range(24)]
diffDecay = [0 for x in range(24)]


def getMask(image):
    frameHSV = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    pinkMask = cv2.inRange(frameHSV, (0, 60, 70), (10, 255, 255))
    pinkMask += cv2.inRange(frameHSV, (170, 60, 70), (180, 255, 255))
    frameHSV = cv2.bitwise_not(image)
    whiteMask = cv2.inRange(frameHSV, (0, 0, 0), (255, 100, 255))
    fullMask = pinkMask + whiteMask

    pinkMask = cv2.morphologyEx(pinkMask, cv2.MORPH_CLOSE, (15, 15), iterations=8)
    fullMask = cv2.morphologyEx(fullMask, cv2.MORPH_CLOSE, (15, 15), iterations=8)
    whiteMask = cv2.morphologyEx(whiteMask, cv2.MORPH_CLOSE, (15, 15), iterations=8)

    pinkMask = cv2.medianBlur(pinkMask, 11)
    fullMask = cv2.medianBlur(fullMask, 11)
    whiteMask = cv2.medianBlur(whiteMask, 11)

    kernel = np.ones((5, 5), dtype=np.uint8)
    pinkMask = cv2.erode(pinkMask, kernel)
    fullMask = cv2.erode(fullMask, kernel)
    whiteMask = cv2.erode(whiteMask, kernel)

    return pinkMask, fullMask, whiteMask


def detectCoralBoards(mask):
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    x_b, y_b, x_b2, y_b2 = 0, 0, 0, 0
    try:
        if len(contours) > 0:
            for con in contours:
                if cv2.contourArea(con) < 500:  # Noise Threshold.
                    continue
                x, y, w, h = cv2.boundingRect(con)
                x_b = max(x, x_b)
                y_b = max(y, y_b)
                x_b2 = max(x_b2, x + w)
                y_b2 = max(y_b2, y + h)
        return x_b, y_b, x_b2, y_b2, contours

    except ValueError:
        return None


def getWhite(mask, originalMask, heightDiff, fullOriginalIMG):
    finalImage = originalMask.copy()
    fullOriginalCopy = fullOriginalIMG.copy()
    points = 0

    for ax1, ay1, ax2, ay2 in axis:
        part = mask[ay1:ay2, ax1:ax2]
        part2 = finalImage[ay1:ay2, ax1:ax2]

        histr1 = cv2.calcHist([part], [0], None, [256], [0, 256])
        histr2 = cv2.calcHist([part2], [0], None, [256], [0, 256])

        match1 = cv2.matchShapes(part, part2, 1, 0)
        match2 = cv2.matchShapes(part, part2, 2, 0)

        if match1 > 0.03 and match2 > 0.35:  # .03 0.35
            if np.sum(np.cumsum(histr1)) < np.sum(np.cumsum(histr2)):
                diffWhite[points] += 1
        else:
            diffWhite[points] -= 1
        if diffWhite[points] > 0:
            cv2.rectangle(
                fullOriginalCopy,
                (ax1, ay1 + heightDiff),
                (ax2, ay2 + heightDiff),
                (0, 0, 255),
                2,
            )

        points += 1

    fullOriginalCopy = cv2.cvtColor(fullOriginalCopy, cv2.COLOR_BGR2HSV)
    fullOriginalCopy = cv2.inRange(fullOriginalCopy, (0, 100, 100), (10, 255, 255))

    contours, _ = cv2.findContours(
        fullOriginalCopy, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    if len(contours) > 0:
        for con in contours:
            cv2.drawContours(fullOriginalIMG, contours, -1, (0, 0, 255), 2)
    return fullOriginalIMG


def getPink(mask, originalMask, heightDiff, fullOriginalIMG):
    finalImage = originalMask.copy()
    fullOriginalCopy = fullOriginalIMG.copy()
    points = 0

    for ax1, ay1, ax2, ay2 in axis:
        part = mask[ay1:ay2, ax1:ax2]
        part2 = finalImage[ay1:ay2, ax1:ax2]

        histr1 = cv2.calcHist([part], [0], None, [256], [0, 256])
        histr2 = cv2.calcHist([part2], [0], None, [256], [0, 256])

        match1 = cv2.matchShapes(part, part2, 1, 0)
        match2 = cv2.matchShapes(part, part2, 2, 0)

        if match1 > 0.1 and match2 > 0.35:  # .03 0.35
            if np.sum(np.cumsum(histr1)) < np.sum(np.cumsum(histr2)):
                print(np.sum(np.cumsum(histr1)) < np.sum(np.cumsum(histr2)))
                diffPink[points] += 1
        else:
            diffPink[points] -= 1
        if diffPink[points] > 0:
            cv2.rectangle(
                fullOriginalCopy,
                (ax1, ay1 + heightDiff),
                (ax2, ay2 + heightDiff),
                (255, 0, 0),
                2,
            )

        points += 1

    fullOriginalCopy = cv2.cvtColor(fullOriginalCopy, cv2.COLOR_BGR2HSV)
    fullOriginalCopy = cv2.inRange(fullOriginalCopy, (100, 150, 0), (140, 255, 255))

    contours, _ = cv2.findContours(
        fullOriginalCopy, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    if len(contours) > 0:
        for con in contours:
            cv2.drawContours(fullOriginalIMG, contours, -1, (255, 0, 0), 2)
    return fullOriginalIMG


def getDecay(mask, originalMask, fullResult, heightDiff, fullOriginalIMG):
    finalImage = originalMask.copy()
    fullOriginalCopy = fullResult.copy()
    points = 0

    for ax1, ay1, ax2, ay2 in axis:
        part = mask[ay1:ay2, ax1:ax2]
        part2 = finalImage[ay1:ay2, ax1:ax2]

        histr = cv2.calcHist([part], [0], None, [256], [0, 256])
        histr1 = cv2.calcHist([part2], [0], None, [256], [0, 256])

        match1 = cv2.matchShapes(part, part2, 1, 0)
        match2 = cv2.matchShapes(part, part2, 2, 0)

        if match1 > 0.1 and match2 > 0.35:  # .03 0.35
            if np.sum(np.cumsum(histr)) > np.sum(np.cumsum(histr1)):
                if diffDecay[points] >= 0:
                    cv2.rectangle(
                        fullOriginalCopy,
                        (ax1, ay1 + heightDiff),
                        (ax2, ay2 + heightDiff),
                        (0, 255, 255),
                        2,
                    )
            else:
                if diffDecay[points] >= 0:
                    cv2.rectangle(
                        fullOriginalCopy,
                        (ax1, ay1 + heightDiff),
                        (ax2, ay2 + heightDiff),
                        (0, 255, 0),
                        2,
                    )

            diffDecay[points] += 1
        else:
            diffDecay[points] -= 1

        points += 1

    fullOriginalCopy = cv2.cvtColor(fullOriginalCopy, cv2.COLOR_BGR2HSV)
    contourmask = cv2.inRange(fullOriginalCopy, (25, 52, 72), (102, 255, 255))
    contours, _ = cv2.findContours(
        contourmask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    if len(contours) > 0:
        for con in contours:
            cv2.drawContours(fullOriginalIMG, contours, -1, (0, 255, 255), 2)
    contourmask = cv2.inRange(fullOriginalCopy, (65, 60, 60), (80, 255, 255))
    contours, _ = cv2.findContours(
        contourmask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    if len(contours) > 0:
        for con in contours:
            cv2.drawContours(fullOriginalIMG, contours, -1, (0, 255, 0), 2)
    return fullOriginalIMG


def setSize(mask, originalMask, fullOriginalIMG):
    image = originalMask.copy()
    height = 600

    for y in range(200):
        part = mask[500:600, 0:600]
        part2 = image[height - 100 : height, 0:600]

        match1 = cv2.matchShapes(part, part2, 1, 0)
        match2 = cv2.matchShapes(part, part2, 2, 0)

        if match1 > 0.03 and match2 > 0.35:
            # print(y, match1, match2)
            image = cv2.resize(image, (600, height + 100), cv2.INTER_AREA)
            height += 100
        else:
            break

    mask = cv2.flip(mask, 0)
    addition = np.zeros((y * 100, 600), np.uint8)
    mask = np.vstack((mask, addition))
    mask = cv2.flip(mask, 0)
    heightDiff = height - 600
    fullOriginalIMG = cv2.resize(fullOriginalIMG, (600, height), cv2.INTER_AREA)
    return mask, image, heightDiff, fullOriginalIMG


def getGrowth(mask, originalMask, heightDiff, fullOriginalIMG):
    div = heightDiff // 100
    y, yp, x, xp = 0, 50, 0, 50

    for cnt1 in range(div + 2):
        for cnt2 in range(12):
            br[cnt1][cnt2] = mask[y : y + yp, x : x + xp]
            x += 50
        y += 50
        x = 0

    y, yp, x, xp = 0, 50, 0, 50
    for cnt1 in range(div + 2):
        for cnt2 in range(12):
            br1[cnt1][cnt2] = originalMask[y : y + yp, x : x + xp]
            x += 50
        y += 50
        x = 0

    for cnt1 in range(div + 2):
        for cnt2 in range(12):
            histr = cv2.calcHist([br[cnt1][cnt2]], [0], None, [256], [0, 256])
            histr1 = cv2.calcHist([br1[cnt1][cnt2]], [0], None, [256], [0, 256])
            match1 = cv2.matchShapes(br[cnt1][cnt2], br1[cnt1][cnt2], 1, 0)
            match2 = cv2.matchShapes(br[cnt1][cnt2], br1[cnt1][cnt2], 2, 0)
            match3 = cv2.compareHist(histr, histr1, cv2.HISTCMP_CORREL)
            if match3 < 0.95:  # .03 0.35
                growth[cnt1][cnt2] += 1
            else:
                growth[cnt1][cnt2] -= 1

    for cnt1 in range(div + 2):
        for cnt2 in range(12):
            if (
                growth[cnt1][cnt2] > 0
                and growth[cnt1][cnt2 - 1] > 0
                and growth[cnt1][cnt2 + 1] > 0
            ):
                continue

            elif (
                growth[cnt1][cnt2] > 0
                and growth[cnt1][cnt2 - 1] > 0
                and growth[cnt1][cnt2 + 1] <= 0
            ):
                cv2.line(
                    fullOriginalIMG,
                    ((cnt2 + 1) * 50, cnt1 * 50),
                    ((cnt2 + 1) * 50, (cnt1 + 1) * 50),
                    (0, 255, 0),
                    2,
                )

            elif (
                growth[cnt1][cnt2] > 0
                and growth[cnt1][cnt2 - 1] <= 0
                and growth[cnt1][cnt2 + 1] > 0
            ):
                cv2.line(
                    fullOriginalIMG,
                    (cnt2 * 50, cnt1 * 50),
                    (cnt2 * 50, (cnt1 + 1) * 50),
                    (0, 255, 0),
                    2,
                )

            elif (
                growth[cnt1][cnt2] > 0
                and growth[cnt1][cnt2 - 1] <= 0
                and growth[cnt1][cnt2 + 1] <= 0
            ):
                cv2.line(
                    fullOriginalIMG,
                    ((cnt2 + 1) * 50, cnt1 * 50),
                    ((cnt2 + 1) * 50, (cnt1 + 1) * 50),
                    (0, 255, 0),
                    2,
                )
                cv2.line(
                    fullOriginalIMG,
                    (cnt2 * 50, cnt1 * 50),
                    (cnt2 * 50, (cnt1 + 1) * 50),
                    (0, 255, 0),
                    2,
                )

            if (
                growth[cnt1][cnt2] > 0
                and growth[cnt1 - 1][cnt2] > 0
                and growth[cnt1 + 1][cnt2] > 0
            ):
                continue
            elif (
                growth[cnt1][cnt2] > 0
                and growth[cnt1 - 1][cnt2] > 0
                and growth[cnt1 + 1][cnt2] <= 0
            ):
                cv2.line(
                    fullOriginalIMG,
                    (cnt2 * 50, (cnt1 + 1) * 50),
                    ((cnt2 + 1) * 50, (cnt1 + 1) * 50),
                    (0, 255, 0),
                    2,
                )
            elif (
                growth[cnt1][cnt2] > 0
                and growth[cnt1 - 1][cnt2] <= 0
                and growth[cnt1 + 1][cnt2] > 0
            ):
                cv2.line(
                    fullOriginalIMG,
                    (cnt2 * 50, cnt1 * 50),
                    ((cnt2 + 1) * 50, cnt1 * 50),
                    (0, 255, 0),
                    2,
                )
            elif (
                growth[cnt1][cnt2] > 0
                and growth[cnt1 - 1][cnt2] <= 0
                and growth[cnt1 + 1][cnt2] <= 0
            ):
                cv2.line(
                    fullOriginalIMG,
                    (cnt2 * 50, (cnt1 + 1) * 50),
                    ((cnt2 + 1) * 50, (cnt1 + 1) * 50),
                    (0, 255, 0),
                    2,
                )
                cv2.line(
                    fullOriginalIMG,
                    (cnt2 * 50, cnt1 * 50),
                    ((cnt2 + 1) * 50, cnt1 * 50),
                    (0, 255, 0),
                    2,
                )
    return fullOriginalIMG
