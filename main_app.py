from app.centroids import CentroidTracker
from app.object import Object
from imutils.video import FPS
import numpy as np
import imutils
import dlib, cv2, re, os
from pathlib import Path


def run(video_name):
    n = 0
    framespath = './frames/'
    with open('./pretrainedModel/classes.txt', 'r') as f:
        CLASSES = f.read().split('\n')

    net = cv2.dnn.readNetFromCaffe('./pretrainedModel/MobileNetSSD_deploy.prototxt',
                                   './pretrainedModel/MobileNetSSD_deploy.caffemodel')
    vs = cv2.VideoCapture(f'./data/{video_name}')

    W = None
    H = None

    ct = CentroidTracker()
    trackers = []
    trackableObjects = {}
    totalFrames = 0
    totalRight = 0
    totalLeft = 0
    total_people = []
    lefts = []
    rights = []

    fps = FPS().start()

    while True:
        succ, frame = vs.read()

        if frame is None:
            break

        frame = imutils.resize(frame, width=500)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        if W is None or H is None:
            (H, W) = frame.shape[:2]

        rects = []

        if totalFrames % 10 == 0:
            trackers = []
            blob = cv2.dnn.blobFromImage(frame, 0.008, (W, H), 127.5)
            net.setInput(blob)
            detections = net.forward()
            for i in np.arange(0, detections.shape[2]):
                confidence = detections[0, 0, i, 2]
                if confidence > 0.6:
                    idx = int(detections[0, 0, i, 1])
                    if CLASSES[idx] != "person":
                        continue
                    box = detections[0, 0, i, 3:7] * np.array([W, H, W, H])
                    (startX, startY, endX, endY) = box.astype("int")

                    tracker = dlib.correlation_tracker()
                    rect = dlib.rectangle(startX, startY, endX, endY)
                    tracker.start_track(rgb, rect)
                    cv2.rectangle(frame, (startX, startY), (endX, endY), 4)
                    cv2.namedWindow("Left Right Moving People", cv2.WINDOW_KEEPRATIO)
                    trackers.append(tracker)
        else:
            for tracker in trackers:
                tracker.update(rgb)
                pos = tracker.get_position()

                startX, startY = int(pos.left()), int(pos.top())
                endX, endY = int(pos.right()), int(pos.bottom())

                rects.append((startX, startY, endX, endY))
                cv2.rectangle(frame, (startX, startY), (endX, endY), 6)
                cv2.namedWindow("Left Right Moving People", cv2.WINDOW_KEEPRATIO)

        objects = ct.update(rects)

        for (objectID, centroid) in objects.items():
            to = trackableObjects.get(objectID, None)
            if to is None:
                to = Object(objectID, centroid)
            else:
                x = [c[0] for c in to.centroids]
                direction = centroid[0] - np.mean(x)
                to.centroids.append(centroid)

                if not to.counted:
                    if direction < 0:
                        totalLeft += 1
                        lefts.append(totalLeft)
                        to.counted = True

                    elif direction > 0:
                        totalRight += 1
                        rights.append(totalRight)
                        to.counted = True

                    total_people = []
                    total_people.append(len(rights)+len(lefts))


            trackableObjects[objectID] = to

            cv2.putText(frame, "ID {}".format(objectID), (centroid[0] - 5, centroid[1] - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)

        info = [
            ("Left", totalLeft),
            ("Right", totalRight),
            ("Total people = ", total_people)
        ]
        for (i, (k, v)) in enumerate(info):
            text = "{}: {}".format(k, v)
            cv2.putText(frame, text, (20, H - ((i * 20) + 40)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)


        if succ:
            print(f'{framespath}{n}.png')
            cv2.imwrite(f'{framespath}{n}.png', frame)
            n += 1

        cv2.imshow("Left Right Moving People", frame)
        cv2.waitKey(1) & 0xFF

        totalFrames += 1
        fps.update()

    fps.stop()
    per_min = fps.fps()
    print(f"FPS: {round(per_min)}")

    cv2.destroyAllWindows()
    return (W, H), per_min



def sorting_data(data):
    text_to_convert = lambda text: int(text) if text.isdigit() else text
    int_key = lambda key: [text_to_convert(c) for c in re.split('([0-9]+)', key)]
    return sorted(data, key=int_key)


if __name__ == "__main__":

    video_name = 'ex_1.mp4'

    p = Path('./frames')

    frame_size, fps = run(video_name)
    frames = os.listdir(p)

    out = cv2.VideoWriter(f'./output_data/output_{video_name}.avi',
                          cv2.VideoWriter_fourcc('M','J','P','G'), fps, frame_size)

    for i in sorting_data(frames):
        out.write(cv2.imread(f'./frames/{i}'))

    for f in os.listdir(p):
        os.remove(os.path.join(p, f))
    out.release()