from scipy.spatial import distance as dist
from collections import OrderedDict
import numpy as np


class CentroidTracker:
    def __init__(self, maxDisappeared=50, maxDistance=50):
        self.nextObjectID = 0
        self.objects = OrderedDict()
        self.disappeared = OrderedDict()
        self.maxDisappeared = maxDisappeared
        self.maxDistance = maxDistance

    def register(self, centroid):
        self.objects[self.nextObjectID] = centroid
        self.disappeared[self.nextObjectID] = 0
        self.nextObjectID += 1

    def deregister(self, id):
        del self.objects[id]
        del self.disappeared[id]

    def update(self, rects):
        if len(rects) == 0:
            for id in list(self.disappeared.keys()):
                self.disappeared[id] += 1
                if self.disappeared[id] > self.maxDisappeared:
                    self.deregister(id)
            return self.objects

        inputCentroids = np.zeros((len(rects), 2), dtype="int")

        for (i, (startX, startY, endX, endY)) in enumerate(rects):
            inputCentroids[i] = (int((startX + endX) / 2.0), int((startY + endY) / 2.0))

        if len(self.objects) == 0:
            for i in range(0, len(inputCentroids)):
                self.register(inputCentroids[i])
        else:
            objectIDs = list(self.objects.keys())
            objectCentroids = list(self.objects.values())

            D = dist.cdist(np.array(objectCentroids), inputCentroids)

            rows = D.min(axis=1).argsort()
            cols = D.argmin(axis=1)[rows]

            usedRows, usedCols = set(), set()

            for (row, col) in zip(rows, cols):
                if row in usedRows or col in usedCols:
                    continue
                if D[row, col] > self.maxDistance:
                    continue

                id = objectIDs[row]
                self.objects[id] = inputCentroids[col]
                self.disappeared[id] = 0

                usedRows.add(row)
                usedCols.add(col)

            unusedRows = set(range(0, D.shape[0])).difference(usedRows)
            unusedCols = set(range(0, D.shape[1])).difference(usedCols)

            if D.shape[0] >= D.shape[1]:
                for row in unusedRows:
                    id = objectIDs[row]
                    self.disappeared[id] += 1
                    if self.disappeared[id] > self.maxDisappeared:
                        self.deregister(id)
            else:
                for col in unusedCols:
                    self.register(inputCentroids[col])

        return self.objects