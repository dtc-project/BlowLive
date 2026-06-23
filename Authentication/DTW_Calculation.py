import os
import csv
import re
import math
import numpy as np

# ----- Data structure -----
class GFCCData:
    def __init__(self, name, data):
        self.name = name
        self.data = data  # list of list of int64


# ----- Local Manhattan distance -----
def localDistanceManhattan(x, y):
    if len(x) != len(y):
        raise ValueError("arrays have different dimensions!")
    dist = 0
    for a, b in zip(x, y):
        delta = a - b
        if delta < 0:
            delta = -delta
        dist += delta
    return dist


# ----- DTW -----
def computeDTW(x, y, unused, distfunc):
    n1 = len(x)
    n2 = len(y)

    if len(x[0]) != len(y[0]):
        raise ValueError("time series elements have different dimensions!")

    # Very large int (MaxInt64 equivalent)
    INF = (1 << 62)

    # dtw[j][i]
    dtw = [[INF] * n1 for _ in range(n2)]
    dtw[0][0] = 0

    for j in range(n2):
        for i in range(n1):
            if i > 0 or j > 0:
                cost = distfunc(x[i], y[j])
                candidates = []
                if i > 0:
                    candidates.append(dtw[j][i - 1])
                if j > 0:
                    candidates.append(dtw[j - 1][i])
                if i > 0 and j > 0:
                    candidates.append(dtw[j - 1][i - 1])

                dtw[j][i] = cost + min(candidates)

    return dtw[-1][-1]


# ----- Name extraction (same regex) -----
def extractName(filename):
    match = re.match(r'^[^_]+', os.path.basename(filename))
    if match:
        return match.group(0).lower()
    return ""


# ----- CSV reader -----
def readCSV(filename):
    matrix = []
    with open(filename, "r") as f:
        reader = csv.reader(f)
        for record in reader:
            row = []
            for value in record:
                float_val = float(value)
                row.append(int(float_val * 1e9))   # same scale as Go
            matrix.append(row)
    return matrix


# ----- Write DTW matrix to CSV -----
def writeDTWMatrix(matrix, filename):
    with open(filename, "w", newline="") as f:
        writer = csv.writer(f)
        for row in matrix:
            writer.writerow([str(v) for v in row])


# ----- Split (A = sit, B = stand) -----
def runSplitDTW(datasetDir):
    csvFilesA = []
    csvFilesB = []

    # Collect files
    for root, _, files in os.walk(datasetDir):
        for file in files:
            if re.search(r"_gfcc260-26.csv$", file):
                fullpath = os.path.join(root, file)
                if "_A_" in file:
                    csvFilesA.append(fullpath)
                elif "_B_" in file:
                    csvFilesB.append(fullpath)

    print("Found A files:", len(csvFilesA))
    print("Found B files:", len(csvFilesB))

    # Load sit (A)
    gfccA = []
    for file in csvFilesA:
        name = extractName(file)
        data = readCSV(file)
        gfccA.append(GFCCData(name, data))

    # Load stand (B)
    gfccB = []
    for file in csvFilesB:
        name = extractName(file)
        data = readCSV(file)
        gfccB.append(GFCCData(name, data))

    
    gfccA.sort(key=lambda x: x.name)
    gfccB.sort(key=lambda x: x.name)

    # Allocate matrices
    dtwMatrixA = [[0] * len(gfccA) for _ in range(len(gfccA))]
    dtwMatrixB = [[0] * len(gfccB) for _ in range(len(gfccB))]

    # Compute DTW for A
    for i in range(len(gfccA)):
        for j in range(len(gfccA)):
            print(f"Comparing A {i} with {j}")
            dtwMatrixA[i][j] = computeDTW(gfccA[i].data, gfccA[j].data, 0, localDistanceManhattan)

    # Compute DTW for B
    for i in range(len(gfccB)):
        for j in range(len(gfccB)):
            print(f"Comparing B {i} with {j}")
            dtwMatrixB[i][j] = computeDTW(gfccB[i].data, gfccB[j].data, 0, localDistanceManhattan)

    writeDTWMatrix(dtwMatrixA, "dtw_gfcc260-26_sit.csv")
    writeDTWMatrix(dtwMatrixB, "dtw_gfcc260-26_stand.csv")

    print("✓ DTW matrices saved: dtw_gfcc260-26_sit.csv and dtw_gfcc260-26_stand.csv")


# ----- Full version -----
def runFullDTW(datasetDir):
    csvFiles = []

    for root, _, files in os.walk(datasetDir):
        for file in files:
            # print(f"Found file: '{file}'")   # debug print
            if "_gfcc.csv" in file:    # simpler match
                csvFiles.append(os.path.join(root, file))

    print("Total matched files:", len(csvFiles))


    # Load all
    gfccList = []
    for file in csvFiles:
        name = extractName(file)
        data = readCSV(file)
        gfccList.append(GFCCData(name, data))

    gfccList.sort(key=lambda x: x.name)

    # Allocate matrix
    n = len(gfccList)
    dtwMatrix = [[0] * n for _ in range(n)]

    

    # Compute DTW
    for i in range(10):
        for j in range(10):
            print(f"Comparing {i} with {j}")
            dtwMatrix[i][j] = computeDTW(
                gfccList[i].data,
                gfccList[j].data,
                0,
                localDistanceManhattan
            )

    # writeDTWMatrix(dtwMatrix, "dtw_gfcc260-26_full.csv")
    print("✓ DTW matrix saved: dtw_gfcc260-26_full.csv")

# ----- Cosine similarity -----
def cosine_similarity(a, b):
    a = np.array(a, dtype=float)
    b = np.array(b, dtype=float)
    dot_product = np.dot(a, b)
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot_product / (norm_a * norm_b)


# ----- Read _img.csv embeddings -----
class ImageEmbedding:
    def __init__(self, name, vector):
        self.name = name
        self.vector = vector  # list or np.array


def read_image_embeddings(datasetDir):
    csvFiles = []
    for root, _, files in os.walk(datasetDir):
        for file in files:
            if "_img.csv" in file:
                csvFiles.append(os.path.join(root, file))

    print("Found _img.csv files:", len(csvFiles))

    embeddings = []
    for file in csvFiles:
        name = extractName(file)
        # robust read: assume one-column numeric CSV
        with open(file, "r") as f:
            vector = [float(line.strip()) for line in f if line.strip()]
        embeddings.append(ImageEmbedding(name, vector))

    # sort by name
    embeddings.sort(key=lambda x: x.name)
    return embeddings


# ----- Compute cosine similarity matrix -----
def compute_cosine_matrix(embeddings):
    n = len(embeddings)
    names_list = [e.name for e in embeddings]
    sim_matrix = np.zeros((n, n), dtype=float)

    for i in range(n):
        for j in range(n):
            if i != j:
                sim_matrix[i, j] = cosine_similarity(embeddings[i].vector, embeddings[j].vector)

    return sim_matrix, names_list


# ----- Save cosine similarity -----
def writeCosineMatrix(matrix, names_list, filename):
    with open(filename, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([""] + names_list)
        for i, row in enumerate(matrix):
            writer.writerow([names_list[i]] + list(row))


# ----- Main for image embeddings -----
def runCosineSimilarity(datasetDir):
    embeddings = read_image_embeddings(datasetDir)
    sim_matrix, names_list = compute_cosine_matrix(embeddings)
    # writeCosineMatrix(sim_matrix, names_list, "cosine_similarity_img.csv")
    print("✓ Cosine similarity matrix saved: cosine_similarity_img.csv")
    print(sim_matrix[0][1])




# ----- Main -----
if __name__ == "__main__":
    datasetDir = "BlowLive_Dataset"
    # runSplitDTW(datasetDir)
    # runFullDTW(datasetDir)
    runCosineSimilarity(datasetDir)

