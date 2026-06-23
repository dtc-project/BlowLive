import numpy as np

def determine_threshold(matrix, k, p):
    """
    matrix: 2D numpy array
    k: the k-th nearest neighbor
    p: the p-th largest threshold candidate
    """
    threshold_candidates = []
    for i in range(matrix.shape[0]):
        distances = matrix[i, :]
        sorted_distances = np.sort(distances)
        threshold_candidates.append(sorted_distances[k])
    threshold_candidates = np.array(threshold_candidates)
    final_threshold = np.sort(threshold_candidates)[::-1][p]
    return final_threshold


def compare_data(mat, new, base, thres, k):
    """
    mat: full distance matrix
    new: list/array of row indices
    base: list/array of column indices
    thres: threshold for decision
    k: the k-th neighbor
    """
    tDists = []
    nCorrect = 0
    
    for i in new:
        dists_for_series = mat[i, base]
        dists_for_series = np.sort(dists_for_series)
        tDist = dists_for_series[k]
        tDists.append(tDist)
        if tDist <= thres:
            nCorrect += 1
    
    return nCorrect / len(new)


def extract_submatrix(matrix, rownames, colnames, session_name):
    """
    matrix: 2D numpy array
    rownames: list of row names
    colnames: list of col names
    """
    row_indices = [i for i, r in enumerate(rownames) if r == session_name]
    col_indices = [j for j, c in enumerate(colnames) if c == session_name]
    
    if row_indices and col_indices:
        return matrix[np.ix_(row_indices, col_indices)]
    else:
        return None  # session name not found


def matrix_compare(matrix, session_names, k, p):
    """
    matrix: 2D numpy array (distance matrix)
    session_names: list of session names (length matches rows/cols)
    """
    unique_session_names = list(set(session_names))
    user_matrices = {}
    
    for session in unique_session_names:
        subm = extract_submatrix(matrix, session_names, session_names, session)
        if subm is not None:
            user_matrices[session] = subm
    
    user_thresholds = {}
    for name, mat in user_matrices.items():
        user_thresholds[name] = determine_threshold(mat, k, p)
    
    TP = FP = TN = FN = TotP = TotN = 0
    
    for i, name_i in enumerate(unique_session_names):
        id1 = [idx for idx, n in enumerate(session_names) if n == name_i]
        thres = user_thresholds[name_i]
        
        for j, name_j in enumerate(unique_session_names):
            id2 = [idx for idx, n in enumerate(session_names) if n == name_j]
            MF = compare_data(matrix, id1, id2, thres, k)
            
            if i == j:
                TP += MF
                FN += 1 - MF
                TotP += 1
            else:
                FP += MF
                TN += 1 - MF
                TotN += 1
    
    print("Precision:", TP / (TP + FP) if (TP + FP) > 0 else 0)
    print("Recall:", TP / (TP + FN) if (TP + FN) > 0 else 0)
    print("False Positive:", FP / (TN + FP) if (TN + FP) > 0 else 0)
    print("False Negative:", FN / (TP + FN) if (TP + FN) > 0 else 0)
    print("Accuracy:", (TP + TN) / (TotP + TotN) if (TotP + TotN) > 0 else 0)
    
    print(TP, TN, FP, FN)



import pandas as pd

#Change the matrix csv file according to which you want to run
matrix_file = "dtw_gfcc260-26_matrix_int64.csv"

# Parameters
num_users = 50     # total users
sessions_per_user = 10  # sessions per user

# Generate session names: user1,user1,...,user2,user2,... etc
session_names = []
for u in range(1, num_users + 1):
    session_names.extend([f"user{u}"] * sessions_per_user)

# Read CSV with pandas
df = pd.read_csv(matrix_file, header=None) 
distance_matrix = df.values                

print("Distance matrix shape:", distance_matrix.shape)
print("Session names:", session_names)

#change the K and Q parameters accordingly to your need.
k = 3   
q = 0   
matrix_compare(distance_matrix, session_names, k, q)