import numpy as np

def Dice3d(a, b):
    if len(a.shape) != 3 or len(b.shape) != 3:
        raise Exception(f"Expecting 3D inputs, got {a.shape} and {b.shape}")
    if a.shape != b.shape:
        raise Exception(f"Shapes do not match: {a.shape} vs {b.shape}")

    a_bin = (a > 0).astype(np.uint8)
    b_bin = (b > 0).astype(np.uint8)

    intersection = np.sum(a_bin * b_bin)
    volume_sum = np.sum(a_bin) + np.sum(b_bin)

    if volume_sum == 0:
        return 1.0  # Both are empty, considered perfect match
    return 2.0 * intersection / volume_sum

def Jaccard3d(a, b):
    if len(a.shape) != 3 or len(b.shape) != 3:
        raise Exception(f"Expecting 3D inputs, got {a.shape} and {b.shape}")
    if a.shape != b.shape:
        raise Exception(f"Shapes do not match: {a.shape} vs {b.shape}")

    a_bin = (a > 0).astype(np.uint8)
    b_bin = (b > 0).astype(np.uint8)

    intersection = np.sum(a_bin & b_bin)
    union = np.sum(a_bin | b_bin)

    if union == 0:
        return 1.0
    return intersection / union

def Sensitivity(a, b):
    a_bin = (a > 0).astype(np.uint8)
    b_bin = (b > 0).astype(np.uint8)

    TP = np.sum((a_bin == 1) & (b_bin == 1))
    FN = np.sum((a_bin == 0) & (b_bin == 1))

    if TP + FN == 0:
        return 1.0
    return TP / (TP + FN)

def Specificity(a, b):
    a_bin = (a > 0).astype(np.uint8)
    b_bin = (b > 0).astype(np.uint8)

    TN = np.sum((a_bin == 0) & (b_bin == 0))
    FP = np.sum((a_bin == 1) & (b_bin == 0))

    if TN + FP == 0:
        return 1.0
    return TN / (TN + FP)