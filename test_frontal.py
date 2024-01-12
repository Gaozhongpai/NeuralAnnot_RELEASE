import torch 

def find_rigid_transform(A, B):
    # Ensure tensors are of the same size
    assert A.size() == B.size()

    # Subtract centroids
    centroid_A = torch.mean(A, dim=0)
    centroid_B = torch.mean(B, dim=0)
    A_centered = A - centroid_A
    B_centered = B - centroid_B

    # Compute the covariance matrix
    H = torch.matmul(A_centered.T, B_centered)

    # SVD
    U, S, Vt = torch.linalg.svd(H)

    # Compute rotation
    R = torch.matmul(Vt.T, U.T)
    if torch.det(R) < 0:
        Vt[-1, :] *= -1
        R = torch.matmul(Vt.T, U.T)

    # Compute direct translation
    t = centroid_B - centroid_A
    return R, t


def transform_points(A, R, t):
    # Apply translation
    A_translated = A + t

    # Rotate the translated points
    A_transformed = torch.matmul(A_translated, R.T)

    return A_transformed

def inverse_transform_points(A_transformed, R, t):
    # Inverse rotation
    A_rotated_back = torch.matmul(A_transformed, R)

    # Inverse translation
    A_original = A_rotated_back - t

    return A_original



if __name__ == "__main__":
    # Example data
    A = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]])  # Set A
    B = torch.tensor([[4.0, 5.0, 6.0], [7.0, 8.0, 9.0], [10.0, 11.0, 12.0]])  # Set B

    # Find rotation and translation
    R, t = find_rigid_transform(A, B)

    # Apply transformation
    A_transformed = transform_points(A, R, t)

    # Output the transformed A
    print("Transformed A:\n", A_transformed)
    
    # Apply the inverse transformation
    A_back = inverse_transform_points(A_transformed, R, t)

    # Output the original A
    print("Original A:\n", A_back)
