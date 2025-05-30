import numpy as np
from skimage.io import imread
import matplotlib.pyplot as plt
import cv2


np.set_printoptions(suppress=True)
def eight_point_alg_lst_sq(points1, points2):
    """
    Computes the fundamental matrix F from pairs of 2D image points using the 8-point algorithm through least squares
    F such that (point2).T @ F @ point1 = 0 For every point
    :param points1: 2d image points in the first image (N,2)
    :param points2: 2d image points in the second image (N,2)
    :return: F: the fundamental matrix that relates camera2 to camera 1.
    """
    N, M = len(points1), len(points2)
    W = np.zeros((N, 9))

    # F such that (point2).T @ F @ point1 = 0 For every point
    # W is constructed as rows of (uu', u'v, u', uv', vv', v', u, v, 1)

    assert N == M, "the lengths of points 1 and points2 do not match"
    for i, (point1, point2) in enumerate(zip(points1, points2)):
        #print((point1, point2))
        u, v, _ = point2
        up, vp, _ = point1
        W[i] = [
            u * up,
            u * vp,
            u,
            v * up,
            v * vp,
            v,
            up,
            vp,
            1  # constant
        ]

    # singular value decomposition
    U, S, Vt = np.linalg.svd(W)
    # take last row of Vt to enforce least squares
    f = Vt[-1]
    F = f.reshape(3,3)

    # Reconstruct the F matrix enforcing rank2 -> last singular value = 0 Sf[-1] = 0
    Uf, Sf, Vtf = np.linalg.svd(F)
    Sf[-1] = 0

    F_rank2 = Uf @ np.diag(Sf) @ Vtf

    #normalize
    return F_rank2/F_rank2[-1,-1]



def get_T_matrix_normalized_8point(points):
    """
    Helper function for the normalized 8-point algorithm.
    :param points:
    :return: T, scale. The translation vector and scale factor that ensure that the image points will lie around the
    origin with tight spread
    """
    centroid = np.average(points, axis=0)
    denominator = ((centroid-points)**2).sum()
    N = np.shape(centroid)[0]
    T = np.zeros((3, 3))
    T[:2, :2] = np.eye(2)
    T[:, -1] = -centroid
    T[-1, -1] = 1
    scale = np.sqrt(2*N/denominator)
    return T, scale

def normalized_eight_point_alg(points1, points2):
    """
    Improved version of the 8-point algorithm with normalization, which pre-conditions the input point distribution and
    increases computation accuracy.

    Computes the fundamental matrix F from pairs of 2D image points using the 8-point algorithm through least squares
    F such that (point2).T @ F @ point1 = 0 For every point

    :param points1: 2d image points in the first image (N,2)
    :param points2: 2d image points in the second image (N,2)
    :return: F: the fundamental matrix that relates camera2 to camera 1.
    """

    T1, scale1 = get_T_matrix_normalized_8point(points1)
    T2, scale2 = get_T_matrix_normalized_8point(points2)

    q1 = points1 + T1[:, -1]
    q2 = points2 + T2[:, -1]

    F_q = eight_point_alg_lst_sq(q1, q2)

    F = T2.T @ F_q @ T1

    return F/F[-1,-1]



def plot_epipolar_lines_one_view(points1, points2, im, F):
    """
    Plots the epipolar lines in one image.
    :param points1:
    :param points2:
    :param im:
    :param F:
    :return:
    """
    H = im.shape[0]
    W = im.shape[1]
    lines = F.T.dot(points2.T)
    plt.imshow(im, cmap='gray')
    for line in lines.T:
        a, b, c = line
        xs = np.linspace(1, W-1)
        ys = -(a * xs + c) / b
        plt.plot(xs, ys, 'r')
    for i in range(points1.shape[0]):
        x, y, _ = points1[i]
        plt.plot(x, y, '*b')
    plt.axis([0, W, H, 0])
def plot_epipolar_lines_two_views(points1, points2, im1, im2, F):
    """
    Takes a set of matching points in im1, im2, and draws the corresponding epipolar lines on the other image using the
    fundamental matrix F that relates both camera views.
    points1 -> epipolar lines in im2
    points2 -> epipolar lines in im1
    :param points1: 2d image points in the first image (N,2)
    :param points2: 2d image points in the second image (N,2)
    :param im1: first view of the scene
    :param im2: second view of the scene
    :param F: Fundamental matrix relating both camera views, such that (point2).T @ F @ point1 = 0 For every point
    :return: None, plots the figure
    """



    fig = plt.figure()
    plt.subplot(121)
    plot_epipolar_lines_one_view(points1, points2, im1, F)
    plt.axis('off')
    plt.subplot(122)
    plot_epipolar_lines_one_view(points2, points1, im2, F.T)
    plt.axis('off')


def get_distance_points_to_epipolar_lines(points1, points2, F):
    """
    Computes the distance of the original points to the estimated epipolar lines freom the other image
    :param points1: 2d image points in the first image (N,2)
    :param points2: 2d image points in the second image (N,2)
    :param F: Fundamental matrix relating both camera views, such that (point2).T @ F @ point1 = 0 For every point
    :return: the average of the distances from point to corresponding epipolar line in pixels
    """
    lines1 = (F.T).dot(points2.T)

    dot_product = np.einsum('ij,ji->j', lines1, points1)
    distances = np.abs(dot_product)/np.linalg.norm(lines1[:2, :], axis=0)

    return np.average(distances)


if __name__ == '__main__':
    data_path = "data/house"
    
    im1 = imread(data_path + '/img1.jpg')
    im2 = imread(data_path + '/img2.jpg')
    points1 = np.loadtxt(data_path + '/keypoints_1.txt')
    points2 = np.loadtxt(data_path + '/keypoints_2.txt')

    points1 = np.fliplr(points1)
    points2 = np.fliplr(points2)
    points1 = np.hstack((points1, np.ones(points1.shape[0])[:, np.newaxis]))
    points2 = np.hstack((points2, np.ones(points2.shape[0])[:, np.newaxis]))

    assert (points1.shape == points2.shape), "the points in img1 and img2 are different!"

    # Comparing simple least squares to normalized least squares
    F_least_sq = eight_point_alg_lst_sq(points1, points2)
    print("Using simple least squares:")
    print("Fundamental Matrix:\n", F_least_sq)
    print("Distances from points to epipolar lines: ")
    print("Image 1:", get_distance_points_to_epipolar_lines(points1, points2, F_least_sq))
    print("Image 2:", get_distance_points_to_epipolar_lines(points2, points1, F_least_sq.T))

    # pFp_least_sq should approach 0
    pFp_least_sq = np.sum(points2 * (F_least_sq @ points1.T).T, axis=1)
    print("p'.T F_least_sq p =", np.abs(pFp_least_sq).max())

    # Now, normalized algorithm
    F_normalized = normalized_eight_point_alg(points1, points2)

    pFp_normalized_algo = np.sum(points2 * (F_normalized @ points1.T).T, axis=1)
    print("p'.T F p =", np.abs(pFp_normalized_algo).max())
    print("Using least squares with normalization:")
    print("Fundamental Matrix:\n", F_normalized)
    print("Distances from points to epipolar lines: ")
    print("Image 1:", get_distance_points_to_epipolar_lines(points1, points2, F_normalized))
    print("Image 2:", get_distance_points_to_epipolar_lines(points2, points1, F_normalized.T))

    # Plots
    plot_epipolar_lines_two_views(points1, points2, im1, im2, F_least_sq)
    plot_epipolar_lines_two_views(points1, points2, im1, im2, F_normalized)

    plt.show()
