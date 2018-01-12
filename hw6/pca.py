import numpy as np
import matplotlib.pyplot as plt

from PIL import Image
from skimage import io, transform

import sys
import os


def pca(a):
    m = a - np.mean(a, axis=0)
    U, s, v = np.linalg.svd(m, full_matrices=False)
    return v


def reconstruct(a, eigen_v):
    m = np.mean(a, axis=0)
    projected_a = np.dot(a - m, eigen_v.T)
    reconstructed_a = m + np.dot(projected_a, eigen_v)
    return reconstructed_a


faces_dir = sys.argv[1]

faces = []
img_file_name = os.listdir(faces_dir)
for name in img_file_name:
    face_img = io.imread(os.path.join(faces_dir, name))
    face = np.array(face_img).flatten()
    faces.append(face)
faces = np.array(faces)

# average_face = np.mean(faces, axis=0)
# io.imsave("average_face.jpg", average_face.astype(np.uint8).reshape(600,600,3))


eigenfaces = pca(faces)

# for i in range(4):
    # M = eigenfaces[i]
    # M -= np.min(M)
    # M /= np.max(M)
    # M = (M * 255).astype(np.uint8)
    # io.imsave("eigen_face{:d}.jpg".format(i+1), M.astype(np.uint8).reshape(600,600,3))

# faces = []
# i = 0
# for name in img_file_name:
    # if i < 3: 
        # face_img = io.imread(os.path.join(faces_dir, name))
        # face = np.array(face_img).flatten()
        # faces.append(face)
# faces = np.array(faces)

# num_pcs = 4
# reconstructed_faces = reconstruct(faces, eigenfaces[:num_pcs])

# for i in range(num_pcs):
    # M = reconstructed_faces[i]
    # M -= np.min(M)
    # M /= np.max(M)
    # M = (M * 255).astype(np.uint8)
    # io.imsave("reconstruction{:d}.jpg".format(i+1), M.astype(np.uint8).reshape(600,600,3))

face_img = io.imread(os.path.join(faces_dir, sys.argv[2]))
face = np.array(face_img).flatten()
num_pcs = 4
reconstructed_faces = reconstruct(face, eigenfaces[:num_pcs])

M = reconstructed_faces
M -= np.min(M)
M /= np.max(M)
M = (M * 255).astype(np.uint8)
io.imsave("reconstruction.jpg", M.astype(np.uint8).reshape(600,600,3))

# plt.figure()
# for i in range(100):
#     plt.subplot(10, 10, i + 1)
#     plt.axis("off")
#     plt.imshow(faces[i].reshape(64, 64), cmap="gray")
# plt.subplots_adjust(wspace=0.2, hspace=0.2)
# plt.suptitle("original faces")
# plt.savefig("original_faces.png")
# plt.close()

# plt.figure()
# for i in range(100):
#     plt.subplot(10, 10, i + 1)
#     plt.axis("off")
#     plt.imshow(reconstructed_faces[i].reshape(64, 64), cmap="gray")
# plt.subplots_adjust(wspace=0.2, hspace=0.2)
# plt.suptitle("reconstructed faces")
# plt.savefig("reconstructed_faces_{:d}.png".format(num_pcs))
# plt.close()

# faces = []
# for i in range(10):
#     for j in range(10):
#         img_file_name = "{}{:02d}.bmp".format(chr(ord("A") + i), j)
#         face_img = Image.open(os.path.join(faces_dir, img_file_name))
#         face = np.array(face_img).flatten()
#         faces.append(face)
# faces = np.array(faces)

# for num_pcs in range(1, 101):
#     reconstructed_faces = reconstruct(faces, eigenfaces[:num_pcs])

#     rmse = (np.mean(((reconstructed_faces - faces) / 255)**2))**0.5
#     print("# of pcs: {:d} RMSE: {:4.2f}%".format(num_pcs, rmse * 100))
#     if rmse < 0.01:
#         break

