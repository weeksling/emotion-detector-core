###################################################################
# Matthew Militante, 100457072
#
# Performs PCA Analysis on a folder containing raw images. When 
# reading images from an input folder, it assumes that the images
# are 16384x1 in size.
#
# Usage:
#   analyze.do_pca(<folder_name>)
# 
# Options: 
#   max_images: Use as a second arg in do_pca(). By default, PCA 
#               will only be performed on 20 images in the folder
#
# The results of PCA are stored in:
#   - SUBSPACE: data/results/subspace.json
#   - RESULTS FROM PROJECTION: data/results/<class_name>_projection_results.json
# 
###################################################################

import os
import numpy as np
import time
import json

def read_images(folder, max_images):
  imlist = []
  img_count = 0
  for file in os.listdir(folder):
    full_path = os.path.abspath(folder+'/'+file)
    if img_count < max_images:
      fd = open(full_path, 'rb')
      f = np.fromfile(fd, dtype=np.uint8)
      imlist.append(f)
      fd.close()
      img_count += 1

  # DO some pre-processing and construct the data matrix required for PCA
  m,n = 128, 128 
  imnbr = len(imlist) 

  img_data = np.zeros((imnbr, (m * n)))
  counter = 0
  for image in imlist:
    temp = np.array(image)
    temp = np.asanyarray(image)

    if len(temp) > 16384:
      temp = resize(temp, m, n)

    img_data[counter] = temp
    counter += 1

  img_data = np.asarray(img_data)
  img_matrix = np.array(img_data)
  img_matrix = np.asanyarray(img_matrix)

  return img_matrix


def pca(X):
  n, d = X.shape

  # 1. Calulate the mean
  mu = np.mean(X, 0)
  X = X - mu

  # 2. Calculate the Eigenvalues and Eigenvectors
  # take the eigenvalue decomposition S = XT X of size NxN instead because 16384x16384 is fucking huge!!!!
  covariance_matrix = np.dot(X,X.T)

  eigenvalues, eigenvectors = np.linalg.eigh(covariance_matrix) 
  # TRICK: Get the original eigenvectors of S = XXT with a left multiplication of the data matrix
  eigenvectors = np.dot(X.T ,eigenvectors)
  for i in xrange (n):
    eigenvectors[:,i]=eigenvectors[:,i]/np.linalg.norm(eigenvectors[:,i])
  
  idx = np.argsort(-eigenvalues)  
  eig_vals = eigenvalues[idx]
  eig_vecs = eigenvectors[:,idx]
  
  return eig_vals, eig_vecs, mu
  

def subspace(e_vals, e_vecs):
  # BUILD THE MATRIX THAT WILL TRANSFORM DATA INTO THE SUBSPACE 
  eig_pairs = [(np.abs(e_vals[i]), e_vecs[:,i]) for i in range(len(e_vals))]
  # create the projection matrix 
  # we reduce the 20-dimensional feature space to a 2 dimensional feature subspace
  matrix_w = np.hstack((eig_pairs[0][1].reshape(16384,1), eig_pairs[1][1].reshape(16384,1)))

  return matrix_w


def do_pca(img_folder, max_images=20):

  X = read_images(img_folder, max_images)

  print "Performing PCA..."
  start = time.time()
  # Perform PCA Analysis
  e_vals, e_vecs, mu = pca(X)
  endtime = time.time()
  print 'PCA completed in', (endtime - start), 'sec'

  # Get the subspace
  S = subspace(e_vals, e_vecs)
  print '\nSubspace:', S

  # Project faces on to the sub-space
  Y = X.dot(S)
  print "\nResults from Projecting the faces onto the calculated subspace:\n", Y

  class_name = os.path.basename(os.path.normpath(img_folder))

  # Store results in JSON File
  # Will eventually be on DB
  with open('../data/results/subspace.json', 'w') as fp:
    json.dump(S.tolist(), fp)

  with open('../data/results/'+class_name+'_projection_results.json', 'w') as fp:
    json.dump(Y.tolist(), fp)



  




