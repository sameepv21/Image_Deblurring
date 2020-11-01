import numpy as np
import math
from PIL import Image, ImageOps
import matplotlib.pyplot as plt

plt.ion()

def getSingularValues(A):
    result = np.zeros(len(A))
    for i in range(0, len(A)):
        result[i] = math.sqrt(A[i])
    return result

def getEigenValues(A):
    #Identity Matrix of nxn
    I = np.identity(n)
    eval = np.zeros(n)
    np.linalg.eig
    return eval

def getTranspose(A):
    result = np.zeros((m,n))
    for i in range(0, n):
        for j in range(0, m):
            result[j][i] = A[i][j]
    return result

#Read and show the image
img = Image.open('C:\\Users\\16692\\Documents\\ExtraProjects\\Image Blurring and Deblurring\\sample.jpg')
plt.title("Original Image")
plt.imshow(img)
plt.show()
#img.show()

#Convert into gray scale
img2 = ImageOps.grayscale(img)
#img2.show()

#Convert image into matrix
b = np.array(img2)

#Fetch the dimensions of image
print(b.shape)
n,m = b.shape

#As we are performing SVD, we need an identity matrix of nxn
I = np.identity(n)

#Find the eigenvalues of b(Transpose)b
#For this, we require to find b(Transpose)
bT = getTranspose(b)
#print(bT)

#Find bTb
S = np.dot(b, bT)

#Find EigenValues of S
eValues, eVectors = np.linalg.eig(S)

#Find Singular Values
singValues = getSingularValues(eValues)

#Input the number of singular values that you want to take
k = int(input('Enter the number of singular values you want to take: '))

#Compute SVD
U, sigma, VT = np.linalg.svd(img2)
sigma = np.diag(sigma)

resultantMatrixApproximated = U[:,:k] @ sigma[0:k,:k] @ VT[:k,:]
print('', resultantMatrixApproximated.shape)
plt.imshow(resultantMatrixApproximated)
plt.show()