import numpy as np
import math
from PIL import Image, ImageOps
import matplotlib.pyplot as plt
from sympy import Matrix
import scipy as sp

# mat1 = np.matrix([[1, 1], [2, 2], [3, 3]])
# mat2 = [[1, 2]]

# answer = multiplyTwoMatricies(mat1, mat2)
# print(answer)

# mat3 = np.zeros((3, 2))
# mat4 = np.zeros((3, 1))
# answer = multiplyTwoMatricies(mat3, mat4)
# print(answer)
# print(answer.shape)

def multiplyTwoMatricies(A, B):
    # Checked and worked correctly
    dim1 = np.shape(A)
    size1 = np.size(A)
    m1 = dim1[0]
    n1 = size1//m1

    dim2 = np.shape(B)
    size2 = np.size(B)
    m2 = dim2[0]
    n2 = size2//m2

    result = np.zeros((m1, n2))
    for i in range(0, m1):
        for j in range(0, n2):
            #result[i][j] = 0
            for k in range(0, m2):
                result[i][j] += (A[i][k] * B[k][j])
    return result


def getNorm(A):
    # Checked and Works Fine
    norm = 0
    for i in range(0, len(A)):
        norm += A[i]*A[i]
    return math.sqrt(norm)


def getNormalisedVector(vector):
    # Checked and Works fine
    length = getNorm(vector)
    for i in range(0, len(vector)):
        vector[i] = vector[i]/length
    return vector

def getTranspose(A):
    # Checked and Works fine
    dim3 = np.shape(A)
    size3 = np.size(A)
    temp1 = dim3[0]
    temp2 = size3//dim3[0]

    result = np.zeros((temp2, temp1))
    for i in range(0, temp1):
        for j in range(0, temp2):
            result[j][i] = A[i][j]
    return result

def multiplyScalarToVector(scale, vect):
    # Checked and Works fine
    for i in range(0, len(vect)):
        vect[i] = scale*vect[i]
    return vect

def findQR(A):
    # Checked and Works fine
    dim4 = np.shape(A)
    size4 = np.size(A)
    row = dim4[0]
    col = size4//row

    q = np.zeros((row, col))
    r = np.zeros((col, col))
    for i in range(0, col):
        q[:, i] = A[:, i]
        for j in range (0, i):
            q[:, j] = getNormalisedVector(q[:, j])
            qTranspose = getTranspose(q[:, j].reshape(row, 1))
            r[j][i] = multiplyTwoMatricies(qTranspose, A[:, i].reshape(row, 1))
            q[:, i] = q[:, i] - multiplyScalarToVector(r[j][i], q[:, j])
            q[:, j] = getNormalisedVector(q[:, j])
        r[i][i] = getNorm(q[:, i])
        q[:, i] = multiplyScalarToVector(1/r[i][i], q[:, i])
    return [q, r]

#Only For Square symmetric matrix
def getEigenValues(A):
    #Checked and Works Fine
    for i in range(0, np.size(A)):
        # print('work')
        [Q, R] = findQR(A)
        A = multiplyTwoMatricies(R, Q)
    eValues = np.zeros(len(A))
    for i in range(0, len(A)):
        eValues[i] = A[i][i]
    return eValues

def getSingularValues(A):
    # Checked and Works fine
    result = np.zeros(len(A))
    for i in range(0, len(A)):
        if(A[i] < 0):
            A[i] = (-1) * A[i]
        result[i] = math.sqrt(A[i])
    return result

mat5 = np.zeros((3, 3))
mat5[0][0] = 1
mat5[0][1] = 3
mat5[0][2] = 4
mat5[1][0] = 3
mat5[1][1] = 1
mat5[1][2] = 2
mat5[2][0] = 4
mat5[2][1] = 2
mat5[2][2] = 1

# def getEigenVectors(A, ev):
#     # Find A-(Lambda)I
#     I = np.identity(len(A))
#     print(I)
#     result = np.zeros((len(A), len(A)))
#     for i in range(0, len(ev)):
#         temp = A - multiplyScalarToVector(ev[i], I)
#         print('temp is: \n', temp)
#         temp = Matrix(temp)
#         eVector = sp.
#         print('eksjdfn\n', len(eVector))
#         print('moiwpeir\n', eVector)
#         for j in range(0, len(eVector)):
#             result[:, j] = eVector
#     return result


def calculateSVD(eValues, eVectors, sv):
    # m = 3
    # n = 3
    U = np.zeros((n,n))
    V = np.zeros((m,m))
    sigma = np.zeros((n,m))

    # V
    for i in range(0, len(eValues)):
        V[:, i] = getNormalisedVector(eVectors[:, i])

    # Sigma
    for i in range(0, len(sv)):
        sigma[i][i] = sv[i]

    # U
    for i in range(0, m):
        # print('wait here also')
        temp = multiplyTwoMatricies(b, eVectors[:, i].reshape(m, 1)).reshape(1, m)
        U[:, i] = multiplyScalarToVector(1/sv[i], temp)
    return U, sigma, getTranspose(V)


# evalu, evect = np.linalg.eig(mat5)
# getU, getSigma, getV = calculateSVD(evalu, evect, getSingularValues(evalu))

# qw,er,ty = np.linalg.svd(mat5)

# print('getV: \n', getV)
# print('ty: \n', ty)

# print('getSigma \n', getSigma)
# print('er: \n', er)

# print('getU: \n', getU)
# print('qw: \n', qw)

#Read and show the image
img = Image.open('C:\\Users\\16692\\Documents\\ExtraProjects\\Image Blurring and Deblurring\\sample.png')
#img.show()

#Convert into gray scale
img2 = ImageOps.grayscale(img)
#img2.show()

#Convert image into matrix
b = np.array(img2)

#Fetch the dimensions of image
print('The dimension of the image is: ', b.shape)
n,m = b.shape
area = m*n

#Find the eigenvalues of b(Transpose)b
bT = getTranspose(b)
#print(bT)

#Find bTb
S = np.dot(b, bT)

# Find EigenValues of S
eValues1, eVectors = np.linalg.eig(S)
eValues = getEigenValues(S)

# Find Singular Values
singValues = getSingularValues(eValues)

# Compute SVD
UCheck, sigmaCheck, VTCheck = np.linalg.svd(b)
U, Sigma, VT = calculateSVD(eValues, eVectors, singValues)

# Blurring an image by taking small number of singular values(say 20)
k = 5

# Performing Slicing operations
resultantBlurredMatrixApproximated = U[:,:k] @ Sigma[0:k,:k] @ VT[:k,:]

# Deblurring an image by taking large number of singular values (say 1000)
k = 10
resultantDeblurredMatrixApproximated = U[:,:k] @ Sigma[0:k,:k] @ VT[:k,:]

# Performing Slicing operations
resultantDeblurredMatrixApproximated = U[:,:k] @ Sigma[0:k,:k] @ VT[:k,:]

# Final Image
k = 100
resultantDeblurredMatrixApproximatedFinal = U[:,:k] @ Sigma[0:k,:k] @ VT[:k,:]

# Show Result/Output
f, axes = plt.subplots(2,2)
plt.suptitle('Results')
axes[0][0].imshow(img)
axes[0][1].imshow(resultantBlurredMatrixApproximated, cmap='gray',vmin=0, vmax=255)
axes[1][0].imshow(resultantDeblurredMatrixApproximated, cmap='gray',vmin=0, vmax=255)
axes[1][1].imshow(resultantDeblurredMatrixApproximatedFinal, cmap='gray',vmin=0, vmax=255)
plt.show()

#https://www.google.com/search?q=circle+with+background&rlz=1C1CHBH_enUS881US881&tbm=isch&source=iu&ictx=1&fir=WLkjRe_Hdz_iBM%252CpEOBIXriMadvJM%252C_&vet=1&usg=AI4_-kSdzHQYwUGyqNjZfzDcy5EpSKvClQ&sa=X&ved=2ahUKEwj6i_mfyu3sAhVRxTgGHVXnC9IQ9QF6BAgDEG4&biw=1536&bih=722#imgrc=WLkjRe_Hdz_iBM