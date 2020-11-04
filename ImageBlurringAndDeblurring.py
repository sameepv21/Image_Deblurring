import numpy as np
import math
from PIL import Image, ImageOps
import matplotlib.pyplot as plt


mat1 = np.matrix([[1, 1], [2, 2], [3, 3]])
# mat2 = [[1, 2]]

# answer = multiplyTwoMatricies(mat1, mat2)
# print(answer)

mat3 = np.zeros((3, 2))
mat4 = np.zeros((3, 1))
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

# print('result is: \n', multiplyTwoMatricies(mat2, mat1))


def getNorm(A):
    # Checked and Works Fine
    norm = 0
    for i in range(0, len(A)):
        norm += A[i]*A[i]
    return math.sqrt(norm)

# print(getNorm(mat1))


def getNormalisedVector(vector):
    # Checked and Works fine
    length = getNorm(vector)
    for i in range(0, len(vector)):
        vector[i] = vector[i]/length
    return vector

# print(getNormalisedVector(mat1))

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

# print(getTranspose(mat1))

def multiplyScalarToVector(scale, vect):
    # Checked and Works fine
    for i in range(0, len(vect)):
        vect[i] = scale*vect[i]
    return vect

# print(mat1.reshape(3, 1))
# print(multiplyScalarToVector(2, mat1.reshape(3, 1)))

def findQR(A):
    dim4 = np.shape(A)
    size4 = np.size(A)
    row = dim4[0]
    col = size4//row

    q = np.zeros((row, col))
    r = np.zeros((col, col))
    for i in range(0, col):
        q[:, i] = A[:, i]
        for j in range (0, i-1):
            qTranspose = getTranspose(q[:, j].reshape(row, 1))
            r[j][i] = multiplyTwoMatricies(qTranspose, A[:, i].reshape(row, 1))
            q[:, i] = q[:, i] - multiplyScalarToVector(r[j][i], q[:, j])
        r[i][i] = getNorm(q[:, i].reshape(row, 1))
        q[:, i] = multiplyScalarToVector(1/r[i][i], q[:, i]).reshape((1, row))
    return [q, r]



mat5 = np.zeros((3, 3))
mat5[0][0] = 1
mat5[0][1] = 1
mat5[1][0] = 1
mat5[1][2] = 1
mat5[2][1] = 1
mat5[2][2] = 1
print(mat5)

Q, R = findQR(mat5)
print('Q is \n', Q)
print('R is: \n', R)


# def calculateSVD(eValues, eVectors, sv):
#     U = np.zeros((m,m))
#     V = np.zeros((n,n))
#     sigma = np.zeros((m,n))
    
#     #Calculate V Matrix
#     for i in range(0, len(eVectors[0])):
#         V[:, i] = getNormalisedVector(eVectors[:, i])
#     VT = getTranspose(V)

#     #Calculate sigma Matrix
#     print(len(sv))   
#     for i in range(0, len(sv)):
#         sigma[i][i] = sv[i]

#     #Calculate U Matrix (bV[i]/sv[i])
#     index = 0
#     # print(index)
#     # # print(m, n)
#     # # for i in range(0, n):
#     # #     if(index < n):
#     # #         U[:, i] = np.dot(b, V[:, index])
#     # #         index+=1
#     # # print(U.shape)
#     return U, sigma, VT

# def getSingularValues(A):
#     result = np.zeros(len(A))
#     for i in range(0, len(A)):
#         result[i] = math.sqrt(A[i])
#     return result

# def getEigenValues(A):
#     result, temp = findQR(A)
#     resultCheck, temp2 = np.linalg.qr(A)
#     print('result is: ', result)
#     print('resultCheck is: ', resultCheck)
#     for i in range(0, 4):
#         [Q, R] = findQR(A)
#         A = np.dot(R, Q)
#     eValues = np.zeros(len(A))
#     for i in range(0, 2):
#         eValues[i] = A[i][i]

#     return eValues

# #Read and show the image
# img = Image.open('C:\\Users\\16692\\Documents\\ExtraProjects\\Image Blurring and Deblurring\\sample.jpg')
# #img.show()

# #Convert into gray scale
# img2 = ImageOps.grayscale(img)
# #img2.show()

# #Convert image into matrix
# b = np.array(img2)

# #Fetch the dimensions of image
# print('The dimension of the image is: ', b.shape)
# n,m = b.shape
# area = m*n

# #Find the eigenvalues of b(Transpose)b
# matrix = [[1,0], [2,4]]
# print('matrix transpose is: ', np.transpose(matrix))
# answer = getEigenValues(matrix)
# answerCheck, answerCheck2 = np.linalg.eig(matrix)
# #For this, we require to find b(Transpose)
# bT = getTranspose(b)
# #print(bT)

# #Find bTb
# S = np.dot(b, bT)

# #Find EigenValues of S
# eValues, eVectors = np.linalg.eig(S)

# #Find Singular Values
# singValues = getSingularValues(eValues)

# #Compute SVD
# UCheck, sigmaCheck, VTCheck = np.linalg.svd(b)
# U, sigma, VT = calculateSVD(eValues, eVectors, singValues)
# # print('VT is: ', VT[:, 0])
# # print('VCheck is: ', VTCheck[:, 0])

# #Blurring an image by taking small number of singular values(say 20)
# k = 20

# #Performing Slicing operations
# resultantBlurredMatrixApproximated = UCheck[:,:k] @ sigma[0:k,:k] @ VTCheck[:k,:]

# #Deblurring an image by taking large number of singular values (say 1000)
# k = 1000

# #Performing Slicing operations
# resultantDeblurredMatrixApproximated = UCheck[:,:k] @ sigma[0:k,:k] @ VTCheck[:k,:]
# f, axes = plt.subplots(2,2)
# plt.suptitle('Results')
# axes[0][0].imshow(img)
# axes[0][1].imshow(img2)
# axes[1][0].imshow(resultantBlurredMatrixApproximated)
# axes[1][1].imshow(resultantDeblurredMatrixApproximated)
# # plt.show()