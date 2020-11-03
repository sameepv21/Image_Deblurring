import numpy as np
import math
from PIL import Image, ImageOps
import matplotlib.pyplot as plt

def getNormalisedVector(vector):
    length = getNorm(vector)
    for i in range(0, len(vector)):
        vector[i] = vector[i]/length
    return vector

def calculateSVD(eValues, eVectors, sv):
    U = np.zeros((m,m))
    V = np.zeros((n,n))
    sigma = np.zeros((m,n))
    
    #Calculate V Matrix
    for i in range(0, len(eVectors[0])):
        V[:, i] = getNormalisedVector(eVectors[:, i])
    VT = getTranspose(V)

    #Calculate sigma Matrix
    print(len(sv))   
    for i in range(0, len(sv)):
        sigma[i][i] = sv[i]

    #Calculate U Matrix (bV[i]/sv[i])
    index = 0
    print(index)
    # print(m, n)
    # for i in range(0, n):
    #     if(index < n):
    #         U[:, i] = np.dot(b, V[:, index])
    #         index+=1
    # print(U.shape)
    return U, sigma, VT

def getNorm(A):
    norm = 0
    for i in range(0, len(A)):
        norm += A[i]*A[i]
    return math.sqrt(norm)

def findQR(A):
    q = np.zeros((2,2))
    r = np.zeros((2,2))
    for i in range(0, 2):
        q[i] = A[i]
        # print('q is: ', q)
        for j in range(0, i):
            qTranspose = np.transpose(q[j])
            print(qTranspose)
            # print('qTranspose ', qTranspose)
            # print(qTranspose.shape)
            #r[j][i] = np.dot(qTranspose[:, np.newaxis], A[i])
            q[i] = q[i] - r[j][i]*q[j]
        r[i,i] = getNorm(q[i])
        for j in range(0, 2):
            q[i][j] = q[i][j]/r[i][i]
    return [q, r]

def getSingularValues(A):
    result = np.zeros(len(A))
    for i in range(0, len(A)):
        result[i] = math.sqrt(A[i])
    return result

def getEigenValues(A):
    result, temp = findQR(A)
    resultCheck, temp2 = np.linalg.qr(A)
    print('result is: ', result)
    print('resultCheck is: ', resultCheck)
    for i in range(0, 4):
        [Q, R] = findQR(A)
        A = np.dot(R, Q)
    eValues = np.zeros(len(A))
    for i in range(0, 2):
        eValues[i] = A[i][i]

    return eValues

def getTranspose(A):
    temp1, temp2 = A.shape
    result = np.zeros((temp2, temp1))
    for i in range(0, temp1):
        for j in range(0, temp2):
            result[j][i] = A[i][j]
    return result

#Read and show the image
img = Image.open('C:\\Users\\16692\\Documents\\ExtraProjects\\Image Blurring and Deblurring\\sample.jpg')
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
matrix = [[1,0], [2,4]]
print('matrix transpose is: ', np.transpose(matrix))
answer = getEigenValues(matrix)
answerCheck, answerCheck2 = np.linalg.eig(matrix)
#For this, we require to find b(Transpose)
bT = getTranspose(b)
#print(bT)

#Find bTb
S = np.dot(b, bT)

#Find EigenValues of S
eValues, eVectors = np.linalg.eig(S)

#Find Singular Values
singValues = getSingularValues(eValues)

#Compute SVD
UCheck, sigmaCheck, VTCheck = np.linalg.svd(b)
U, sigma, VT = calculateSVD(eValues, eVectors, singValues)
# print('VT is: ', VT[:, 0])
# print('VCheck is: ', VTCheck[:, 0])

#Blurring an image by taking small number of singular values(say 20)
k = 20

#Performing Slicing operations
resultantBlurredMatrixApproximated = UCheck[:,:k] @ sigma[0:k,:k] @ VTCheck[:k,:]

#Deblurring an image by taking large number of singular values (say 1000)
k = 1000

#Performing Slicing operations
resultantDeblurredMatrixApproximated = UCheck[:,:k] @ sigma[0:k,:k] @ VTCheck[:k,:]
f, axes = plt.subplots(2,2)
plt.suptitle('Results')
axes[0][0].imshow(img)
axes[0][1].imshow(img2)
axes[1][0].imshow(resultantBlurredMatrixApproximated)
axes[1][1].imshow(resultantDeblurredMatrixApproximated)
plt.show()