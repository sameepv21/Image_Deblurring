import numpy as np
from PIL import Image, ImageOps

def getEigenValues(A):
    print(A)

def getTranspose(A):
    result = np.zeros((m,n))
    for i in range(0, n):
        for j in range(0, m):
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
eValues = getEigenValues(S)