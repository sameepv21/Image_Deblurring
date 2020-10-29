from PIL import Image, ImageOps

#Read and Open Original Image
originalImage = Image.open('C:\\Users\\16692\\Documents\\ExtraProjects\\Image Blurring and Deblurring\\sample.jpg')
originalImage.show()

#Convert and showing grey scale images
grayScaleImage = ImageOps.grayscale(originalImage)
grayScaleImage.show()

#Fetch the dimensions
width, height = grayScaleImage.size
print(width, height)

#Input the number of singular values
k = input('Number of singular values')