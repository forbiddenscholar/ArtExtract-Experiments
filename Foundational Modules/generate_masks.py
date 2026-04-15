import cv2

img_dir = '/home/daddy/Downloads/Temporary/ArtExtract-Experiments/img/'

rgb_img = cv2.imread(img_dir + 'rgb/oil_painting_RGB.bmp')
# now this converted rgb is the surface baseline
rgb_img = cv2.cvtColor(rgb_img, cv2.COLOR_BGR2GRAY)

# the nir image is the penetrated layer
nir_img = cv2.imread(img_dir + 'msi/oil_painting_ms_31.png', 0)

# this is the subtracted image
sub_img = cv2.subtract(nir_img, rgb_img)

ret, thresh_img = cv2.threshold(sub_img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
cv2.imwrite('../img/output/subtracted_img.png', thresh_img)

cv2.imshow('Original', rgb_img)
cv2.imshow('NIR', nir_img)
cv2.imshow('Subtracted', sub_img)
cv2.imshow('Otsu Mask', thresh_img)
cv2.waitKey(0)
cv2.destroyAllWindows()
