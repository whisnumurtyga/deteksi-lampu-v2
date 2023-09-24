import cv2

path_nyala = '../../dataset-baru/beat-nyala/1.jpg'
path_mati = '../../dataset-baru/beat-mati/0.jpg'

def rszImg(image, rate):
    h, w, _ = image.shape
    new_h, new_w = (int(h*rate), int(w*rate))
    image = cv2.resize(image, (new_w, new_h))
    return image
    
img = cv2.imread(path_nyala)
# img = cv2.resize(img, None, fx=0.2, fy=0.2)
img = rszImg(img, 0.25)

cv2.imshow('image', img)
cv2.waitKey(0)
cv2.destroyAllWindows()
