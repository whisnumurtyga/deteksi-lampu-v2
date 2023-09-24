import cv2
from yolo_segmentation import YOLOSegmentation


def rszImg(image, rate):
    h, w, _ = image.shape
    new_h, new_w = (int(h*rate), int(w*rate))
    image = cv2.resize(image, (new_w, new_h))
    return image


path_nyala = '../../dataset-baru/beat-nyala/1.jpg'
path_mati = '../../dataset-baru/beat-mati/0.jpg'

    
img = cv2.imread(path_nyala)
# img = cv2.resize(img, None, fx=0.2, fy=0.2)
img = rszImg(img, 0.25)


# Segmentation
# ys = YOLOSegmentation('../../yolov8n-seg.pt')
ys = YOLOSegmentation('yolov8n-seg.pt')
bboxes, classes, segmentation, scores = ys.detect(img)

for bbox, class_id, seg, score in zip(bboxes, classes, segmentation, scores):
    # print("bbox:", bbox, " class_id:", class_id, " seg:", seg, " score:", score)
    (x, y, x2, y2) = bbox
    
    cv2.rectangle(img, (x,y), (x2,y2), (0,0,255), 2)



cv2.imshow('image', img)
cv2.waitKey(0)
cv2.destroyAllWindows()
