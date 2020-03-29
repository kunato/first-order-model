import face_alignment
import cv2
import numpy as np


def main():
    img = cv2.imread('demo/p_29507.jpg')
    fa = face_alignment.FaceAlignment(face_alignment.LandmarksType._2D, flip_input=False, device='cpu')
    bboxes = fa.face_detector.detect_from_image(img)
    bbox = bboxes[0]
    bbox = np.array(bbox, dtype=np.int32)
    print(bbox)
    crop_img = img[bbox[1]-80:bbox[3], bbox[0]-20: bbox[2]+20,:]
    crop_img = cv2.resize(crop_img, (256, 256))
    cv2.imwrite('demo/crop.png', crop_img)



if __name__ == "__main__":
    main()