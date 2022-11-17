import cv2
import numpy as np

def preprocessing(img) :
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return (img, img_gray)

template_img = cv2.imread("Template4.png")
template_img, template_gray = preprocessing(template_img)

sift = cv2.SIFT_create()
bf = cv2.BFMatcher()

def feature_object_detection(template_img, template_gray, query_img, query_gray, min_match_number) :
    template_kpts, template_desc = sift.detectAndCompute(template_gray, None)
    query_kpts, query_desc = sift.detectAndCompute(query_gray, None)
    matches = bf.knnMatch(template_desc, query_desc, k=2)
    good_matches = list()
    good_matches_list = list()
    for m, n in matches :
        if m.distance < 0.8*n.distance :
            good_matches.append(m)
            good_matches_list.append([m])
    
    if len(good_matches) > min_match_number :
        src_pts = np.float32([ template_kpts[m.queryIdx].pt for m in good_matches ]).reshape(-1,1,2)
        dst_pts = np.float32([ query_kpts[m.trainIdx].pt for m in good_matches ]).reshape(-1,1,2)

        H, inlier_masks = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
        h, w = template_img.shape[:2]
        template_box = np.float32([[0, 0], [0, h-1], [w-1, h-1], [w-1, 0]]).reshape(-1,1,2)
        transformed_box = cv2.perspectiveTransform(template_box, H)

        detected_img = cv2.polylines(query_img, [np.int32(transformed_box)], True, (255,0,0), 3, cv2.LINE_AA)
        drawmatch_img = cv2.drawMatchesKnn(template_img, template_kpts, detected_img, query_kpts, good_matches_list, None, flags=2, matchesMask=inlier_masks)

        return detected_img, drawmatch_img
    else :
        print('Keypoints not enough')
        return

vid = cv2.VideoCapture('Right_output.avi')
  
while(True):
    ret, frame = vid.read()
    frame, frame_gray = preprocessing(frame)
    detected_img, drawmatch_img = feature_object_detection(template_img, template_gray, frame, frame_gray, 1)
    cv2.imshow('frame', detected_img)
    if cv2.waitKey(60) & 0xFF == ord('q'):
        break
vid.release()
cv2.destroyAllWindows()