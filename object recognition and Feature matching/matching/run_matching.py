import cv2
import os
import numpy as np
import sys

def load_image(image_path):
    if not os.path.isfile(image_path):
        raise ValueError(f"File does not exist: {image_path}")
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Error loading image: {image_path}")
    if image.dtype != np.uint8:
        raise ValueError(f"Image depth is not 8-bit for image: {image_path}")
    return image

def find_object(base, obj, result):
    # Find SIFT descriptors for both base and object images
    sift = cv2.SIFT_create()
    keypts_base, descr_base = sift.detectAndCompute(base["image"], None)
    keypts_obj, descr_obj = sift.detectAndCompute(obj["image"], None)

    # Create and execute brute-force k-nn matching on descriptors
    FLANN_INDEX_KDTREE = 0
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)  # or pass an empty dictionary
    flann = cv2.FlannBasedMatcher(index_params, search_params)
    matches = flann.knnMatch(descr_obj, descr_base, k=2)

    matches = [[i] for i, j in matches if i.distance < 0.75 * j.distance]

    if len(matches) > 10:
        src_pts = np.float32([keypts_obj[m[0].queryIdx].pt for m in matches]).reshape(-1, 1, 2)
        dst_pts = np.float32([keypts_base[m[0].trainIdx].pt for m in matches]).reshape(-1, 1, 2)

        M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

        h, w, d = obj["image"].shape
        pts = np.float32([[0, 0], [0, h - 1], [w - 1, h - 1], [w - 1, 0]]).reshape(-1, 1, 2)
        dst = np.int32(cv2.perspectiveTransform(pts, M))

        matching_top_point = tuple(dst[np.argmin([x[0][1] for x in dst])][0])
        result = cv2.polylines(result, [dst], True, (100, 250, 200), 1, cv2.LINE_AA)
        cv2.putText(result, obj["name"], matching_top_point, cv2.FONT_HERSHEY_COMPLEX_SMALL, 0.7, (255, 255, 255), 1)
    else:
        print("Could not find any satisfying matches for {}".format(obj["name"]))
        return False

    matching_img = cv2.drawMatchesKnn(obj["image"], keypts_obj, base["image"], keypts_base, matches, None, flags=2)

    cv2.imshow('Feature matching for {}'.format(obj["name"]), matching_img)
    if cv2.waitKey(0) & 0xff == 27:
        cv2.destroyAllWindows()

    return True

def find_objects(base, objects):
    result = base["image"].copy()
    found_objects = []
    for obj in objects:
        if find_object(base, obj, result) == True:
            found_objects.append(obj["name"])

    cv2.imshow('Final result for {} (objects found: {})'.format(base["name"], ", ".join(found_objects)), result)
    if cv2.waitKey(0) & 0xff == 27:
        cv2.destroyAllWindows()

if __name__ == "__main__":
    # Get the base image and object images from the command-line arguments
    base_image_path = sys.argv[1]
    object_image_paths = sys.argv[2:]

    # Read the base image
    base = {"name": os.path.basename(base_image_path).split('.')[0], "image": load_image(base_image_path)}

    # Read the object images
    objects = [{"name": os.path.basename(x).split('.')[0], "image": load_image(x)} for x in object_image_paths]

    # Run the object matching
    find_objects(base, objects)
