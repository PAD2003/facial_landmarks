import sys
sys.path.append("src/facenet_pytorch")
sys.path.append("src/data")

from deepface import DeepFace
from PIL import Image
import pyrootutils
from PIL import Image, ImageDraw
from torchvision import transforms
import torch
import numpy as np
from models.dlib_module import DlibModule
import sys
from models.components.simple_resnet import SimpleResnet
import cv2
import time
import csv
import faceBlendCommon as fbc
import math

# do you want to display landmarks and bounding box or not?
VISUALIZE_FACE_POINTS = False
VISUALIZE_FILTER = True
source = 0
# source = "deploy/data/IMG_1842.MOV"
# source = "deploy/data/barack-obama-500.jpg"

# config for filter
filters_config = {
    'squid_game_front_man':
        [{'path': "filters/squid_game_front_man.png",
          'anno_path': "filters/squid_game_front_man.csv",
          'morph': True, 'animated': False, 'has_alpha': True}],
    'anonymous':
        [{'path': "filters/anonymous.png",
          'anno_path': "filters/anonymous_annotations.csv",
          'morph': True, 'animated': False, 'has_alpha': True}],
    'dog':
        [{'path': "filters/dog-ears.png",
          'anno_path': "filters/dog-ears_annotations.csv",
          'morph': False, 'animated': False, 'has_alpha': True},
         {'path': "filters/dog-nose.png",
          'anno_path': "filters/dog-nose_annotations.csv",
          'morph': False, 'animated': False, 'has_alpha': True}],
    'cat':
        [{'path': "filters/cat-ears.png",
          'anno_path': "filters/cat-ears_annotations.csv",
          'morph': False, 'animated': False, 'has_alpha': True},
         {'path': "filters/cat-nose.png",
          'anno_path': "filters/cat-nose_annotations.csv",
          'morph': False, 'animated': False, 'has_alpha': True}],
    'jason-joker':
        [{'path': "filters/jason-joker.png",
          'anno_path': "filters/jason-joker_annotations.csv",
          'morph': True, 'animated': False, 'has_alpha': True}],
    'flower-crown':
        [{'path': "filters/flower-crown.png",
          'anno_path': "filters/flower-crown_annotations.csv",
          'morph': False, 'animated': False, 'has_alpha': True}],
}

# set up path
path = pyrootutils.find_root(search_from=__file__, indicator=".project-root")

# create transforms
mean=[0.485, 0.456, 0.406]
std=[0.229, 0.224, 0.225]
simple_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=mean, std=std)
])

# load model
model = DlibModule.load_from_checkpoint(net=SimpleResnet(), checkpoint_path="deploy/checkpoints/last3.ckpt")

# functions
def get_landmarks(img: Image) -> np.array:
    # prepare input
    input = simple_transform(img).unsqueeze(dim=0)

    # use model to predict
    model.eval()
    with torch.inference_mode():
        output = model(input)
        output = output.squeeze()

    # denormalise output
    landmarks = (output + 0.5) * np.array([box["w"], box["h"]]) + np.array([box["x"], box["y"]])

    # add landmarks
    landmarks = np.vstack([landmarks, np.array([landmarks[0][0], int(box["y"])])])
    landmarks = np.vstack([landmarks, np.array([landmarks[16][0], int(box["y"])])])

    return landmarks

def load_filter_img(img_path, has_alpha):
    # Read the image
    img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)

    alpha = None
    if has_alpha:
        b, g, r, alpha = cv2.split(img)
        img = cv2.merge((b, g, r))

    return img, alpha

def load_landmarks(annotation_file):
    with open(annotation_file) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=",")
        points = {}
        for i, row in enumerate(csv_reader):
            # skip head or empty line if it's there
            try:
                x, y = int(row[1]), int(row[2])
                points[row[0]] = (x, y)
            except ValueError:
                continue
        return points

def find_convex_hull(points):
    hull = []
    hullIndex = cv2.convexHull(np.array(list(points.values())), clockwise=False, returnPoints=False)
    addPoints = [
        [48], [49], [50], [51], [52], [53], [54], [55], [56], [57], [58], [59],  # Outer lips
        [60], [61], [62], [63], [64], [65], [66], [67],  # Inner lips
        [27], [28], [29], [30], [31], [32], [33], [34], [35],  # Nose
        [36], [37], [38], [39], [40], [41], [42], [43], [44], [45], [46], [47],  # Eyes
        [17], [18], [19], [20], [21], [22], [23], [24], [25], [26]  # Eyebrows
    ]
    hullIndex = np.concatenate((hullIndex, addPoints))
    for i in range(0, len(hullIndex)):
        hull.append(points[str(hullIndex[i][0])])

    return hull, hullIndex

def load_filter(filter_name="dog"):

    filters = filters_config[filter_name]

    multi_filter_runtime = []

    for filter in filters:
        temp_dict = {}

        img1, img1_alpha = load_filter_img(filter['path'], filter['has_alpha'])

        temp_dict['img'] = img1
        temp_dict['img_a'] = img1_alpha

        points = load_landmarks(filter['anno_path'])

        temp_dict['points'] = points

        if filter['morph']:
            # Find convex hull for delaunay triangulation using the landmark points
            hull, hullIndex = find_convex_hull(points)

            # Find Delaunay triangulation for convex hull points
            sizeImg1 = img1.shape
            rect = (0, 0, sizeImg1[1], sizeImg1[0])
            dt = fbc.calculateDelaunayTriangles(rect, hull)

            temp_dict['hull'] = hull
            temp_dict['hullIndex'] = hullIndex
            temp_dict['dt'] = dt

            if len(dt) == 0:
                continue

        if filter['animated']:
            filter_cap = cv2.VideoCapture(filter['path'])
            temp_dict['cap'] = filter_cap

        multi_filter_runtime.append(temp_dict)

    return filters, multi_filter_runtime

# prepare video capture
cap = cv2.VideoCapture(source)
cap_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))

# create the output video file
if source != 0:
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter('deploy/outputs/annotated_video.mp4', fourcc, fps, frame_size)

# prepare for filter
iter_filter_keys = iter(filters_config.keys())
filters, multi_filter_runtime = load_filter(next(iter_filter_keys))
isFirstFrame = True
sigma = 50

# loop through each frame
prev_time = time.time()
while(cap.isOpened()):
    # common
    ret, frame = cap.read()
    if ret == False:
        break
    if source == 0:
        frame = cv2.flip(frame, 1)
    
    # frame_image
    frame_image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

    # detect faces
    resp_objs = DeepFace.extract_faces(img_path=frame, target_size=(224, 224), detector_backend="opencv", enforce_detection=False)
    if resp_objs is not None:
        for resp_obj in resp_objs:
            # deal with extract_faces
            if resp_obj["facial_area"]["w"] == cap_width:
                break
            box = resp_obj["facial_area"]

            # prepare input image: crop & transform
            input_image = frame_image.crop(box=(box["x"], box["y"], box["x"] + box["w"], box["y"] + box["h"]))

            # 1. Points2
            landmarks = get_landmarks(input_image) 
            points2 = landmarks.tolist()

            # 2. Optical Flow
            ################ Optical Flow and Stabilization Code #####################
            img2Gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            if isFirstFrame:
                points2Prev = np.array(points2, np.float32)
                img2GrayPrev = np.copy(img2Gray)
                isFirstFrame = False

            lk_params = dict(winSize=(101, 101), maxLevel=15,
                            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 20, 0.001))
            points2Next, st, err = cv2.calcOpticalFlowPyrLK(img2GrayPrev, img2Gray, points2Prev,
                                                            np.array(points2, np.float32),
                                                            **lk_params)

            # Final landmark points are a weighted average of detected landmarks and tracked landmarks

            for k in range(0, len(points2)):
                d = cv2.norm(np.array(points2[k]) - points2Next[k])
                alpha = math.exp(-d * d / sigma)
                points2[k] = (1 - alpha) * np.array(points2[k]) + alpha * points2Next[k]
                points2[k] = fbc.constrainPoint(points2[k], frame.shape[1], frame.shape[0])
                points2[k] = (int(points2[k][0]), int(points2[k][1]))

            # Update variables for next pass
            points2Prev = np.array(points2, np.float32)
            img2GrayPrev = img2Gray
            ################ End of Optical Flow and Stabilization Code ###############
            
            # applying filter
            if VISUALIZE_FILTER == True:
                for idx, filter in enumerate(filters):

                    filter_runtime = multi_filter_runtime[idx]
                    img1 = filter_runtime['img']
                    points1 = filter_runtime['points']
                    img1_alpha = filter_runtime['img_a']

                    if filter["morph"]:
                        # 3. Delaunay Triangulation & 4. Apply Filter

                        hull1 = filter_runtime['hull']
                        hullIndex = filter_runtime['hullIndex']
                        dt = filter_runtime['dt']

                        # create copy of frame
                        warped_img = np.copy(frame)

                        # Find convex hull
                        hull2 = []
                        for i in range(0, len(hullIndex)):
                            hull2.append(points2[hullIndex[i][0]])

                        mask1 = np.zeros((warped_img.shape[0], warped_img.shape[1]), dtype=np.float32)
                        mask1 = cv2.merge((mask1, mask1, mask1))
                        img1_alpha_mask = cv2.merge((img1_alpha, img1_alpha, img1_alpha))

                        # Warp the delaunay triangles
                        for i in range(0, len(dt)):
                            t1 = []
                            t2 = []

                            for j in range(0, 3):
                                t1.append(hull1[dt[i][j]])
                                t2.append(hull2[dt[i][j]])

                            fbc.warpTriangle(img1, warped_img, t1, t2)
                            fbc.warpTriangle(img1_alpha_mask, mask1, t1, t2)

                        # Blur the mask before blending
                        mask1 = cv2.GaussianBlur(mask1, (3, 3), 10)

                        mask2 = (255.0, 255.0, 255.0) - mask1

                        # Perform alpha blending of the two images
                        temp1 = np.multiply(warped_img, (mask1 * (1.0 / 255)))
                        temp2 = np.multiply(frame, (mask2 * (1.0 / 255)))
                        output = temp1 + temp2
                    else:
                        dst_points = [points2[int(list(points1.keys())[0])], points2[int(list(points1.keys())[1])]]
                        tform = fbc.similarityTransform(list(points1.values()), dst_points)

                        # Apply similarity transform to input image
                        trans_img = cv2.warpAffine(img1, tform, (frame.shape[1], frame.shape[0]))
                        trans_alpha = cv2.warpAffine(img1_alpha, tform, (frame.shape[1], frame.shape[0]))
                        mask1 = cv2.merge((trans_alpha, trans_alpha, trans_alpha))

                        # Blur the mask before blending
                        mask1 = cv2.GaussianBlur(mask1, (3, 3), 10)

                        mask2 = (255.0, 255.0, 255.0) - mask1

                        # Perform alpha blending of the two images
                        temp1 = np.multiply(trans_img, (mask1 * (1.0 / 255)))
                        temp2 = np.multiply(frame, (mask2 * (1.0 / 255)))
                        output = temp1 + temp2
                    
                    frame = output = np.uint8(output)
            
            # draw landmarks and bounding box
            if VISUALIZE_FACE_POINTS:
                # draw landmarks on frame image (not input image)
                for i, l in enumerate(landmarks):
                    x = int(l[0])
                    y = int(l[1])
                    cv2.circle(frame, (x, y), 2, (0, 255, 0), -1)
                    # cv2.putText(frame, str(i), (x, y), cv2.FONT_HERSHEY_SIMPLEX, .6, (255, 255, 255), 1)
                
                # # draw boxs
                # x1 = int(box["x"])
                # y1 = int(box["y"])
                # x2 = int(box["x"] + box["w"])
                # y2 = int(box["y"] + box["h"])
                # cv2.rectangle(frame, (x1, y1), (x2, y2), color=(0, 255, 0))
        
    if source == 0:
        # # fps
        # cur_time = time.time()
        # print(1 / (cur_time - prev_time))
        # prev_time = cur_time

        # show frame
        cv2.imshow("Filter app", frame)

        # handle keypress
        keypressed = cv2.waitKey(1) & 0xFF
        if keypressed == 27:
            break
        elif keypressed == ord('f'):
            try:
                filters, multi_filter_runtime = load_filter(next(iter_filter_keys))
            except:
                iter_filter_keys = iter(filters_config.keys())
                filters, multi_filter_runtime = load_filter(next(iter_filter_keys))
    else:
        out.write(frame)

# save & free resource
cap.release()
if source != 0:
    out.release()
cv2.destroyAllWindows()