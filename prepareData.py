import os, shutil, time
import subprocess, shlex
import cv2
import json
import numpy as np
from sklearn.cluster import KMeans
import mySVM
#data_dir_name = "busesTrain/"
#output_file_name = "output.txt"
#final_output_file_name = "output_final.txt"
#gt_file_name = "annotationsTrain.txt"
intermediate_file_folder = "Intermediate/"
input_line_start = "python predict.py --predict "
input_line_end = " --load npz1.npz --config FPN.CASCADE=True FPN.NORM=GN BACKBONE.NORM=GN FPN.FRCNN_HEAD_FUNC=fastrcnn_4conv1fc_gn_head FPN.MRCNN_HEAD_FUNC=maskrcnn_up4conv_gn_head PREPROC.TRAIN_SHORT_EDGE_SIZE=[640,800] TRAIN.LR_SCHEDULE=4x"
IN_IMG_SIZE = (3648, 2736)
OUT_IMG_SIZE = (500, 375)


def rcnn_predict_pics(dirname):
    for filename in os.listdir(dirname):
        if(filename.endswith(".JPG")):
            cmdline = input_line_start + dirname + filename + input_line_end
            args = shlex.split(cmdline)
            subprocess.run(args)
        else:
            continue

    # move all txt files to intermediate folder
    try:
        os.mkdir(dirname + intermediate_file_folder)
    except FileExistsError:
        pass
    for filename in os.listdir(dirname):
        if(filename.startswith("DSCF") and filename.endswith(".txt")):
            shutil.move(dirname + filename, dirname + intermediate_file_folder + filename)
        else:
            continue


def prepare_svm_data(dirname, mode='fit'):
    final_dict = {}
    final_dict['pictures'] = []

    if (mode=='fit'):
        gt_file = open(dirname+'annotationsTrain.txt', 'r')
    for filename in os.listdir(dirname + intermediate_file_folder):
        if(filename.startswith("DSCF") and filename.endswith(".txt")):
            picname = os.path.splitext(filename)[0]+'.JPG'
            with open(dirname+intermediate_file_folder+filename, 'r') as infile:
                rcnn_predictions = json.load(infile)
            if (mode == 'fit'):
                for ln in gt_file:
                    if (ln.startswith(picname)):
                        gt_detections = parse_gt_line_to_dict(ln)
                        break
            else:
                gt_detections = []
            dict = {
                'picname': picname,
                'rcnn_predictions': rcnn_predictions['rcnn_predictions'],
                'gt_detections': gt_detections }
            dict_augmented = calculate_svm_input_data(dirname, dict)
            final_dict['pictures'].append(dict_augmented)
        else:
            continue

    with open(dirname+intermediate_file_folder+'rcnnPredictions.txt', 'w+') as outfile:
        json.dump(final_dict, outfile)


# in: an 'annotationTrain.txt' line
# out: an array [{},{},{}] of ground truth detection dicts, resized to fit output size.
def parse_gt_line_to_dict(line):
    out_arr = []
    resize_ratio = OUT_IMG_SIZE[0] / IN_IMG_SIZE[0]
    det_strs = line.split(':')[1].split(']')
    id = 0
    for det in det_strs[:-1]:   # last element is always '\n'
        str_arr = det.strip('[,').split(',')
        num_arr = []
        for s in str_arr[:-1]:
            num_arr.append( int( float(s)*resize_ratio ) )
        box = {
            'xmin': str(num_arr[0]),
            'ymin': str(num_arr[1]),
            'xmax': str(num_arr[0] + num_arr[2]),
            'ymax': str(num_arr[1] + num_arr[3]),
        }
        dict = {
            'id': str(id),
            'box': box,
            'color': str_arr[4]
        }
        out_arr.append(dict)
        id += 1
    return out_arr


# in : image dictionary { 'picname':'' , 'rcnn_predicitons':[] , 'gt_detections':[] }
# out: same dictionary augmented with additions to 'rcnn_predictions' entries
def calculate_svm_input_data(dirname, img_dict):
    img_path = dirname + img_dict['picname']
    img_raw = cv2.imread(img_path)
    img = cv2.cvtColor(cv2.resize(img_raw, OUT_IMG_SIZE), cv2.COLOR_BGR2Lab)
    #img = cv2.resize(img_raw, OUT_IMG_SIZE)

    # Iterate over all rcnn predicitons
    for pred in img_dict['rcnn_predictions']:

        # Calculate IOUs with all ground truth detections.
        pred['ious'] = []
        pred_box = {}
        for k, v in pred['box'].items():
            pred_box[k] = int(float(v))
        for gt_det in img_dict['gt_detections']:
            det_box = {}
            for k, v in gt_det['box'].items():
                det_box[k] = int(float(v))
            iou = get_iou(pred_box, det_box)
            pred['ious'].append({
                'id': gt_det['id'],
                'iou': str(iou)
            })

        # Calculate dominant color in HSV space
        img_crop = img[ pred_box['ymin']:pred_box['ymax'] , pred_box['xmin']:pred_box['xmax'] ]
        Z = np.float32(img_crop.reshape((-1, 3)))
        clt = KMeans(n_clusters=6)
        clt.fit(Z)
        numLabels = np.arange(0, len(np.unique(clt.labels_)) + 1)
        (hist, _) = np.histogram(clt.labels_, bins=numLabels)
        dominant_color = clt.cluster_centers_[np.argmax(hist)]
        pred['dominant_lab'] = {'l': str(dominant_color[0]), 'a': str(dominant_color[1]), 'b': str(dominant_color[2])}
    return img_dict

def get_iou(bb1, bb2):
    """
    Calculate the Intersection over Union (IoU) of two bounding boxes.

    Parameters
    ----------
    bb1 : dict
        Keys: {'xmin', 'xmax', 'ymin', 'ymax'}
        The (xmin, ymin) position is at the top left corner,
        the (xmax, ymax) position is at the bottom right corner
    bb2 : dict
        Keys: {'xmin', 'xmax', 'ymin', 'ymax'}
        The (xmin, ymin) position is at the top left corner,
        the (xmax, ymax) position is at the bottom right corner

    Returns
    -------
    float
        in [0, 1]
    """
    assert bb1['xmin'] < bb1['xmax']
    assert bb1['ymin'] < bb1['ymax']
    assert bb2['xmin'] < bb2['xmax']
    assert bb2['ymin'] < bb2['ymax']

    # determine the coordinates of the intersection rectangle
    x_left = max(bb1['xmin'], bb2['xmin'])
    y_top = max(bb1['ymin'], bb2['ymin'])
    x_right = min(bb1['xmax'], bb2['xmax'])
    y_bottom = min(bb1['ymax'], bb2['ymax'])

    if x_right < x_left or y_bottom < y_top:
        return 0.0

    # The intersection of two axis-aligned bounding boxes is always an
    # axis-aligned bounding box
    intersection_area = (x_right - x_left) * (y_bottom - y_top)

    # compute the area of both AABBs
    bb1_area = (bb1['xmax'] - bb1['xmin']) * (bb1['ymax'] - bb1['ymin'])
    bb2_area = (bb2['xmax'] - bb2['xmin']) * (bb2['ymax'] - bb2['ymin'])

    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    iou = intersection_area / float(bb1_area + bb2_area - intersection_area)
    assert iou >= 0.0
    assert iou <= 1.0
    return iou


def create_annotations_file(svm_predictions_file, output_file_name):
    # Assuming all photos are of the same size. (similar as inputs)
    resize_ratio = IN_IMG_SIZE[0] / OUT_IMG_SIZE[0]

    with open(svm_predictions_file, 'r') as infile:
        data_dict = json.load(infile)

    outfile = open(output_file_name, 'w+')
    for pic in data_dict['pictures']:
        pic_str = pic['picname'] + ':'
        for pred in pic['rcnn_predictions']:
            box = pred['box']
            toybus = float(pred['final_output']['toybus']) > 0
            color = int(float(pred['final_output']['color']))
            if(toybus):
                xmin = int( float(box['xmin']) * resize_ratio )
                ymin = int( float(box['ymin']) * resize_ratio )
                xmax = int( float(box['xmax']) * resize_ratio )
                ymax = int( float(box['ymax']) * resize_ratio )
                width = xmax-xmin
                height = ymax-ymin
                pred_str = '['
                for val in [xmin, ymin, width, height, color]:
                    pred_str = pred_str + str(val) + ','
                pred_str = pred_str[:-1] + ']'
                pic_str = pic_str + pred_str + ','
            else:
                continue
        pic_str = pic_str[:-1] + '\n'
        outfile.write(pic_str)
    outfile.close()

#rcnn_predict_pics("buses/train/")
#prepare_svm_data("buses/train/", mode='fit')
#mySVM.fit('buses/train/rcnnPredictions.txt', 'buses/train/svmPickle.pickle')

#rcnn_predict_pics("buses/test/")
#prepare_svm_data("buses/test/", mode='predict')
# data_file_name = 'buses/test/Intermediate/rcnnPredictions.txt'
# pickle_file_name = 'buses/train/svmPickle.pickle'
# output_file_name = 'buses/test/Intermediate/svmPredictions.txt'
# mySVM.predict(data_file_name, pickle_file_name, output_file_name)
create_annotations_file('buses/test/Intermediate/svmPredictions.txt', 'buses/test/myAnns.txt')




#unify_outputs()
#calc_max_ious()


# Calculate mean color in HSV space
# img_crop = img[ pred_box['ymin']:pred_box['ymax'] , pred_box['xmin']:pred_box['xmax'] ]
# meanhsv = cv2.mean(img_crop)
# pred['meanhsv'] = {'h':str(meanhsv[0]), 's':str(meanhsv[1]), 'v':str(meanhsv[2])}


# def unify_outputs():
#     outfile = open(output_file_name, 'w+')
#     for filename in os.listdir(data_dir_name):
#         if(filename.endswith(".txt")):
#             infile = open(data_dir_name+filename, 'r')
#             outfile.write(os.path.splitext(filename)[0]+'.JPG:')
#             for line in infile:
#                 outfile.write('[')
#                 box, score, cat = line.split(';')
#                 box = box.replace('[','').replace(']','').split(' ')
#                 box = [s for s in box if s!='']
#                 intbox = [int(float(b.strip(''))) for b in box]
#                 intbox[2] = intbox[2] - intbox[0]
#                 intbox[3] = intbox[3] - intbox[1]
#                 for b in intbox:
#                     outfile.write(str(b)+',')
#                 score = score.strip(' ')
#                 cat = cat.strip(' \n')
#                 outfile.write(str(cat)+','+str(score)+']')
#             outfile.write('\n')
#         else:
#             continue
#
#
# def calc_max_ious():
#     infile = open(output_file_name, 'r')
#     gtfile = open(gt_file_name, 'r')
#     outfile = open(final_output_file_name, 'w+')
#     for line in infile:
#         # Find corresponding line in the Ground Truth file.
#         # Generate list of all Ground Truth results in the picture.
#         picname = line[:8]
#         for ln in gtfile:
#             if(ln.startswith(picname)):
#                 gt_line = ln
#                 break
#         gt_results = gt_line.split('[')[1:]
#         spl = line.split('[')
#         newline = spl[0]
#         results = spl[1:]
#         # For each result in output file: calculate maximum iou.
#         for result in results:
#             if result.endswith('\n'):
#                 res = result[:-3].split(',')
#             else:
#                 res = result[:-2].split(',')
#             [xmin, ymin] = [int(a) for a in res[0:2]]
#             xmax = xmin + int(res[2])
#             ymax = ymin + int(res[3])
#             bb1 = {'x1': xmin, 'x2': xmax, 'y1': ymin, 'y2': ymax}
#             ious = []
#             for gt_result in gt_results:
#                 if gt_result.endswith('\n'):
#                     gt_res = gt_result[:-3].split(',')
#                 else:
#                     gt_res = gt_result[:-2].split(',')
#                 ratio = 500.0 / 3648.0
#                 [xmin1, ymin1] = [float(a) for a in gt_res[0:2]]
#                 xmax1 = xmin1 + float(gt_res[2])
#                 ymax1 = ymin1 + float(gt_res[3])
#                 [xmin1, ymin1, xmax1, ymax1] = [int(a*ratio) for a in [xmin1, ymin1, xmax1, ymax1]]
#                 bb2 = {'x1': xmin1, 'x2': xmax1, 'y1': ymin1, 'y2': ymax1}
#                 ious.append(get_iou(bb1, bb2))
#             max_iou = max(ious)
#
#             #calculate average color in the box.
#             #print(picname+".jpg")
#             img = cv2.resize(cv2.imread(data_dir_name+picname+".jpg"), (500, 375))
#             crop_img = cv2.cvtColor(img[ymin:ymax, xmin:xmax], cv2.COLOR_BGR2HSV)
#             mean_clr = cv2.mean(crop_img)
#             new_vals_str = str(round(max_iou, 2)) + ',' + str(int(float(mean_clr[0]))) + ',' + str(int(float(mean_clr[1])))+ ',' + str(int(float(mean_clr[2])))
#             newline = newline + '[' + new_vals_str +','+ result
#         outfile.write(newline)
