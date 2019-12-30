import sklearn.model_selection as skm
import sklearn.svm as svm
import numpy as np
import matplotlib.pyplot as plt
import json, pickle
from mpl_toolkits.mplot3d import Axes3D

IOU_TRESHOLD = 0.7
#data_file_name = "output_final.txt"


def fit(data_filename, pickle_filename):
    with open(data_filename, 'r') as infile:
        data_dict = json.load(infile)
    X_toybus, X_color, Y_toybus, Y_color = extract_data_for_training(data_dict)
    clf1 = svm.SVC()
    clf1.fit(X_toybus, np.ravel(Y_toybus))

    idxs = Y_toybus[:, -1] > 0
    X_color = X_color[idxs, :]
    Y_color = Y_color[idxs, :]

    clf2 = svm.SVC()
    clf2.fit(X_color, np.ravel(Y_color))

    with open(pickle_filename, 'wb+') as outfile:
        #pickle.dump(clf1, outfile)
        pickle.dump([clf1, clf2], outfile)



def predict(data_filename, pickle_filename, output_filename):
    with open(data_filename, 'r') as infile:
        data_dict = json.load(infile)
    with open(pickle_filename, 'rb') as infile:
        [clf1, clf2] = pickle.load(infile)
        #clf2 = pickle.load(infile)

    for picdata in data_dict['pictures']:
        for pred in picdata['rcnn_predictions']:
            x_toybus, x_color, _, _ = extract_data_single_pred(pred, picdata, mode='predict')
            y_toybus = clf1.predict([x_toybus])
            if(y_toybus > 0):
                y_color = clf2.predict([x_color])
            else:
                y_color = ['0']
            pred['final_output'] = {
                'toybus': str(y_toybus[0]),
                'color': str(y_color[0])
            }
    with open(output_filename, 'w+') as outfile:
        json.dump(data_dict, outfile)



def extract_data_for_training(data_dict):
    X_toybus = np.empty((0, 3), int)
    Y_toybus = np.empty((0, 1), int)
    X_color = np.empty((0, 3), int)
    Y_color = np.empty((0, 1), int)

    # if (mode == 'fit'):
    #     iterable = data_dict['pictures']
    # elif (mode == 'predict'):
    #     for pic in data_dict['pictures']:
    #         if(pic['picname'] == picname):
    #             iterable = [pic]

    for picdata in data_dict['pictures']:
        # Iterate over all rcnn predictions
        #pred_ints = {}
        for pred in picdata['rcnn_predictions']:
            x_toybus, x_color, y_toybus, y_color = extract_data_single_pred(pred, picdata, mode='fit')
            X_toybus = np.append(X_toybus, np.array([x_toybus]), axis=0)
            X_color = np.append(X_color, np.array([x_color]), axis=0)
            Y_toybus = np.append(Y_toybus, np.array([[y_toybus]]), axis=0)
            Y_color = np.append(Y_color, np.array([[y_color]]), axis=0)

            # # Extract training data for toybus classifier: "typical size", class, score (as floats) ;  Y: binary label (+1,-1).
            # # Extract training data for color classifier: X: dominant-L, dominant-a, dominant-b (as floats) ;  Y: actual color (1-6).
            # box = {}
            # for k, v in pred['box'].items():
            #     box[k] = float(v)
            # ious = {}
            # for item in pred['ious']:
            #     ious[item['id']] = float(item['iou'])
            # domlab = {}
            # for k, v in pred['dominant_hsv'].items():
            #     domlab[k] = float(v)
            # typ_size = (box['xmax']-box['xmin'] + box['ymax']-box['ymin']) / 2
            # x = [typ_size, float(pred['class']), float(pred['score'])]
            # X_toybus = np.append(X_toybus, np.array([x]), axis=0)
            # if (mode=='fit'):
            #     max_iou = max([val for val in ious.values()])
            #     y = 1 if max_iou > IOU_TRESHOLD else -1
            #     Y_toybus = np.append(Y_toybus, np.array([[y]]), axis=0)
            #
            # x = [domlab['l'], domlab['a'], domlab['b']]
            # # search for the matching ground truth detection.
            # for k, v in ious.items():
            #     if v == max_iou:
            #         matching_id = k
            #         break
            # X_color = np.append(X_color, np.array([x]), axis=0)
            # if (mode=='fit'):
            #     for det in picdata['gt_detections']:
            #         if det['id'] == matching_id:
            #             y = det['color']
            #             break
            #     Y_color = np.append(Y_color, np.array([[y]]), axis=0)

    return X_toybus, X_color, Y_toybus, Y_color


def extract_data_single_pred(pred_dict, pic_dict, mode='fit'):
    # Extract training data for toybus classifier: "typical size", class, score (as floats) ;  Y: binary label (+1,-1).
    # Extract training data for color classifier: X: dominant-L, dominant-a, dominant-b (as floats) ;  Y: actual color (1-6).
    box = {}
    for k, v in pred_dict['box'].items():
        box[k] = float(v)
    ious = {}
    for item in pred_dict['ious']:
        ious[item['id']] = float(item['iou'])
    domlab = {}
    for k, v in pred_dict['dominant_lab'].items():
        domlab[k] = float(v)
    typ_size = (box['xmax'] - box['xmin'] + box['ymax'] - box['ymin']) / 2
    x_toybus = [typ_size, float(pred_dict['class']), float(pred_dict['score'])]
    #X_toybus = np.append(X_toybus, np.array([x]), axis=0)
    if (mode == 'fit'):
        max_iou = max([val for val in ious.values()])
        y_toybus = 1 if max_iou > IOU_TRESHOLD else -1
    else:
        y_toybus = 0
        #Y_toybus = np.append(Y_toybus, np.array([[y]]), axis=0)

    x_color = [domlab['l'], domlab['a'], domlab['b']]
    # search for the matching ground truth detection.

    #X_color = np.append(X_color, np.array([x]), axis=0)
    if (mode == 'fit'):
        for k, v in ious.items():
            if v == max_iou:
                matching_id = k
                break
        for det in pic_dict['gt_detections']:
            if det['id'] == matching_id:
                y_color = det['color']
                break
    else:
        y_color = 0
        #Y_color = np.append(Y_color, np.array([[y]]), axis=0)

    return x_toybus, x_color, y_toybus, y_color


# for line in infile:
#     results = line.split('[')[1:]
#     for result in results:
#         if result.endswith('\n'):
#             res = result[:-3].split(',')
#         else:
#             res = result[:-2].split(',')
#         max_iou = float(res[0])
#         label_binary = 1 if max_iou > IOU_TRESHOLD else -1
#         meanh, means, meanv = [int(res[i]) for i in range(1,3)]
#         typical_size = int((int(res[6]) + int(res[7])) / 2)
#         cls = int(res[8])
#         #label_color =
#         X = np.append(X, np.array([[typical_size, cls]]), axis=0)
#         Y = np.append(Y, np.array([[label_binary]]), axis=0)




# X_train1, X_test1, y_train1, y_test1 = skm.train_test_split(X_toybus, Y_toybus, test_size=0.33)
# clf1 = svm.SVC()
# clf1.fit(X_train1, np.ravel(y_train1))
# y_pred1 = clf1.predict(X_test1)
# err = np.sum(np.abs(y_pred1 - np.ravel(y_test1))) / 2
# acc = 1 - err / y_test1.shape[0]
# print(acc)
#
#
# idxs = Y_toybus[:,-1] > 0
# X_color = X_color[idxs, :]
# Y_color = Y_color[idxs, :]
#
# X_train2, X_test2, y_train2, y_test2 = skm.train_test_split(X_color, Y_color, test_size=10)
# #Cs = np.linspace(1.0, 2.0, 15)
# #for C in Cs:
# clf2 = svm.SVC(C=C)
# clf2.fit(X_train2, np.ravel(y_train2))
# y_pred2 = clf2.predict(X_test2)
# err = np.sum((y_pred2 != np.ravel(y_test2)).astype(int))
# acc = 1 - err / y_test2.shape[0]
# print(str(round(C*100)/100) + ': ' +str(acc))
#
# # #plt.scatter(X_train2[:,0], X_train2[:,1], c=np.ravel(y_train2))
# # x_min, x_max = X_color[:, 0].min() - 1, X_color[:, 0].max() + 1
# # y_min, y_max = X_color[:, 1].min() - 1, X_color[:, 1].max() + 1
# # xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1),
# #                      np.arange(y_min, y_max, 0.1))
# # Z = clf2.predict(np.c_[xx.ravel(), yy.ravel()])
# # Z = Z.reshape(xx.shape)
# #
# # fig1, ax2 = plt.subplots(constrained_layout=True)
# # ax2.contourf(xx, yy, Z, levels=6, alpha=0.4, cmap='Spectral')
# aux = np.ravel(y_train2)
# c=np.true_divide(aux.astype(float), np.ones(aux.shape)*6.0)
# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')
# ax.scatter(X_train2[:, 0], X_train2[:, 1], X_train2[:,2], c=c,
#                               s=20, edgecolor='k', cmap='Spectral')
# ax.set_xlabel('H')
# ax.set_ylabel('S')
# ax.set_zlabel('V')
#
# plt.show()
#




#plt.scatter(X_train[:,0], X_train[:,1], c=np.ravel(y_train))
# x_min, x_max = X_toybus[:, 0].min() - 1, X_toybus[:, 0].max() + 1
# y_min, y_max = X_toybus[:, 1].min() - 1, X_toybus[:, 1].max() + 1
# xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1),
#                      np.arange(y_min, y_max, 0.1))
# Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
# Z = Z.reshape(xx.shape)
#
# fig1, ax2 = plt.subplots(constrained_layout=True)
# ax2.contourf(xx, yy, Z, alpha=0.4)
# ax2.scatter(X_train[:, 0], X_train[:, 1], c=np.ravel(y_train),
#                               s=20, edgecolor='k')
# ax2.set_title('Decision Boundaries for our SVM')
# ax2.set_xlabel('Typical box size ( (width+height)/2 )')
# ax2.set_ylabel('Class')
#
# plt.show()