import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
from AnalyzeFrame import get_digits
from skimage.io import imread
import cv2
from ImagePreprocess import measure_blur


def show_bboxes(img, bboxes ,areas=''):
    """
    Show bboxes on images
    :param img: an image
    :param bboxes: dict of boxes, in  boundries format
    :param areas:
    :return: plot an image with bboxes
    """
    fig ,ax = plt.subplots(1 ,figsize=(10, 10))
    # Display the image
    ax.imshow(img)
    for i in bboxes:
        rect = patches.Rectangle(bboxes[i][0], bboxes[i][1][0]-bboxes[i][0][0], bboxes[i][1][1 ]-bboxes[i][0][1] ,linewidth=1,edgecolor='r',facecolor='none')
        ax.add_patch(rect)
    if areas:
        for i in areas:
            # xmin xmax ymin ymax
            h, w,_= img.shape
            print(areas[i], w ,h)
            xmin, xmax=np.array (areas[i][2:4])*w
            ymin, ymax=np.array ( areas[i][:2])*h
            rect = patches.Rectangle((xmin, ymin), xmax-xmin, ymax-ymin, linewidth=6, edgecolor='b', facecolor='none')
            ax.add_patch(rect)
    plt.show()

def get_digits_wrapper(frame, computervision_client):
    return get_digits(cv2.imencode(".jpg", frame)[1], computervision_client)

def eval(img_files, filters, result_list=[12, 500, 35, 0, 3, 21], ):
    """
    evaluate results, given img paths, filters and expected results list (unordered)
    :param img_files: list of image paths
    :param filters: list of filter functions (image as inpput and output
    :param result_list:
    :return: result dict
    """
    # for img_files list and filters list, evaluate results
    eval_dict = {fil.__name__: [] for fil in filters}
    for file in img_files:
        im = imread(file)
        for fil in filters:
            im = fil(im)
            results = [i[0] for i in get_digits_wrapper(im)]
            print(fil.__name__, len(np.intersect1d(results, result_list)))
            print('blur', measure_blur(im))
            eval_dict[fil.__name__].append(len(np.intersect1d(results, result_list)))
    return eval_dict