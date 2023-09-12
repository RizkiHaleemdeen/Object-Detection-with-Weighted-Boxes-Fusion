# Object Detection
from gluoncv import model_zoo, data, utils
from matplotlib import pyplot as plt
import numpy as np
from numpy import asarray
from ensemble_boxes import weighted_boxes_fusion
import os
from PIL import Image
import streamlit as st
import skimage as ski
import math
from tempfile import NamedTemporaryFile
import mxnet as mx
from process import normalize_list

st.title("Object Detection Model")

def get_results_yolo(x_yolo):      
    # Detect Using YOLO
    class_ids_yolo, scores_yolo, bboxs_yolo = yolov3(x_yolo)
    st.write('YoloV3 Prediction') 
    ax_yolo = utils.viz.plot_bbox(img_yolo, bboxs_yolo[0], scores_yolo[0],class_ids_yolo[0], class_names=yolov3.classes) 
    plt.axis('off')
    st.pyplot(plt.gcf())
    return class_ids_yolo, scores_yolo, bboxs_yolo
    
def get_results_frcnn(x_frcnn): 
    # Detect Using FRCNN
    class_ids_frcnn, scores_frcnn, bboxes_frcnn = frcnn(x_frcnn)
    st.write('FRCNN Prediction')  
    ax_frcnn = utils.viz.plot_bbox(img_frcnn, bboxes_frcnn[0], scores_frcnn[0], class_ids_frcnn[0], class_names=frcnn.classes)
    plt.axis('off')
    st.pyplot(plt.gcf())
    return class_ids_frcnn, scores_frcnn, bboxes_frcnn

def normalize_for_weight(x_yolo,x_frcnn):
    class_ids_yolo, scores_yolo, bboxs_yolo=get_results_yolo(x_yolo)
    class_ids_frcnn, scores_frcnn, bboxes_frcnn=get_results_frcnn(x_frcnn)

    B1 = normalize_list(bboxs_yolo[0])
    B2 = normalize_list(bboxes_frcnn[0])
    S1 = normalize_list(scores_yolo[0])
    S2 = normalize_list(scores_frcnn[0])
    C1=class_ids_yolo[0].asnumpy().tolist()
    C2=class_ids_frcnn[0].asnumpy().tolist()
    
    # Weighted Boxes Fusion
    boxes_list = [B1,B2]
    scores_list = [np.array(S1).flatten(), np.array(S2).flatten()]
    labels_list = [np.array(C1).flatten(),np.array(C2).flatten()]
    
    return boxes_list,scores_list,labels_list


yolov3 = model_zoo.get_model('yolo3_darknet53_voc', pretrained=True)
frcnn = model_zoo.get_model('faster_rcnn_resnet50_v1b_voc', pretrained=True)


im_fname = st.file_uploader("UPLOAD AN IMAGE HERE TO START OBJECT PREDICTION", type=['png', 'jpeg', 'jpg'])

prompt=None
if im_fname is not None:
    Img=Image.open(im_fname)
    st.write('Orginal Image')
    st.image(Img)
    x_yolo, img_yolo = data.transforms.presets.yolo.transform_test(mx.nd.array(Img))
    x_frcnn, img_frcnn = data.transforms.presets.rcnn.transform_test(mx.nd.array(Img))

# col1, col2 = st.columns(2) 
    boxes_list,scores_list,labels_list=normalize_for_weight(x_yolo,x_frcnn)

    weights = None

    iou_thr = 0.5
    skip_box_thr = 0.0001
    sigma = 0.1

    boxes, scores, labels = weighted_boxes_fusion(boxes_list, scores_list, labels_list, weights=weights, iou_thr=iou_thr, skip_box_thr=skip_box_thr)

    # Post Processing
    FINAL_CONF_THRESH=1e-3
    res = img_yolo.shape

    # [W, H, W, H]
    res_array = np.array([res[1], res[0], res[1], res[0]])
    
    # Filter out boxes having scores < FINAL_CONF_THRESH
    final_scores_ids = np.where(scores > FINAL_CONF_THRESH)[0]
    
    final_boxes = boxes[final_scores_ids]
    final_scores = scores[final_scores_ids]

    # De Normalize Box coordinates in [xmin, ymin, xmax, ymax]
    final_boxes = (final_boxes*res_array).clip(min=[0.,0.,0.,0.],
                                                    max=[res[1]-1, res[0]-1, res[1]-1, res[0]-1])
    
    final_boxes = final_boxes.astype("int")
    final_boxes[:,2:] = final_boxes[:,2:] - final_boxes[:, :2]
    final_boxes=final_boxes.tolist()

    ax_weightedfusion = utils.viz.plot_bbox(img_yolo, final_boxes, final_scores,labels, class_names=frcnn.classes)

    st.write('Weighted Boxes Fusion Prediction')  
    plt.axis('off')
    st.pyplot(plt.gcf())
        

if st.button('Mask Objects'):
    if im_fname is None:
        st.write('Error: Upload an Image first')
    else:
        class_ids_frcnn, scores_frcnn, bboxes_frcnn=get_results_frcnn(x_frcnn)
        # Finding the predictions which has a score more than 0.9
        S2_new = np.array(normalize_list(scores_frcnn[0])).flatten()
        frcnn_ids=np.where(S2_new>0.9)

        masking_bboxes=bboxes_frcnn[0][frcnn_ids]
        masking_bboxes=masking_bboxes.asnumpy()

        # create the mask image
        imgobjects=[]
        vals=img_frcnn.shape[0:2]

        for line in masking_bboxes:
            # line=masking_bboxes[0]
            xmin=line[0]
            xmax=line[2]
            if xmax>vals[1]:
                xmax=vals[1]-1
            ymin=line[1]
            ymax=line[3]
            if ymax>vals[0]:
                ymax=vals[0]-1

            rr, cc = ski.draw.rectangle(start=(ymin,xmin), end=(ymax,xmax))
            masktemp=np.ones(shape=img_frcnn.shape[0:2], dtype="bool")

            imgtemp=img_frcnn.copy()
            masktemp[rr, cc] = False

            imgtemp[masktemp] = 0
            imgobjects.append(imgtemp)

        # display the result
        ids=np.array(class_ids_frcnn[0][frcnn_ids].asnumpy().tolist()).flatten()
        n=len(imgobjects)
        image_size = (64, 64)

        #Create Grid
        rows = 2 
        cols = (n + 1) // rows 

        # # Create a new figure
        # plt.figure(figsize=(10, 6))

        #Plot each image
        for i in range(n):
            plt.subplot(rows, cols, i + 1)
            plt.imshow(imgobjects[i])
            plt.axis('off')
            plt.title(f'{frcnn.classes[int(ids[i])]}')

        # Adjust spacing between subplots
        plt.tight_layout()

        # Show the plot
        st.write("Masked objects of FRCNN Model predicted objects")
        st.pyplot(plt.gcf())
