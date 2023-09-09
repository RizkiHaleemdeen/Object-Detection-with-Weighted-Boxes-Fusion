# Object Detection
from gluoncv import model_zoo, data, utils
from matplotlib import pyplot as plt
import numpy as np
from ensemble_boxes import weighted_boxes_fusion
import os
from PIL import Image
import streamlit as st

st.title("Object Detection Model")

#Pre Processing to Use weighted boxes fusion ensembling
def normalize_list(pass_list):
    all_values = [val for sublist in pass_list for val in sublist]
    min_val = min(all_values)
    max_val = max(all_values)

    # Normalize the elements within each sublist
    normalized_list = [[(x - min_val) / (max_val - min_val) for x in sublist] for sublist in pass_list]
    return normalized_list

yolov3 = model_zoo.get_model('yolo3_darknet53_voc', pretrained=True)
frcnn = model_zoo.get_model('faster_rcnn_resnet50_v1b_voc', pretrained=True)

# im_fname="test-samples/dog.jpg"
im_fname = st.file_uploader("Upload an image here.", type=['png', 'jpeg', 'jpg'])
prompt=None
col1, col2 = st.columns(2)
out=""
if im_fname is not None:
    Img=Image.open(im_fname)
    st.write('Orginal Image')
    st.image(Img)

if st.button('Test Models for Object Detection'):      
    # Detect Using YOLO
    x_yolo, img_yolo = data.transforms.presets.yolo.load_test(im_fname.name)
    class_ids_yolo, scores_yolo, bboxs_yolo = yolov3(x_yolo)
    ax_yolo = utils.viz.plot_bbox(img_yolo, bboxs_yolo[0], scores_yolo[0],class_ids_yolo[0], class_names=yolov3.classes)
    st.write('YoloV3 Prediction')  
    st.pyplot(plt.gcf())

    # Detect Using FRCNN
    x_frcnn, img_frcnn = data.transforms.presets.rcnn.load_test(im_fname.name)
    class_ids_frcnn, scores_frcnn, bboxes_frcnn = frcnn(x_frcnn)
    ax_frcnn = utils.viz.plot_bbox(img_frcnn, bboxes_frcnn[0], scores_frcnn[0], class_ids_frcnn[0], class_names=frcnn.classes)
    st.write('FRCNN Prediction')  
    st.pyplot(plt.gcf())

    B1=bboxs_yolo[0].asnumpy().tolist()
    B1 = normalize_list(B1)
    B2=bboxes_frcnn[0].asnumpy().tolist()
    B2 = normalize_list(B2)
    S1=scores_yolo[0].asnumpy().tolist()
    S1 = normalize_list(S1)
    S2=scores_frcnn[0].asnumpy().tolist()
    S2 = normalize_list(S2)
    C1=class_ids_yolo[0].asnumpy().tolist()
    C2=class_ids_frcnn[0].asnumpy().tolist()

    # Weighted Boxes Fusion
    boxes_list = [B1,B2]
    scores_list = [np.array(S1).flatten(), np.array(S2).flatten()]
    labels_list = [np.array(C1).flatten(),np.array(C2).flatten()]

    weights = None

    iou_thr = 0.6
    skip_box_thr = 0.0001
    sigma = 0.1

    boxes, scores, labels = weighted_boxes_fusion(boxes_list, scores_list, labels_list, weights=weights, iou_thr=iou_thr, skip_box_thr=skip_box_thr)

    # Post Processing
    FINAL_CONF_THRESH=1e-3
    res = Img.size

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
    st.pyplot(plt.gcf())