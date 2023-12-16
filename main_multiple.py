import pandas as pd
import numpy as np
from glob import glob
import streamlit as st
from PIL import Image
import cv2

from ultralytics import YOLO
from paddleocr import PaddleOCR

PLATE_NUM = 2

# Carrega o modelo do Yolov8
model = YOLO('../best_v8.pt')  # pretrained YOLOv8n model

# OCR from LearnOpenCV
ocr = PaddleOCR(use_angle_cls=True, lang='en')


def get_tx(tx):
    try:
        x = int(tx)
        return x if len(tx) >= PLATE_NUM else []
    except:
        tx = []
    return tx


def get_nums(box_texts, detecs):

    if detecs == 0:
        resps = ['ni']
    else:
        resps = ['ni' for x in range(detecs)]

    for n, tx in enumerate(box_texts):
        text = get_tx(tx)
        resps[n] = text if text else 'ni'
    return resps


def check_list(txts):
    lists = []
    for txt in txts:
        if txt.isdigit():
            lists.append(txt)
    if lists:
        return max(lists, key = len)
    return []


def return_detections(img_p):
    # Passa pelo modelo
    out = model.predict(img_p, imgsz=800, conf=0.5)
    box_texts = []
    detecs = 0

    for o in out:
        boxes = o.boxes
        for bs in boxes.xyxy:
            x1, y1, x2, y2 = bs
            detecs +=1

            i_res = cv2.resize(img_p[int(y1):int(y2), int(x1):int(x2), :], (240, 240))
            # i_res = 
            result = ocr.ocr(i_res, cls=True)
            txts = [line[1][0] for line in result]

            txts = check_list(txts)
            box_texts.append(txts if txts else [])

    resps = get_nums(box_texts, detecs)

    return resps


st.title("Identificador de números de corrida")
st.header("Faça o upload de sua imagem")

result_list = []
with st.form("Arquivos", clear_on_submit=True):
        file = st.file_uploader('', type=['jpeg', 'jpg', 'png'], accept_multiple_files=True)
        submitted = st.form_submit_button("Upload")

if submitted and file is not None:
    st.write("Imagens adicionadas: {}".format(len(file)))
    for i in range(len(file)):
        print(file[i])
        # Read image
        image = Image.open(file[i])

        # Convert to OpenCV image
        img_p = np.array(image)
        box_texts = return_detections(img_p)
        box_texts.insert(0, file[i].name)

        result_list.append(box_texts)

print(result_list)
if result_list:
    df = pd.DataFrame(result_list)
    csv = df.to_csv(index=False).encode('utf-8')

    st.download_button(
        "Press to Download",
        csv,
        "file.csv",
        "text/csv",
        key='download-csv'
    )