#Bibliotecas
import os
import tensorflow as tf
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as viz_utils
from object_detection.builders import model_builder
from object_detection.utils import config_util
import cv2 
import numpy as np
from matplotlib import pyplot as plt
import easyocr
import csv
import uuid
import re
import datetime
import mysql.connector

#Caminhos dos Ficheiros
NOME_MODELO = 'my_ssd_mobnet'
LABEL_MAPA = 'label_map.pbtxt'

caminho = {
    'CHECKPOINT_PATH': os.path.join('Tensorflow', 'workspace','models', NOME_MODELO),
    'ANOTACAO_PATH': os.path.join('Tensorflow', 'workspace','annotations')
}
ficheiros = {
    'PIPELINE_CONFIG':os.path.join('Tensorflow', 'workspace','models', NOME_MODELO, 'pipeline.config'),
    'LABELMAP': os.path.join(caminho['ANOTACAO_PATH'], LABEL_MAPA) 
}
labels = [{'name':'license', 'id':1}]

#Mapa de classes
with open(ficheiros['LABELMAP'], 'w') as f:
    for label in labels:
        f.write('item { \n')
        f.write('\tname:\'{}\'\n'.format(label['name']))
        f.write('\tid:{}\n'.format(label['id']))
        f.write('}\n')

# Carregar pieline e construir um modelo de detecção
configs = config_util.get_configs_from_pipeline_file(ficheiros['PIPELINE_CONFIG'])
modelo_detecao = model_builder.build(model_config=configs['model'], is_training=False)

# Restaurar checkpoint
ckpt = tf.compat.v2.train.Checkpoint(model=modelo_detecao)
ckpt.restore(os.path.join(caminho['CHECKPOINT_PATH'], 'ckpt-11')).expect_partial()

# Função deteção
@tf.function
def detetar_fn(imagem):
    imagem, formas = modelo_detecao.preprocess(imagem)
    predicacao = modelo_detecao.predict(imagem, formas)
    detecoes = modelo_detecao.postprocess(predicacao, formas)
    return detecoes

#Carregar as Classes
categoria_index = label_map_util.create_category_index_from_labelmap(ficheiros['LABELMAP'])

# Função filtragem
def filter_text(region, ocr_result, region_threshold):
    rectangle_size = region.shape[0]*region.shape[1]
    
    plate = [] 
    for result in ocr_result:
        length = np.sum(np.subtract(result[0][1], result[0][0]))
        height = np.sum(np.subtract(result[0][2], result[0][1]))
        
        if length*height / rectangle_size > region_threshold:
            plate.append(result[1])
    return plate

def ocr_it(imagem, detecoes, detection_threshold, region_threshold):
    
    # Scores, boxes and classes above threhold
    scores = list(filter(lambda x: x> detection_threshold, detecoes['detection_scores']))
    boxes = detecoes['detection_boxes'][:len(scores)]
    classes = detecoes['detection_classes'][:len(scores)]
    
    # Full imagem dimensions
    width = imagem.shape[1]
    height = imagem.shape[0]
    
    # Apply ROI filtering and OCR
    for idx, box in enumerate(boxes):
        roi = box*[height, width, height, width]
        region = imagem[int(roi[0]):int(roi[2]),int(roi[1]):int(roi[3])]
        reader = easyocr.Reader(['en'])
        ocr_result = reader.readtext(region)
        
        text = filter_text(region, ocr_result, region_threshold)
        
        plt.imshow(cv2.cvtColor(region, cv2.COLOR_BGR2RGB))
        plt.show()
        print(text)
        return text, region

# Salvar Resultados em CSV e Pasta com imagemns
def save_results(text, region, csv_filename, folder_path):
    img_name = '{}.jpg'.format(uuid.uuid1())
    
    cv2.imwrite(os.path.join(folder_path, img_name), region)
    
    with open(csv_filename, mode='a', newline='') as f:
        csv_writer = csv.writer(f, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        csv_writer.writerow([img_name, text])

detection_threshold = 0.1
region_threshold = 0.1

conn = mysql.connector.connect(host = 'localhost', user = 'root', password = 'aedas', database = 'parkit')
cursor = conn.cursor ()

#Captura em Direto
cap = cv2.VideoCapture(1)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
times = 0

while cap.isOpened(): 
    ret, frame = cap.read()
    imagem_np = np.array(frame)
    
    input_tensor = tf.convert_to_tensor(np.expand_dims(imagem_np, 0), dtype=tf.float32)
    detecoes = detetar_fn(input_tensor)
    
    num_detecoes = int(detecoes.pop('num_detections'))
    detecoes = {key: value[0, :num_detecoes].numpy()
                  for key, value in detecoes.items()}
    detecoes['num_detections'] = num_detecoes

    # detection_classes should be ints.
    detecoes['detection_classes'] = detecoes['detection_classes'].astype(np.int64)

    label_id_offset = 1
    imagem_np_with_detections = imagem_np.copy()

    viz_utils.visualize_boxes_and_labels_on_imagem_array(
                imagem_np_with_detections,
                detecoes['detection_boxes'],
                detecoes['detection_classes']+label_id_offset,
                detecoes['detection_scores'],
                categoria_index,
                use_normalized_coordinates=True,
                max_boxes_to_draw=5,
                min_score_thresh=.8,
                agnostic_mode=False)
    
    try:
        scores = list(filter(lambda x: x> detection_threshold, detecoes['detection_scores']))
        if(scores[0] >.8):
            text, region = ocr_it(imagem_np_with_detections, detecoes, detection_threshold, region_threshold)
            if(times == 0):
                times = times+1
                pass
            else:
                try:
                    texto_formatado = "".join(text)
                    if(texto_formatado == ""):
                        pass
                    else:
                        texto_limpo = re.sub(r'\W+', '', texto_formatado).upper()
                        data_hora = datetime.datetime.now ()
                        data_hora_formatada = data_hora.strftime ('%Y-%m-%d %H:%M:%S')
                        sql = "INSERT INTO matriculas (matricula, dataentrada, pagou) VALUES (%s, %s, %s)"
                        params = (texto_limpo, data_hora_formatada, 0)
                        cursor.execute (sql, params)
                        conn.commit()
                        passagem = input("Matrícula Gravada, Passagem completa? (Y): ")
                        while(passagem != "Y"):
                            passagem = input("Passagem completa? (Y): ")
                except Exception as ex:
                    print("ERROR: " + str(ex))
                    pass 
                times = 0
                pass
    except Exception as ex:
        # print("ERROR: " + str(ex))
        pass

    cv2.imshow('object detection',  cv2.resize(imagem_np_with_detections, (800, 600)))
    
    if cv2.waitKey(10) & 0xFF == ord('q'):
        cursor.close ()
        conn.close ()
        cap.release()
        cv2.destroyAllWindows()
        break

