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
CUSTOM_MODEL_NAME = 'my_ssd_mobnet'
LABEL_MAP_NAME = 'label_map.pbtxt'

paths = {
    'CHECKPOINT_PATH': os.path.join('Tensorflow', 'workspace','models', CUSTOM_MODEL_NAME),
    'ANNOTATION_PATH': os.path.join('Tensorflow', 'workspace','annotations')
}
files = {
    'PIPELINE_CONFIG':os.path.join('Tensorflow', 'workspace','models', CUSTOM_MODEL_NAME, 'pipeline.config'),
    'LABELMAP': os.path.join(paths['ANNOTATION_PATH'], LABEL_MAP_NAME) 
}
labels = [{'name':'license', 'id':1}]

#Mapa de classes
with open(files['LABELMAP'], 'w') as f:
    for label in labels:
        f.write('item { \n')
        f.write('\tname:\'{}\'\n'.format(label['name']))
        f.write('\tid:{}\n'.format(label['id']))
        f.write('}\n')

# Carregar pieline e construir um modelo de detecção
configs = config_util.get_configs_from_pipeline_file(files['PIPELINE_CONFIG'])
detection_model = model_builder.build(model_config=configs['model'], is_training=False)

# Restaurar checkpoint
ckpt = tf.compat.v2.train.Checkpoint(model=detection_model)
ckpt.restore(os.path.join(paths['CHECKPOINT_PATH'], 'ckpt-11')).expect_partial()

# Função deteção
@tf.function
def detect_fn(image):
    image, shapes = detection_model.preprocess(image)
    prediction_dict = detection_model.predict(image, shapes)
    detections = detection_model.postprocess(prediction_dict, shapes)
    return detections

#Carregar as Classes
category_index = label_map_util.create_category_index_from_labelmap(files['LABELMAP'])

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

def levenshtein_distance(s1, s2):
    m = len(s1)
    n = len(s2)

    dp = [[0] * (n + 1) for _ in range(m + 1)]

    for i in range(m + 1):
        dp[i][0] = i
    for j in range(n + 1):
        dp[0][j] = j

    for i in range(1, m + 1):
        for j in range(1, n + 1):
            cost = 0 if s1[i - 1] == s2[j - 1] else 1
            dp[i][j] = min(
                dp[i - 1][j] + 1,         
                dp[i][j - 1] + 1,         
                dp[i - 1][j - 1] + cost   
            )

    return dp[m][n]

# Encontrar na tabela a matrícula mais semelhante ou igual(limite 3)
def find_closest_value(input_str):
    connection = mysql.connector.connect(
        host="10.1.31.46",
        user="root",
        password="aedas",
        database="parkit"
    )

    cursor = connection.cursor()

    query = "SELECT matricula FROM subscricao"
    cursor.execute(query)

    closest_value = None
    min_distance = float('inf')

    for row in cursor.fetchall():
        value = row[0]
        distance = levenshtein_distance(input_str, value)
        if distance <= 3 and distance < min_distance:
            closest_value = value
            min_distance = distance

    cursor.close()
    connection.close()

    if min_distance <= 3:
        return closest_value
    else:
        return 0
    



def ocr_it(image, detections, detection_threshold, region_threshold):
    
    # Scores, boxes and classes above threhold
    scores = list(filter(lambda x: x> detection_threshold, detections['detection_scores']))
    boxes = detections['detection_boxes'][:len(scores)]
    classes = detections['detection_classes'][:len(scores)]
    
    # Full image dimensions
    width = image.shape[1]
    height = image.shape[0]
    
    # Apply ROI filtering and OCR
    for idx, box in enumerate(boxes):
        roi = box*[height, width, height, width]
        region = image[int(roi[0]):int(roi[2]),int(roi[1]):int(roi[3])]
        reader = easyocr.Reader(['en'])
        ocr_result = reader.readtext(region)
        
        text = filter_text(region, ocr_result, region_threshold)
        
        plt.imshow(cv2.cvtColor(region, cv2.COLOR_BGR2RGB))
        plt.show()
        print(text)
        return text, region

# Salvar Resultados em CSV e Pasta com Imagens
def save_results(text, region, csv_filename, folder_path):
    img_name = '{}.jpg'.format(uuid.uuid1())
    
    cv2.imwrite(os.path.join(folder_path, img_name), region)
    
    with open(csv_filename, mode='a', newline='') as f:
        csv_writer = csv.writer(f, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        csv_writer.writerow([img_name, text])

detection_threshold = 0.1
region_threshold = 0.1

conn = mysql.connector.connect(host = '10.1.31.46', user = 'root', password = 'aedas', database = 'parkit')
cursor = conn.cursor ()

#Captura em Direto
cap = cv2.VideoCapture(1)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
times = 0

while cap.isOpened(): 
    ret, frame = cap.read()
    image_np = np.array(frame)
    
    input_tensor = tf.convert_to_tensor(np.expand_dims(image_np, 0), dtype=tf.float32)
    detections = detect_fn(input_tensor)
    
    num_detections = int(detections.pop('num_detections'))
    detections = {key: value[0, :num_detections].numpy()
                  for key, value in detections.items()}
    detections['num_detections'] = num_detections

    # detection_classes should be ints.
    detections['detection_classes'] = detections['detection_classes'].astype(np.int64)

    label_id_offset = 1
    image_np_with_detections = image_np.copy()

    viz_utils.visualize_boxes_and_labels_on_image_array(
                image_np_with_detections,
                detections['detection_boxes'],
                detections['detection_classes']+label_id_offset,
                detections['detection_scores'],
                category_index,
                use_normalized_coordinates=True,
                max_boxes_to_draw=5,
                min_score_thresh=.8,
                agnostic_mode=False)

    try:
        scores = list(filter(lambda x: x> detection_threshold, detections['detection_scores']))
        if(scores[0] >.8):
            text, region = ocr_it(image_np_with_detections, detections, detection_threshold, region_threshold)
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
                        matricula = find_closest_value(texto_limpo)
                        if(matricula == 0):
                            sql = "INSERT INTO matriculas (matricula, dataentrada, pagou) VALUES (%s, %s, %s)"
                            params = (texto_limpo, data_hora_formatada, 0)
                            cursor.execute (sql, params)
                            conn.commit()
                        else:
                            sql = "INSERT INTO matriculas (matricula, dataentrada, pagou) VALUES (%s, %s, %s)"
                            params = (matricula, data_hora_formatada, 1)
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

    cv2.imshow('object detection',  cv2.resize(image_np_with_detections, (800, 600)))
    
    if cv2.waitKey(10) & 0xFF == ord('q'):
        cursor.close ()
        conn.close ()
        cap.release()
        cv2.destroyAllWindows()
        break

