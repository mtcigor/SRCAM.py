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
MAPA_LABEL_NOME = 'label_map.pbtxt'

caminho = {
    'CHECKPOINT_CAMINHO': os.path.join('Tensorflow', 'workspace','models', NOME_MODELO),
    'ANOTACAO_CAMINHO': os.path.join('Tensorflow', 'workspace','annotations')
}
ficheiro = {
    'PIPELINE_CONFIG':os.path.join('Tensorflow', 'workspace','models', NOME_MODELO, 'pipeline.config'),
    'MAPALABEL': os.path.join(caminho['ANOTACAO_CAMINHO'], MAPA_LABEL_NOME) 
}
labels = [{'name':'license', 'id':1}]

#Mapa de classes
with open(ficheiro['MAPALABEL'], 'w') as f:
    for label in labels:
        f.write('item { \n')
        f.write('\tname:\'{}\'\n'.format(label['name']))
        f.write('\tid:{}\n'.format(label['id']))
        f.write('}\n')

# Carregar pieline e construir um modelo de detecção
configs = config_util.get_configs_from_pipeline_file(ficheiro['PIPELINE_CONFIG'])
modelo_detecao = model_builder.build(model_config=configs['model'], is_training=False)

# Restaurar checkpoint
ckpt = tf.compat.v2.train.Checkpoint(model=modelo_detecao)
ckpt.restore(os.path.join(caminho['CHECKPOINT_CAMINHO'], 'ckpt-11')).expect_partial()

# Função deteção
@tf.function
def detect_fn(imagem):
    imagem, formas = modelo_detecao.preprocess(imagem)
    predicacao = modelo_detecao.predict(imagem, formas)
    detecoes = modelo_detecao.postprocess(predicacao, formas)
    return detecoes

#Carregar as Classes
categoria = label_map_util.create_category_index_from_labelmap(ficheiro['MAPALABEL'])

# Função filtragem
def filtro_texto(regiao, ocr_resultado, regiao_limite):
    tamanho_retangulo = regiao.shape[0]*regiao.shape[1]
    
    caixa = [] 
    for resultado in ocr_resultado:
        comprimento = np.sum(np.subtract(resultado[0][1], resultado[0][0]))
        altura = np.sum(np.subtract(resultado[0][2], resultado[0][1]))
        
        if comprimento*altura / tamanho_retangulo > regiao_limite:
            caixa.append(resultado[1])
    return caixa

# Algoritimo de Levenshtein
def levenshtein_distancia(s1, s2):
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

# Encontrar na tabela a matrícula mais semelhante ou igual(limite 4)
def find_closest_value(input_str):
    conexao = mysql.connector.connect(
        host="10.1.31.46",
        user="root",
        password="aedas",
        database="parkit"
    )

    cursor = conexao.cursor()

    sql = "SELECT matricula FROM subscricao"
    cursor.execute(sql)

    valor_perto = None
    distancia_min = float('inf')

    for row in cursor.fetchall():
        valor = row[0]
        distancia = levenshtein_distancia(input_str, valor)
        if distancia <= 4 and distancia < distancia_min:
            valor_perto = valor
            distancia_min = distancia

    cursor.close()
    conexao.close()

    if distancia_min <= 4:
        return valor_perto
    else:
        return 0
    

def ocr_it(imagem, detecoes, detecao_limite, regiao_limite):
    
    # Pontuações, caixas e classes acima do limite
    pontos = list(filter(lambda x: x> detecao_limite, detecoes['detection_scores']))
    caixas = detecoes['detection_boxes'][:len(pontos)]
    classes = detecoes['detection_classes'][:len(pontos)]
    
    # Dimensões completas da imagem
    largura = imagem.shape[1]
    altura = imagem.shape[0]
    
    # Aplicar filtragem ROI(Regiao de Interese) e OCR(Reconhecimento ótico de caracteres)
    for idx, caixa in enumerate(caixas):
        roi = caixa*[altura, largura, altura, largura]
        regiao = imagem[int(roi[0]):int(roi[2]),int(roi[1]):int(roi[3])]
        leitor = easyocr.Reader(['en'])
        ocr_resultado = leitor.readtext(regiao)
        
        texto = filtro_texto(regiao, ocr_resultado, regiao_limite)
        
        plt.imshow(cv2.cvtColor(regiao, cv2.COLOR_BGR2RGB))
        plt.show()
        print(texto)
        return texto, regiao

detecao_limite = 0.1
regiao_limite = 0.1

conn = mysql.connector.connect(host = '10.1.31.46', user = 'root', password = 'aedas', database = 'parkit')
cursor = conn.cursor ()

#Captura em Direto
cap = cv2.VideoCapture(1)
largura = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
altura = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
tempos = 0

while cap.isOpened(): 
    ret, frame = cap.read()
    imagem_np = np.array(frame)
    
    input_tensor = tf.convert_to_tensor(np.expand_dims(imagem_np, 0), dtype=tf.float32)
    detecoes = detect_fn(input_tensor)
    
    num_detecoes = int(detecoes.pop('num_detections'))
    detecoes = {key: value[0, :num_detecoes].numpy()
                  for key, value in detecoes.items()}
    detecoes['num_detections'] = num_detecoes

    # detection_classes should be ints.
    detecoes['detection_classes'] = detecoes['detection_classes'].astype(np.int64)

    label_id_desvio = 1
    imagem_np_com_detecoes = imagem_np.copy()

    viz_utils.visualize_boxes_and_labels_on_image_array(
                imagem_np_com_detecoes,
                detecoes['detection_boxes'],
                detecoes['detection_classes']+label_id_desvio,
                detecoes['detection_scores'],
                categoria,
                use_normalized_coordinates=True,
                max_boxes_to_draw=5,
                min_score_thresh=.8,
                agnostic_mode=False)

    try:
        pontos = list(filter(lambda x: x> detecao_limite, detecoes['detection_scores']))
        if(pontos[0] >.8):
            texto, regiao = ocr_it(imagem_np_com_detecoes, detecoes, detecao_limite, regiao_limite)
            if(tempos == 0):
                tempos = tempos+1
                pass
            else:
                try:
                    texto_formatado = "".join(texto)
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
                tempos = 0
                pass
    except Exception as ex:
        # print("ERROR: " + str(ex))
        pass

    cv2.imshow('object detection',  cv2.resize(imagem_np_com_detecoes, (800, 600)))
    
    if cv2.waitKey(10) & 0xFF == ord('q'):
        cursor.close ()
        conn.close ()
        cap.release()
        cv2.destroyAllWindows()
        break

