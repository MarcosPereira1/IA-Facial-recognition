import cv2
import os
import numpy as np

lbph = cv2.face.LBPHFaceRecognizer_create()

def getImagemComId():
    caminhos = [os.path.join('Fotos', f) for f in os.listdir('Fotos')]
    faces = []
    ids = []
    for caminhosImagem in caminhos:
        imagemFace = cv2.cvtColor(cv2.imread(caminhosImagem), cv2.COLOR_BGR2GRAY)
        id = int(os.path.split(caminhosImagem)[-1].split('.')[1])
        ids.append(id)
        faces.append(imagemFace)
    return np.array(ids), faces

ids, faces = getImagemComId()

print("Treinando....")

lbph.train(faces, ids)
lbph.write('classificadorLBPH_V1.yml')

print(" O Treinamento concluído ...")
print(" Executa lá o reconhecimento ;) ")