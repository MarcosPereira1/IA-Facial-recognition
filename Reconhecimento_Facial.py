
#  importando a Biblioteca de Visão Computacional (OpenCV)
import cv2 

#  O metodo cascadeClassifer classifica as imagens positivas e negativas
#  Algortimo  que seleciona características visuais da face do rosto e as utiliza para a etapa de detecção
classificador = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")  

# Usamos o cv2.face.LBHFaceRecognizer_create para treinar nosso reconhecedor de rosto no conjunto de dados datbase
reconhecedor = cv2.face.LBPHFaceRecognizer_create() 

reconhecedor.read('classificadorLBPH_V1.yml')

 # INICIAR CAMERA
camera = cv2.VideoCapture(0)

while True:
    conectado, imagem = camera.read()
    imagemCinza = cv2.cvtColor(imagem, cv2.COLOR_BGR2GRAY)
    faceDetectadas = classificador.detectMultiScale(imagemCinza, scaleFactor=1.5)

    for (x, y, l, a) in faceDetectadas:
        imagemFace = cv2.resize(imagemCinza[y:y + a, x:x + l], (100, 100))
        cv2.rectangle(imagem, (x, y), (x + l, y + a), (0, 0, 255), 2)
        id, confianca = reconhecedor.predict(imagemFace)

        if id == 1:
            nome = "Marcos Vinicius"
        elif id == 2:
            nome = "Juan Carlos"
        elif id == 3:
            nome = "Mauricio Araujo"
        elif id == 4:
            nome = "Ana Paula"


        cv2.putText(imagem, nome, (x,y + (a + 30)), cv2.FONT_HERSHEY_COMPLEX_SMALL, 2, (0, 0, 255))

    cv2.imshow("Face", imagem)
    if cv2.waitKey(1) == ord('q'):
        break

camera.release()
cv2.destroyAllWindows()