
import cv2, os
import numpy as np
from PIL import Image
from surya.detection import batch_text_detection
from surya.layout import batch_layout_detection
from surya.model.detection.segformer import load_model, load_processor
from surya.settings import settings
from surya.model.recognition.model import load_model as rec_load_model
from surya.model.recognition.processor import load_processor as rec_load_processor
from surya.ocr import run_ocr
from django.http import JsonResponse

class Scan:
    def __init__(self, oriImg) :
        self.oriImg = oriImg


    def adjust(self, img):
        # # 이미지 전처리 및 외곽선 추출
        edged = self.extractEdge(img)
        
        contours, hierarchy = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        draw = img.copy()
        cv2.drawContours(draw,  contours, -1, (0, 255, 0))
        
        # 사각형 중 최대크기의 컨투어 꼭지점
        pts = self.getPointsOfMaxRectangle(contours)

        # 각각의 좌표 찾기
        sumXY = pts.sum(axis=1)
        diff = np.diff(pts, axis=1)

        topLeft = pts[np.argmin(sumXY)]
        bottomRight = pts[np.argmax(sumXY)]
        topRight = pts[np.argmin(diff)]
        bottomLeft = pts[np.argmax(diff)]

        # 사진을 변환할 때 사용할 서류의 높이
        widthTop = abs(topRight[0] - topLeft[0])
        widthBottom = abs(bottomRight[0] - bottomLeft[0])
        heightRight = abs(topRight[1] - bottomRight[1])
        heightLeft = abs(topLeft[1] - bottomLeft[1])
        print(widthBottom, widthTop, heightLeft, heightRight)

        width = max([widthTop, widthBottom])
        height = max([heightRight, heightLeft])

        pts1 = np.float32([topLeft, topRight, bottomRight, bottomLeft])
        pts2 = np.float32([[0,0], [width, 0], [width, height], [0, height]])

        matrix = cv2.getPerspectiveTransform(pts1, pts2) # 좌표를 변환하기 위해 사용할 변환행렬
        result = cv2.warpPerspective(img, matrix, (width, height)) # 이미지 변환(변환행렬 적용)

        return result
    
    def extractEdge(self, img):
        # 이미지 전처리 및 외곽선 추출
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (3, 3), 0) # 이미지를 흐리게 처리함 (noise 제거를 위해 사용)
        edged = cv2.Canny(gray, 75, 250) # edged를 검출하는 함수 (img, minVal, maxVal)
        return edged
    
    def getPointsOfMaxRectangle(self, contours) :
        # 크기순으로 컨투어 정렬
        contours = sorted(contours, key=cv2.contourArea, reverse=True)[:5]
        for c in contours:
            peri = cv2.arcLength(c, True) # 외곽선 길이
            # print(peri)
            verticles = cv2.approxPolyDP(c, 0.02 * peri, closed=True) # 외곽선 근사화
            if len(verticles) == 4 : 
                break
        pts = verticles.reshape(4, 2) # 배열을 4 * 2 크기로 조정
        return pts
    
def ocrProcess(img) : 

    # 텍스트 영역 추출 및 레이아웃 분석하기
    image = Image.open("./image/result.jpg")
    model = load_model(checkpoint=settings.LAYOUT_MODEL_CHECKPOINT)
    processor = load_processor(checkpoint=settings.LAYOUT_MODEL_CHECKPOINT)
    det_model = load_model()
    det_processor = load_processor()

    # layout_predictions is a list of dicts, one per image

    line_predictions = batch_text_detection([image], det_model, det_processor)
    layout_predictions = batch_layout_detection([image], model, processor, line_predictions)

    bboxArr = []
    for b in layout_predictions[0].bboxes:
        # print(b.bbox)
        bboxArr.append(b.bbox)

    # 레이아웃에 따라 읽는 순서 분석하기
    from surya.ordering import batch_ordering
    from surya.model.ordering.processor import load_processor as order_load_processor
    from surya.model.ordering.model import load_model as order_load_model

    ord_model = order_load_model()
    ord_processor = order_load_processor()

    order_predictions = batch_ordering([image], [bboxArr], ord_model, ord_processor)

    # 분석한 레이아웃에 따라 정렬하기
    order_predictions[0].bboxes.sort(key=lambda e : e.position)
    order_predictions[0].bboxes

    # bbox에 따라 이미지 잘라내기
    layout_img = cv2.imread("./image/result.jpg")
    directory = "./image/fragments/"
    cnt = 0
    for b in order_predictions[0].bboxes:
        start = (int(b.bbox[0]), int(b.bbox[1]))
        end = (int(b.bbox[2]), int(b.bbox[3]))
        # print(start, end)
        cv2.rectangle(layout_img, start, end, (255, 255, 255), 2)

        # 특정 폴더에 이미지 저장하기
        directory = "./image/fragments/"
        filenm = directory + "result_"+str(cnt) + ".jpg"
        
        cv2.imwrite(filenm, layout_img[int(b.bbox[1]):int(b.bbox[3]), int(b.bbox[0]):int(b.bbox[2])]) # 세로, 가로
        cnt = cnt + 1
        # ordering 숫자 추가하기
        # print(b.position)
        # cv2.putText(layout_img, str(b.position), start, cv2.FONT_HERSHEY_SIMPLEX, 2, 2, 2)

    # OCR 진행
    langs = ["en"] # Replace with your languages
    det_processor, det_model = load_processor(), load_model()
    rec_model, rec_processor = rec_load_model(), rec_load_processor()
    # OCR 진행
    predictions = []
    directory = "./image/fragments/"
    fileList = os.listdir(directory)
    for f in fileList : 
        tmpImg = Image.open(directory  + f)
        prediction = run_ocr([tmpImg], [langs], det_model, det_processor, rec_model, rec_processor)
        predictions.append(prediction)

    return layout_img, predictions


def getTextFromPredictions(predictions) : 
    text = ""
    for p in predictions:
        for lines in p[0].text_lines:
            text = text + " " + lines.text
        text = text + "\n"
    return text
