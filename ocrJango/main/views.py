from django.shortcuts import render
from django.http import HttpResponse
from django.views.decorators.csrf import csrf_exempt
from django.core.files.storage import FileSystemStorage
from rest_framework.response import Response
from rest_framework.decorators import api_view

# 모듈 경로 지정
import sys, os
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)
import ocr

# Create your views here.
def helloworld(request):
    return render(request, 'ocrJango/helloworld.html')

@csrf_exempt
def processCutImg(request):
    
    if request.method == 'POST' and request.FILES['file']:
        # input 파일 읽기
        myfile = request.FILES['file']
        '''
        print('myfile read:', myfile.read()) # file 읽기
        print('myfile size:', myfile.size) # file 읽기
        print('myfile content_type:', myfile.content_type)
        print('myfile open:', myfile.open())
        myfile_read = myfile.read()
        print('myfile read type:', type(myfile_read))
        '''
        # 원본파일 저장
        fs = FileSystemStorage(location='image', base_url='image')
        filename = fs.save('result.jpg', myfile)
        uploaded_file_url = fs.url(filename)
        image_path = 'image/' + filename
        
        # 이미지 파일 자르고 저장
        img = ocr.cv2.imread(image_path)
        scanner = ocr.Scan(img)
        myfile = scanner.adjust(img)
        ocr.cv2.imwrite(image_path, myfile)

        # 파일 읽기
        with open(image_path, 'rb') as image_file:
            image_data = image_file.read()

        response = HttpResponse(image_data, content_type='image/jpeg')
        response['Content-Disposition'] = f'inline; filename="{filename}"'
        return response
        
    return render(request, 'ocrJango/helloworld.html')

@api_view(['post'])
def processOcrImg(request):
    image_path = 'image/result.jpg'
    img = ocr.cv2.imread(image_path)
    myfile, predictions = ocr.ocrProcess(img)
    content = ocr.getTextFromPredictions(predictions)
    return  Response(data={
        "data": content
    })
