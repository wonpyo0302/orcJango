from django.shortcuts import render
from django.http import HttpResponse
from django.views.decorators.csrf import csrf_exempt
from django.core.files.storage import FileSystemStorage


# Create your views here.
def helloworld(request):
    return render(request, 'ocrJango/helloworld.html')

@csrf_exempt
def processCutImg(request):
    
    if request.method == 'POST' and request.FILES['file']:
        myfile = request.FILES['file']
        '''
        print('myfile read:', myfile.read()) # file 읽기
        print('myfile size:', myfile.size) # file 읽기
        print('myfile content_type:', myfile.content_type)
        print('myfile open:', myfile.open())
        myfile_read = myfile.read()
        print('myfile read type:', type(myfile_read))
        '''
        fs = FileSystemStorage(location='image', base_url='image')
        filename = fs.save(myfile.name, myfile)
        uploaded_file_url = fs.url(filename)
        return render(request, 'ocrJango/helloworld.html', {
            'uploaded_file_url': uploaded_file_url
        })
        
    return render(request, 'ocrJango/helloworld.html')
