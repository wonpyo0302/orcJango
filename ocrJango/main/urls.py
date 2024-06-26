from django.urls import path
from . import views

urlpatterns = [
    path('', views.helloworld, name='helloworld'),
<<<<<<< HEAD
    path('checkdata', views.checkdata, name='checkdata'),
=======
    path('processCutImg', views.processCutImg, name='processCutImg'),
    path('processOcrImg', views.processOcrImg, name='processOcrImg'),
>>>>>>> jpan/v1
]