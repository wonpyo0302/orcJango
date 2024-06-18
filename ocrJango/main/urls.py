from django.urls import path
from . import views

urlpatterns = [
    path('', views.helloworld, name='helloworld'),
    path('processCutImg', views.processCutImg, name='processCutImg'),
    path('processOcrImg', views.processOcrImg, name='processOcrImg'),
]