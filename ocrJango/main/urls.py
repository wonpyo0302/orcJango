from django.urls import path
from . import views

urlpatterns = [
    path('', views.helloworld, name='helloworld'),
    path('checkdata', views.checkdata, name='checkdata'),
]