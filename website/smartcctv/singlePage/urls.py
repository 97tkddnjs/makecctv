from django.urls import path, include
from singlePage import views

urlpatterns = [
    path('', views.landing),
]