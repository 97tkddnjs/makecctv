from django.urls import path
from .import views
urlpatterns = [
    path('', views.index, name='index'),
    path('video_stream', views.video_stream, name='video_stream'),
    path('video_stream1', views.video_stream1, name='video_stream1'),
]