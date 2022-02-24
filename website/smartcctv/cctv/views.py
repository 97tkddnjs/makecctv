from django.shortcuts import render
from django.http.response import StreamingHttpResponse
from cctv.camera import VideoCamera
from django.contrib.auth.models import User

# Create your views here.
def index(request):
    user = None
    if request.session.get('id'):
        user = User.objects.get(id=request.session.get('id'))
    context = {
        'user': user
    }

    return render(request, 'cctv/live.html', context=context)

def gen(camera):
    while True:
        frame = camera.get_frame()
        yield (b'--frame\r\n'
                b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')

def video_stream(request):
    return StreamingHttpResponse(gen(VideoCamera()),
                    content_type='multipart/x-mixed-replace; boundary=frame')

