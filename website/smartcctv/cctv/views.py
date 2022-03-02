from django.shortcuts import render
from django.http.response import StreamingHttpResponse
from cctv.camera import VideoCamera
from django.contrib.auth.models import User
from threading import Thread


CAMERA  = None

# Create your views here.
def index(request):
    global CAMERA
    user = None
    CAMERA = VideoCamera()
    t = Thread(target=CAMERA.run_server)
    t.start()
    if request.session.get('id'):
        user = User.objects.get(id=request.session.get('id'))
    context = {
        'user': user
    }

    return render(request, 'cctv/live.html', context=context)

def gen(camera):
    while len(camera.threads)==0:
        pass
    while True:
        print(22222222222)
        frame = camera.threads[0].get_frame()
        yield (b'--frame\r\n'
                b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')

def video_stream(request):
    return StreamingHttpResponse(gen(CAMERA),
                    content_type='multipart/x-mixed-replace; boundary=frame')

def gen1(camera):
    while len(camera.threads)>=2 and True:
        print(33333333333)
        frame = camera.threads[1].get_frame()
        yield (b'--frame\r\n'
                b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')

def video_stream1(request):
    return StreamingHttpResponse(gen1(CAMERA),
                    content_type='multipart/x-mixed-replace; boundary=frame')
