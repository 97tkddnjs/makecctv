from django.shortcuts import render
from django.contrib.auth.models import User

# Create your views here.
def landing(request):
    user = None
    if request.session.get('id'):
        user = User.objects.get(id=request.session.get('id'))

    context = {
        'user': user
    }
    return render(request, "singlePage/index.html", context=context)