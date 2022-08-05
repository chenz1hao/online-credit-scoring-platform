"""model_visual URL Configuration

The `urlpatterns` list routes URLs to myviews. For more information please see:
    https://docs.djangoproject.com/en/3.2/topics/http/urls/
Examples:
Function myviews
    1. Add an import:  from my_app import myviews
    2. Add a URL to urlpatterns:  path('', myviews.home, name='home')
Class-based myviews
    1. Add an import:  from other_app.myviews import Home
    2. Add a URL to urlpatterns:  path('', Home.as_view(), name='home')
Including another URLconf
    1. Import the include() function: from django.urls import include, path
    2. Add a URL to urlpatterns:  path('blog/', include('blog.urls'))
"""
from django.conf.urls import url
from django.contrib import admin
from django.urls import path, include

from model_visual import settings

urlpatterns = [
    path('admin/', admin.site.urls),
    path('', include('mysite.urls')),
]
