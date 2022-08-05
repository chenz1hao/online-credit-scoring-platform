from django.urls import path
from . import views

urlpatterns = [
    path('', views.TaskList.as_view()),
    path('tasks/', views.TaskList.as_view()),
    path('tasks/startTask/', views.StartTask.as_view()),
    path('startTask/', views.StartTask.as_view()),
    path('tasks/TaskCompare/', views.TaskCompare.as_view()),

    path('dataset/', views.DataList.as_view()),
    path('dataset/uploadData/', views.UploadData.as_view()),
    path('dataset/uploadTempData/', views.UploadTempData.as_view()),
]