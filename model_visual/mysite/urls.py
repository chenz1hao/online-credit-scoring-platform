from django.urls import path
from . import views

urlpatterns = [
    path('tasks/', views.TaskList.as_view()),
    path('tasks/startTask/', views.StartTask.as_view()),
    path('tasks/create_scorecard/', views.CreateScoreCard.as_view()),
    path('tasks/score_card/<int:id>', views.ScoreCard.as_view()),
    path('TaskCompare/', views.TaskCompare.as_view()),
    path('tasks/TaskCompare/', views.TaskCompare.as_view()),
    path('tasks/TaskResult/<int:id>', views.TaskResult.as_view()),
    path('tasks/nonnegaLogRegSC/', views.StartNonNegativeLogRegSC.as_view()),
    path('tasks/custom_score_card/<int:id>', views.CustomScoreCard.as_view()),
    path('tasks/sample_predict/<int:id>', views.SamplePredict.as_view()),
    path('tasks/sample_predict/', views.SamplePredict.as_view()),

    path('', views.DataList.as_view()),
    path('dataset/', views.DataList.as_view()),
    path('dataset/bin/<int:id>', views.DatasetBin.as_view()),
    path('dataset/uploadData/', views.UploadData.as_view()),
    path('dataset/uploadTempData/', views.UploadTempData.as_view()),

    path('binning/', views.BinningLibrary.as_view()),
    path('binning/startBinning/', views.StartBinning.as_view()),
    path('binning/result/<int:id>', views.BinningResult.as_view()),
    path('binning/update/<int:id>', views.BinningUpdate.as_view()),

    path('dataGenerate/', views.GeneratedDatasetList.as_view()),
    path('dataGenerate/startGenerateData/', views.StartGenerateData.as_view()),


    path('getDataFeatures/', views.GetDatasetFeatures.as_view()),
    path('getBinnedFeatures/', views.GetBinnedFeatures.as_view()),

]