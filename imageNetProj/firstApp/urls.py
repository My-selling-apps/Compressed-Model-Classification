from django.urls import path
from . import views

app_name = 'firstApp'

urlpatterns = [
    # Web interface routes
    path('', views.index, name='homepage'),
    path('predictImage', views.predictImage, name='predictImage'),
    
    # API routes - prefixed with api/
    path('api/predict/', views.predict_api, name='predict_api'),
]