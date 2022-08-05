from __future__ import absolute_import, unicode_literals
import os
from celery import Celery
from django.apps import apps

# 为Celery应用配置Django配置模块
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'model_visual.settings')
app = Celery('model_visual')

app.config_from_object('django.conf:settings')
# 自动发现任务模块
app.autodiscover_tasks(lambda : [n.name for n in apps.get_app_configs()])