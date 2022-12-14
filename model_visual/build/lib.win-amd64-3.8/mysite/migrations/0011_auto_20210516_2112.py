# Generated by Django 3.2.2 on 2021-05-16 13:12

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('mysite', '0010_auto_20210516_1919'),
    ]

    operations = [
        migrations.AddField(
            model_name='task',
            name='model_run_random_state',
            field=models.FloatField(default=None, verbose_name='随机种子'),
        ),
        migrations.AlterField(
            model_name='task',
            name='task_desc',
            field=models.CharField(default=None, max_length=100, verbose_name='模型运行描述'),
        ),
    ]
