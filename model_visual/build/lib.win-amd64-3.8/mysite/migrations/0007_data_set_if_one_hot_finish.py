# Generated by Django 3.2.2 on 2021-05-14 11:44

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('mysite', '0006_auto_20210513_2016'),
    ]

    operations = [
        migrations.AddField(
            model_name='data_set',
            name='if_one_hot_finish',
            field=models.BooleanField(default=False, verbose_name='one-hot编码文件是否生成完成'),
        ),
    ]