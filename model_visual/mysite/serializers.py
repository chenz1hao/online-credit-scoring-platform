from rest_framework import serializers
from .models import data_set

class DataSetModelSerializer(serializers.ModelSerializer):
    class Meta:
        model = data_set
        fields = "__all__"