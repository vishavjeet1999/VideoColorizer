from datetime import datetime
from django.db import models

# Create your models here.
class Videos(models.Model):
    video = models.FileField(upload_to='', null=True)
    date_uploaded = models.DateTimeField(blank=True, null=True, default=datetime.now)
