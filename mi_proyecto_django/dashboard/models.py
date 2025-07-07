from django.db import models

# Create your models here.
class RegistroHumedad(models.Model):
    timestamp = models.DateTimeField(auto_now_add=True)
    raw = models.IntegerField()
    humedad   = models.IntegerField() # Porcentaje de humedad

    def __str__(self):
        return f"{self.timestamp} - {self.humedad}%"