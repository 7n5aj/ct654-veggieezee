from django.db import models

# Create your models here.

class PersonalInfo(models.Model):
    first_name = models.CharField(max_length=100)
    last_name = models.CharField(max_length=100)
    email = models.EmailField(unique=True)
    password = models.CharField(max_length=100)


class VegetablePrice(models.Model):
    """
    Model to store historical vegetable prices from Kalimati Market
    """
    commodity_name = models.CharField(max_length=200)
    commodity_unit = models.CharField(max_length=50, default='KG')
    min_price = models.DecimalField(max_digits=10, decimal_places=2)
    max_price = models.DecimalField(max_digits=10, decimal_places=2)
    avg_price = models.DecimalField(max_digits=10, decimal_places=2)
    date = models.DateField()
    created_at = models.DateTimeField(auto_now_add=True)
    
    class Meta:
        unique_together = ['commodity_name', 'date']
        ordering = ['-date', 'commodity_name']
        indexes = [
            models.Index(fields=['commodity_name', 'date']),
            models.Index(fields=['date']),
        ]
    
    def __str__(self):
        return f"{self.commodity_name} - Rs.{self.avg_price} ({self.date})"
    
    @classmethod
    def get_latest_price(cls, commodity_name):
        """Get the most recent price for a commodity"""
        return cls.objects.filter(
            commodity_name__icontains=commodity_name
        ).order_by('-date').first()
    
    @classmethod
    def get_price_history(cls, commodity_name, days=30):
        """Get price history for a commodity"""
        from datetime import datetime, timedelta
        start_date = datetime.now().date() - timedelta(days=days)
        return cls.objects.filter(
            commodity_name__icontains=commodity_name,
            date__gte=start_date
        ).order_by('date')
