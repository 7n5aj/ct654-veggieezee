from django.db import models


class VegetablePrice(models.Model):
    """
    Daily vegetable price records from Kalimati Market.
    
    Stores minimum, maximum, and average prices for each commodity.
    Data is synced daily from the official Kalimati Market API.
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
        """Retrieve the most recent price record for a commodity."""
        return cls.objects.filter(
            commodity_name__icontains=commodity_name
        ).order_by('-date').first()
    
    @classmethod
    def get_price_history(cls, commodity_name, days=30):
        """Retrieve historical prices for a commodity within specified days."""
        from datetime import datetime, timedelta
        start_date = datetime.now().date() - timedelta(days=days)
        return cls.objects.filter(
            commodity_name__icontains=commodity_name,
            date__gte=start_date
        ).order_by('date')
