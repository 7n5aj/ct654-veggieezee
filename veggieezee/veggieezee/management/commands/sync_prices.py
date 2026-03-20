"""
Django management command to sync prices from Kalimati Market API
Usage: python manage.py sync_prices
"""
from django.core.management.base import BaseCommand
from datetime import datetime
from veggieezee.live_data_service import fetch_live_prices
from veggieezee.models import VegetablePrice


class Command(BaseCommand):
    help = 'Sync vegetable prices from Kalimati Market API'

    def add_arguments(self, parser):
        parser.add_argument(
            '--force',
            action='store_true',
            help='Force update even if data for today already exists',
        )

    def handle(self, *args, **options):
        self.stdout.write('Fetching prices from Kalimati Market API...')
        
        data = fetch_live_prices()
        
        if not data or 'prices' not in data:
            self.stdout.write(
                self.style.ERROR('Failed to fetch data from API')
            )
            return
        
        market_date = data.get('date')
        if market_date:
            market_date = datetime.strptime(market_date, '%Y-%m-%d').date()
        else:
            market_date = datetime.now().date()
        
        self.stdout.write(f'Market date: {market_date}')
        self.stdout.write(f'Found {len(data["prices"])} commodities')
        
        created_count = 0
        updated_count = 0
        skipped_count = 0
        
        for item in data['prices']:
            commodity_name = item['commodityname']
            
            try:
                price_obj, created = VegetablePrice.objects.update_or_create(
                    commodity_name=commodity_name,
                    date=market_date,
                    defaults={
                        'commodity_unit': item['commodityunit'],
                        'min_price': float(item['minprice']),
                        'max_price': float(item['maxprice']),
                        'avg_price': float(item['avgprice']),
                    }
                )
                
                if created:
                    created_count += 1
                else:
                    updated_count += 1
                    
            except Exception as e:
                self.stdout.write(
                    self.style.WARNING(f'Error saving {commodity_name}: {e}')
                )
                skipped_count += 1
        
        self.stdout.write(
            self.style.SUCCESS(
                f'Sync complete: {created_count} created, '
                f'{updated_count} updated, {skipped_count} skipped'
            )
        )
