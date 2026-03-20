"""
Django Management Command: Sync Prices

Fetches current vegetable prices from Kalimati Market API and stores them in database.
Run daily to maintain up-to-date price data.

Usage:
    python manage.py sync_prices
    python manage.py sync_prices --timeout=30
"""
from django.core.management.base import BaseCommand
from datetime import datetime
from veggieezee.live_data_service import fetch_live_prices
from prices.models import VegetablePrice


class Command(BaseCommand):
    help = 'Sync vegetable prices from Kalimati Market API'

    def add_arguments(self, parser):
        parser.add_argument(
            '--force',
            action='store_true',
            help='Force update even if data for today already exists',
        )
        parser.add_argument(
            '--timeout',
            type=int,
            default=15,
            help='API request timeout in seconds (default: 15)',
        )

    def handle(self, *args, **options):
        timeout = options.get('timeout', 15)
        self.stdout.write(f'Fetching prices from Kalimati Market API (timeout: {timeout}s)...')
        
        data = fetch_live_prices(timeout=timeout)
        
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
