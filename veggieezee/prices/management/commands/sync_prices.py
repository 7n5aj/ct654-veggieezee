"""
Django Management Command: Sync Prices

Loads Kalimati daily JSON into the database. Per calendar day (Asia/Kathmandu), performs
at most one successful API-backed import unless --force is passed.

Usage:
    python manage.py sync_prices
    python manage.py sync_prices --timeout=30
    python manage.py sync_prices --force
"""
from decimal import Decimal
from datetime import datetime

from django.core.management.base import BaseCommand
from django.core.cache import cache
from django.utils import timezone

from veggieezee.live_data_service import fetch_live_prices
from prices.models import VegetablePrice
from prices.snapshot import (
    should_skip_api_fetch,
    should_skip_api_for_market_date,
)


class Command(BaseCommand):
    help = 'Sync vegetable prices from Kalimati Market API'

    def add_arguments(self, parser):
        parser.add_argument(
            '--force',
            action='store_true',
            help='Force a fresh API fetch and upsert',
        )
        parser.add_argument(
            '--timeout',
            type=int,
            default=15,
            help='API request timeout in seconds (default: 15)',
        )

    def handle(self, *args, **options):
        timeout = options.get('timeout', 15)
        force = options.get('force', False)
        from django.conf import settings

        min_rows = getattr(settings, 'PRICES_SYNC_MIN_ROW_SKIP', 20)

        local_today = timezone.localdate()
        day_done_key = f'price_sync_done:{local_today.isoformat()}'

        if not force and should_skip_api_fetch(min_rows):
            self.stdout.write(
                self.style.SUCCESS(
                    f'Skip: already have {min_rows}+ rows for {local_today}.'
                )
            )
            return

        if not force and cache.get(day_done_key):
            self.stdout.write(
                self.style.SUCCESS(
                    f'Skip: sync already completed once for local day {local_today}.'
                )
            )
            return

        self.stdout.write(f'Fetching Kalimati API (timeout: {timeout}s)...')
        data = fetch_live_prices(timeout=timeout)

        if not data or 'prices' not in data:
            self.stdout.write(self.style.ERROR('Failed to fetch data from API'))
            return

        market_date_raw = data.get('date')
        if market_date_raw:
            market_date = datetime.strptime(market_date_raw, '%Y-%m-%d').date()
        else:
            market_date = local_today

        if not force and should_skip_api_for_market_date(market_date, min_rows):
            self.stdout.write(
                self.style.SUCCESS(
                    f'Skip upsert: {market_date} already has {min_rows}+ rows.'
                )
            )
            cache.set(day_done_key, 1, timeout=90000)
            return

        self.stdout.write(
            f'Market date: {market_date} ({len(data["prices"])} commodities)'
        )

        existing = {
            r.commodity_name: r
            for r in VegetablePrice.objects.filter(date=market_date)
        }

        to_create = []
        to_update = []
        skipped_count = 0

        for item in data['prices']:
            name = item['commodityname']
            try:
                unit = item['commodityunit']
                mn = Decimal(str(item['minprice']))
                mx = Decimal(str(item['maxprice']))
                avg = Decimal(str(item['avgprice']))
            except Exception as e:
                self.stdout.write(
                    self.style.WARNING(f'Parse error for {name}: {e}')
                )
                skipped_count += 1
                continue

            if name in existing:
                obj = existing[name]
                if (
                    obj.min_price != mn
                    or obj.max_price != mx
                    or obj.avg_price != avg
                    or obj.commodity_unit != unit
                ):
                    obj.commodity_unit = unit
                    obj.min_price = mn
                    obj.max_price = mx
                    obj.avg_price = avg
                    to_update.append(obj)
            else:
                to_create.append(
                    VegetablePrice(
                        commodity_name=name,
                        date=market_date,
                        commodity_unit=unit,
                        min_price=mn,
                        max_price=mx,
                        avg_price=avg,
                    )
                )

        batch = 300
        if to_create:
            VegetablePrice.objects.bulk_create(to_create, batch_size=batch)
        if to_update:
            VegetablePrice.objects.bulk_update(
                to_update,
                ['commodity_unit', 'min_price', 'max_price', 'avg_price'],
                batch_size=batch,
            )

        created_count = len(to_create)
        updated_count = len(to_update)
        cache.set(day_done_key, 1, timeout=90000)
        for d in {local_today, market_date}:
            cache.delete(f'veg_list:v2:{d.isoformat()}')

        self.stdout.write(
            self.style.SUCCESS(
                f'Sync complete: {created_count} created, {updated_count} updated, '
                f'{skipped_count} skipped (market {market_date})'
            )
        )
