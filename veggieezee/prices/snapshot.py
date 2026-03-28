"""Helpers for resolving which Kalimati snapshot date is available in the database."""
from django.db.models import Max
from django.utils import timezone

from prices.models import VegetablePrice


def effective_price_date(min_rows=20):
    """
    Date to join for listings: prefer a full row set for local today, else most recent
    date with enough commodities (so the dashboard still loads before sync finishes).
    """
    local_today = timezone.localdate()
    if VegetablePrice.objects.filter(date=local_today).count() >= min_rows:
        return local_today
    latest = VegetablePrice.objects.aggregate(m=Max('date'))['m']
    if latest is None:
        return None
    if VegetablePrice.objects.filter(date=latest).count() >= min_rows:
        return latest
    # Sparse table — still show whatever is newest
    if VegetablePrice.objects.filter(date=latest).exists():
        return latest
    return None


def should_skip_api_fetch(min_rows=20):
    """
    Skip Kalimati HTTP when today's snapshot is already in the DB (once per calendar day).

    We intentionally do *not* treat "yesterday still has rows" as skip, so a new local day
    can trigger a single fresh fetch after midnight in Asia/Kathmandu.
    """
    local_today = timezone.localdate()
    return VegetablePrice.objects.filter(date=local_today).count() >= min_rows


def should_skip_api_for_market_date(market_date, min_rows=20):
    """After reading API `date`, skip persistence if that market_date is already complete."""
    if not market_date:
        return False
    return VegetablePrice.objects.filter(date=market_date).count() >= min_rows
