# Nepal Veggie Price Tracker

Machine learning-based vegetable price prediction system for Nepal's Kalimati Market. Combines historical data analysis with real-time API integration to forecast prices for 50+ vegetables.

## Features

- Real-time price predictions using hybrid data approach
- Live integration with Kalimati Market API
- Intelligent vegetable name mapping system
- Daily automatic price synchronization
- Interactive dashboard and visualization
- RESTful API endpoints

## Tech Stack

- **Backend**: Django 6.0, Python 3.12
- **Data Processing**: Pandas, NumPy
- **Database**: SQLite (development) / PostgreSQL (production)
- **Frontend**: HTML5, CSS3, JavaScript

## Installation

1. Clone repository and navigate to project directory

2. Create and activate virtual environment:
```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

3. Install dependencies:
```bash
cd veggieezee
pip install -r requirements.txt
```

4. Run migrations:
```bash
python manage.py migrate
```

5. Sync initial price data:
```bash
python manage.py sync_prices
```

6. Start development server:
```bash
python manage.py runserver
```

7. Access application at `http://127.0.0.1:8000/trade/`

## Project Structure

```
veggieezee/
├── veggieezee/          # Main Django project
│   ├── predict_service.py    # ML prediction engine
│   ├── live_data_service.py  # API integration
│   ├── views.py              # Application views
│   └── urls.py               # URL routing
├── prices/              # Price data app
│   ├── models.py             # Database models
│   └── management/
│       └── commands/
│           └── sync_prices.py  # Data sync command
├── templates/           # HTML templates
└── predict/ml/         # Training data
```

## Usage

### Dashboard
View current market prices and trends at `/trade/`

### Predictions
Forecast future prices at `/predictions/`

### API Endpoints
- `POST /api/predict/` - Get price prediction
- `GET /api/vegetables/` - List available vegetables
- `GET /api/historical/` - Historical price data
- `GET /api/live-prices/` - Current market prices

### Daily Sync
Run manually or via cron:
```bash
python manage.py sync_prices --timeout=30
```

## License

[Add your license]

## Contributors

[Add your team members]
