# 🥬 Veggieezee: Price Predictor

**AI-Powered Vegetable Price Prediction System**

![Python](https://img.shields.io/badge/Python-3.9%2B-blue)
![Django](https://img.shields.io/badge/Django-6.0-green)
![Machine Learning](https://img.shields.io/badge/ML-XGBoost-orange)
![License](https://img.shields.io/badge/License-MIT-yellow)

[Features](#-features) • [Installation](#-installation) • [Usage](#-usage) • [Architecture](#-architecture) • [Contributing](#-contributing)

**Veggieezee** is an intelligent vegetable price prediction system designed to help farmers, traders, and agricultural businesses make data-driven decisions. By leveraging XGBoost machine learning algorithms and real-time Kalimati Market data, Veggieezee provides accurate price forecasts, seasonal insights, and trend analysis.

---

## 🌟 Features

- **XGBoost ML Model**: Advanced prediction using 62 engineered features
- **Real-time Data Integration**: Live price updates from Kalimati Market API
- **Historical Analysis**: 5+ years of price data (2021-2025)
- **Automated Data Sync**: Daily background price updates
- **Interactive Dashboard**: Visualize trends and predictions
- **Price Trend Analysis**: Intelligent detection of market movements
- **Multi-vegetable Support**: 50+ vegetables with predictive models

---

## 🚀 Installation

### Prerequisites

Before setting up the project, ensure you have:

1. **Python 3.10+**  
   ```bash
   python --version
   ```

2. **Google Chrome** (for data fetching)

3. **Git**  
   ```bash
   git --version
   ```

### Setup Instructions

1. **Clone the Repository**
   ```bash
   git clone https://github.com/7n5aj/ct654-veggieezee.git
   cd ct654-veggieezee
   ```

2. **Create Virtual Environment**
   ```bash
   python -m venv .venv
   ```

3. **Activate Virtual Environment**
   
   **Windows:**
   ```bash
   .venv\Scripts\activate
   ```
   
   **Mac/Linux:**
   ```bash
   source .venv/bin/activate
   ```

4. **Install Dependencies**
   ```bash
   cd veggieezee
   pip install -r requirements.txt
   ```

5. **Run Database Migrations**
   ```bash
   python manage.py migrate
   ```

6. **Fetch Initial Market Data**
   ```bash
   python manage.py sync_prices
   ```
   *Note: This takes 15-20 seconds on first run*

7. **Start Development Server**
   ```bash
   python manage.py runserver 8080
   ```

---

## 🌐 Access the Application

Open your browser and visit:

```
http://localhost:8080/trade/
```

**Available Pages:**
- `/trade/` - Main dashboard with live prices
- `/predictions/` - Price prediction interface
- `/insights/` - Market trends and analysis
- `/about-model/` - ML model documentation

---

## 💻 Usage

### Making Price Predictions

1. Navigate to the **Predictions** page
2. Select a vegetable from the dropdown (predictable vegetables only)
3. Choose a future date
4. Click **Predict Price**
5. View prediction with confidence intervals and historical trends

### Viewing Market Insights

1. Go to the **Insights** page
2. Select vegetables to compare
3. Choose date range (7, 14, or 30 days)
4. Analyze price movements and trends

### Dashboard Features

- **Live Price Updates**: Real-time data from Kalimati Market
- **Quick Predictions**: Fast access to common vegetables
- **Market Overview**: Statistics and price movers
- **Auto-sync Status**: Automatic daily data updates

---

## 🏗 Architecture

### Technology Stack

**Backend:**
- Django 6.0.3 - Web framework
- XGBoost 2.0+ - Machine learning model
- Selenium 4.0+ - Automated data fetching
- Pandas & NumPy - Data processing

**Database:**
- SQLite - Development database
- Django ORM - Database abstraction

**ML Pipeline:**
- 62 engineered features (lag, calendar, seasonal, festival)
- Trained on 30,000+ historical price records
- Log-transformed price predictions
- Dynamic confidence intervals

**Data Sources:**
- Historical Excel dataset (2021-2025)
- Kalimati Market API (real-time)
- Automated daily sync via middleware

### Project Structure

```
ct654-veggieezee/
├── veggieezee/
│   ├── veggieezee/              # Main application
│   │   ├── settings.py          # Configuration
│   │   ├── predict_service.py   # XGBoost prediction logic
│   │   ├── live_data_service.py # API integration
│   │   ├── selenium_fetcher.py  # Data fetching
│   │   └── middleware.py        # Auto-sync
│   ├── prices/                  # Price data models
│   ├── templates/               # HTML templates
│   ├── models/                  # ML model files
│   │   ├── nepal_veg_price_xgboost.pkl
│   │   └── nepal_veg_label_encoder.pkl
│   └── predict/ml/              # Historical data
│       └── data.xlsx
└── requirements.txt             # Dependencies
```

---

## 🤝 Contributing

We welcome contributions! Please follow these guidelines:

### How to Contribute

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/YourFeature`)
3. Make your changes
4. Run tests and verify functionality
5. Commit with clear messages
6. Push to your fork
7. Create a Pull Request

### Development Guidelines

- Follow PEP 8 style guide for Python code
- Add docstrings to functions and classes
- Test changes thoroughly before submitting
- Update documentation as needed

### Reporting Issues

- Use GitHub Issues for bug reports
- Provide detailed reproduction steps
- Include system information (OS, Python version)

---

## 📊 Model Information

**XGBoost Regressor:**
- 2746 decision trees
- 62 input features
- Log-transformed target variable
- Mean Absolute Error optimized

**Features Include:**
- 30-day price lags
- Calendar features (day, month, season)
- Festival indicators (Dashain, Tihar, Holi)
- Rolling statistics (7, 14, 30 days)
- Supply/demand scores

---

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.

---

## 🙏 Acknowledgments

- Kalimati Fruits and Vegetable Market Development Board for API access
- Historical price data sources
- Open source community

---

## 📞 Support

For questions or support:
- Open an issue on GitHub
- Check existing documentation
- Review the About Model page in the application

---

**Made with ❤️ for Agricultural Progress**

⭐ Star this repository if you find it helpful!
