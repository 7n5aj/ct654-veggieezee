"""
Post-Clone Setup Verification Script

Run this after cloning to verify your environment is ready.
"""
import sys
import os

def check_python_version():
    """Check Python version"""
    version = sys.version_info
    print(f"Python version: {version.major}.{version.minor}.{version.micro}")
    
    if version.major < 3 or (version.major == 3 and version.minor < 10):
        print("  [WARNING] Python 3.10+ recommended")
        return False
    print("  [OK] Python version is compatible")
    return True

def check_required_packages():
    """Check if key packages can be imported"""
    required = [
        ('django', 'Django'),
        ('pandas', 'Pandas'),
        ('numpy', 'NumPy'),
        ('xgboost', 'XGBoost'),
        ('sklearn', 'scikit-learn'),
        ('selenium', 'Selenium'),
        ('cloudscraper', 'Cloudscraper'),
    ]
    
    print("\nChecking packages:")
    all_good = True
    
    for module, name in required:
        try:
            __import__(module)
            print(f"  [OK] {name}")
        except ImportError:
            print(f"  [MISSING] {name}")
            all_good = False
    
    return all_good

def check_project_structure():
    """Check if essential files/folders exist"""
    print("\nChecking project structure:")
    
    required_files = [
        'manage.py',
        'veggieezee/settings.py',
        'models/nepal_veg_price_xgboost.pkl',
        'models/nepal_veg_label_encoder.pkl',
        'predict/ml/data.xlsx',
    ]
    
    all_good = True
    for file_path in required_files:
        if os.path.exists(file_path):
            print(f"  [OK] {file_path}")
        else:
            print(f"  [MISSING] {file_path}")
            all_good = False
    
    return all_good

def check_database():
    """Check if database file exists"""
    print("\nChecking database:")
    
    if os.path.exists('db.sqlite3'):
        print("  [INFO] Database exists (will use existing data)")
    else:
        print("  [INFO] Database not found (will be created on first migration)")
    
    return True

def main():
    print("="*60)
    print("Nepal Veggie Price Tracker - Setup Verification")
    print("="*60)
    
    results = []
    
    results.append(("Python Version", check_python_version()))
    results.append(("Required Packages", check_required_packages()))
    results.append(("Project Structure", check_project_structure()))
    results.append(("Database", check_database()))
    
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    
    all_passed = all(result[1] for result in results)
    
    for name, passed in results:
        status = "[OK]" if passed else "[FAIL]"
        print(f"{status} {name}")
    
    if all_passed:
        print("\n[SUCCESS] Environment is ready!")
        print("\nNext steps:")
        print("1. python manage.py migrate")
        print("2. python manage.py sync_prices")
        print("3. python manage.py runserver 8080")
    else:
        print("\n[ERROR] Please fix the issues above")
        print("\nTo install missing packages:")
        print("  pip install -r requirements.txt")
    
    print("="*60)

if __name__ == '__main__':
    main()
