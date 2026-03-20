from django.http import HttpResponse, JsonResponse
from .forms import UserRegistrationForm
from django.shortcuts import render, HttpResponse, redirect
from django.contrib.auth.models import User
from django.contrib.auth import authenticate, login, logout
from django.contrib import messages
from django.contrib.auth.decorators import login_required
from django.views.decorators.http import require_http_methods
from django.views.decorators.csrf import csrf_exempt
import json

from .predict_service import (
    get_vegetables_list,
    get_predictable_vegetables,
    predict_price,
    get_price_trends,
    get_market_overview,
    get_top_movers,
    get_historical_prices,
)

from .live_data_service import (
    fetch_live_prices,
    get_all_live_prices,
    get_live_price,
    get_market_date,
)


def home(request):
    return render(request, "website/index.html")


def trade(request):
    """
    Dashboard view displaying current market prices and overview.
    
    Integrates live data from database and Kalimati API.
    """
    from prices.models import VegetablePrice
    from datetime import date
    
    vegetables = get_vegetables_list()
    overview = get_market_overview()
    market_date = get_market_date()
    
    has_db_data = VegetablePrice.objects.filter(date=date.today()).exists()
    is_live = has_db_data or market_date is not None
    
    context = {
        'vegetables': vegetables,
        'overview': overview,
        'market_date': market_date or str(date.today()) if has_db_data else None,
        'is_live': is_live,
    }
    return render(request, "trade/trade.html", context)


def insights(request):
    """Market insights view with price trend analysis."""
    vegetables = get_vegetables_list()
    top_movers = get_top_movers()

    context = {
        'vegetables': vegetables,
        'top_increases': top_movers.get('top_increases', []),
        'top_decreases': top_movers.get('top_decreases', []),
    }
    return render(request, "trade/insights.html", context)


def predictions(request):
    """
    Price prediction interface.
    
    Allows users to forecast vegetable prices for future dates.
    Only shows vegetables with available prediction models.
    """
    vegetables = get_predictable_vegetables()
    all_vegetables = get_vegetables_list()
    result = None

    if request.method == 'POST':
        vegetable = request.POST.get('vegetable')
        date = request.POST.get('date')

        if vegetable and date:
            result = predict_price(vegetable, date)

    context = {
        'vegetables': vegetables,
        'all_vegetables': all_vegetables,
        'result': result,
        'predictable_count': len(vegetables),
        'total_count': len(all_vegetables),
    }
    return render(request, "trade/predictions.html", context)


@csrf_exempt
@require_http_methods(["POST"])
def predict_api(request):
    """API endpoint for price predictions"""
    try:
        if request.content_type == 'application/json':
            data = json.loads(request.body)
        else:
            data = request.POST
        
        vegetable = data.get('vegetable')
        date = data.get('date')
        
        if not vegetable or not date:
            return JsonResponse({
                'success': False,
                'error': 'Vegetable and date are required'
            }, status=400)
        
        result = predict_price(vegetable, date)
        return JsonResponse(result)
    
    except Exception as e:
        return JsonResponse({
            'success': False,
            'error': str(e)
        }, status=500)


def vegetables_api(request):
    """API endpoint to get list of vegetables"""
    vegetables = get_vegetables_list()
    return JsonResponse({'vegetables': vegetables})


def historical_api(request):
    """API endpoint for historical price data"""
    vegetable = request.GET.get('vegetable')
    days = int(request.GET.get('days', 30))
    
    if not vegetable:
        return JsonResponse({
            'success': False,
            'error': 'Vegetable parameter is required'
        }, status=400)
    
    trends = get_price_trends(vegetable, days=days)
    
    if trends:
        return JsonResponse({
            'success': True,
            **trends
        })
    else:
        return JsonResponse({
            'success': False,
            'error': f'No data found for {vegetable}'
        }, status=404)


def market_overview_api(request):
    """API endpoint for market overview"""
    overview = get_market_overview()
    top_movers = get_top_movers()
    
    return JsonResponse({
        'success': True,
        'overview': overview,
        'top_movers': top_movers,
    })


def live_prices_api(request):
    """API endpoint for live prices from Kalimati Market"""
    try:
        prices = get_all_live_prices()
        market_date = get_market_date()
        
        return JsonResponse({
            'success': True,
            'date': market_date,
            'prices': prices,
            'count': len(prices),
        })
    except Exception as e:
        return JsonResponse({
            'success': False,
            'error': str(e)
        }, status=500)


def live_price_api(request):
    """API endpoint for a single commodity's live price"""
    commodity = request.GET.get('commodity')
    
    if not commodity:
        return JsonResponse({
            'success': False,
            'error': 'Commodity parameter is required'
        }, status=400)
    
    price = get_live_price(commodity)
    
    if price:
        return JsonResponse({
            'success': True,
            **price
        })
    else:
        return JsonResponse({
            'success': False,
            'error': f'No live price found for {commodity}'
        }, status=404)


def signup(request): 
    """
    Handle user registration for VegePrediction platform
    """
    if request.method == "POST":
        # Get form data
        first_name = request.POST.get('firstName')
        last_name = request.POST.get('lastName')
        email = request.POST.get('email')
        phone = request.POST.get('phone')
        farm_location = request.POST.get('farmLocation')
        crop_type = request.POST.get('cropType')
        password = request.POST.get('password')
        terms_accepted = request.POST.get('terms')
        
        # Validation checks
        if not terms_accepted:
            messages.error(request, "You must accept the Terms of Service and Privacy Policy")
            return render(request, 'website/signup.html')
        
        if not all([first_name, last_name, email, password, farm_location, crop_type]):
            messages.error(request, "Please fill in all required fields")
            return render(request, 'website/signup.html')
        
        # Check if email already exists
        if User.objects.filter(email=email).exists():
            messages.error(request, "An account with this email already exists")
            return render(request, 'website/signup.html')
        
        # Create username from email (or you can use a different approach)
        username = email.split('@')[0]
        
        # Check if username already exists, if so, append a number
        base_username = username
        counter = 1
        while User.objects.filter(username=username).exists():
            username = f"{base_username}{counter}"
            counter += 1
        
        # Password validation
        if len(password) < 8:
            messages.error(request, "Password must be at least 8 characters long")
            return render(request, 'website/signup.html')
        
        try:
            # Create user
            my_user = User.objects.create_user(
                username=email,
                email=email,
                password=password,
                first_name=first_name,
                last_name=last_name
            )
            my_user.save()
            
            # TODO: Store additional profile data (phone, farm_location, crop_type)
            # You'll need to create a UserProfile model for this
            # Example:
            # UserProfile.objects.create(
            #     user=my_user,
            #     phone=phone,
            #     farm_location=farm_location,
            #     primary_crop=crop_type
            # )
            
            # Success message
            messages.success(request, "Your account has been successfully created! Welcome to Veggieezee.")
            
            # Optional: Auto-login after signup
            # login(request, my_user)
            # return redirect('dashboard')  # or wherever you want to redirect
            
            return redirect('login')
            
        except Exception as e:
            messages.error(request, f"An error occurred during registration: {str(e)}")
            return render(request, 'signup.html')
    
    # GET request - render signup page
    return render(request, 'signup.html')


# Optional: Create a UserProfile model in models.py
"""
from django.db import models
from django.contrib.auth.models import User

class UserProfile(models.Model):
    user = models.OneToOneField(User, on_delete=models.CASCADE, related_name='profile')
    phone = models.CharField(max_length=20, blank=True, null=True)
    farm_location = models.CharField(max_length=200)
    primary_crop = models.CharField(max_length=50)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
    
    def __str__(self):
        return f"{self.user.username}'s Profile"
"""
from django.http import HttpResponse

def loginpage(request): 
    if request.method == "POST":
        email = request.POST.get('email')
        password = request.POST.get('password')
        
        # Find user by email
        user = User.objects.filter(email=email).first()
        if user is None:
            return HttpResponse("No account found with this email address.")
        # Authenticate user
        user = authenticate(request, email=email, password=password)
        
        if user is not None:
            # Login the user
            login(request, user)
            
            # Store session data
            request.session['email'] = email
            
        else:
            return redirect('trade')
    
    return render(request, 'website/login.html', {'form': UserRegistrationForm})
# ORIGINAL CODE
# def loginpage(request): 
#     if request.method == "POST":
#         email = request.POST.get('email')
#         password = request.POST.get('password')
        
#         # Find user by email
#         user = User.objects.filter(email=email).first()
#         if user is None:
#             messages.error(request, "No account found with this email address.")
#             return redirect('login')
        
#         # Authenticate user
#         user = authenticate(request, email=email, password=password)
        
#         if user is not None:
#             # Login the user
#             login(request, user)
            
#             # Store session data
#             request.session['email'] = user.email
            
#             # Redirect to homepage or dashboard
#             return redirect('home') 
#         else:
#             messages.error(request, "Incorrect password.")
#             return redirect('login')
    
#     return render(request, 'website/login.html')
def LogoutPage(request):
    """
    Handle user logout
    """
    logout(request)
    messages.success(request, "You have been successfully logged out.")
    return redirect('login')


def ForgotPasswordPage(request):
    """
    Handle forgot password requests
    """
    if request.method == "POST":
        email = request.POST.get('email')
        
        try:
            user = User.objects.get(email=email)
            
            # TODO: Implement password reset functionality
            # 1. Generate password reset token
            # 2. Send email with reset link
            # Example:
            # from django.contrib.auth.tokens import default_token_generator
            # from django.utils.http import urlsafe_base64_encode
            # from django.utils.encoding import force_bytes
            # from django.core.mail import send_mail
            #
            # token = default_token_generator.make_token(user)
            # uid = urlsafe_base64_encode(force_bytes(user.pk))
            # reset_link = request.build_absolute_uri(f'/reset-password/{uid}/{token}/')
            # 
            # send_mail(
            #     'Password Reset Request',
            #     f'Click here to reset your password: {reset_link}',
            #     'noreply@vegeprediction.com',
            #     [email],
            #     fail_silently=False,
            # )
            
            messages.success(request, "Password reset instructions have been sent to your email.")
            return redirect('login')
            
        except User.DoesNotExist:
            # Don't reveal if email exists or not (security best practice)
            messages.success(request, "If an account exists with this email, password reset instructions have been sent.")
            return redirect('login')
    
    return render(request, 'forgot_password.html')


# Optional: Decorator to protect views that require login
# from django.contrib.auth.decorators import login_required

# @login_required(login_url='login')
# def HomePage(request):
#     """
#     Example protected view - requires login
#     """
#     username = request.session.get('username', request.user.username)
#     context = {
#         'username': username,
#         'user': request.user
#     }
#     return render(request, 'home.html', context)

