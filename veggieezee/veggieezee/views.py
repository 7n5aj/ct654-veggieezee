from django.http import HttpResponse
from .forms import UserRegistrationForm
from django.shortcuts import render,HttpResponse,redirect
from django.contrib.auth.models import User
from django.contrib.auth import authenticate,login,logout
from django.contrib import messages
from django.contrib.auth.decorators import login_required

def home(request):
  return render(request,"website/index.html")
def trade(request):
  return render(request,"trade/trade.html")
def insights(request):
  return render(request,"trade/insights.html")
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

