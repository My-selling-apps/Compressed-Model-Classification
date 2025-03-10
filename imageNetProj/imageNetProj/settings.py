from pathlib import Path
import os
# Build paths inside the project like this: BASE_DIR / 'subdir'.
BASE_DIR = Path(__file__).resolve().parent.parent
# SECURITY WARNING: keep the secret key used in production secret!
SECRET_KEY = 'django-insecure-^qc^)5+=rkv6%y$cxav9ga92a)t_+!b(l4mpx-qq)r08)_v@2w'
# SECURITY WARNING: don't run with debug turned on in production!
DEBUG = True
# Modify ALLOWED_HOSTS to include all required hosts
ALLOWED_HOSTS = [
    'localhost',
    '127.0.0.1',
    '0.0.0.0',  # Allow external requests on all interfaces
    '13.49.227.21',  # Your AWS IP address
    'phonebechpk.com',
    'www.phonebechpk.com',
    '*',  # Replace with your actual domain name or IP
]
# Application definition
INSTALLED_APPS = [
    'django.contrib.admin',
    'django.contrib.auth',
    'django.contrib.contenttypes',
    'django.contrib.sessions',
    'django.contrib.messages',
    'django.contrib.staticfiles',
    'corsheaders',  # Add CORS headers app
    'firstApp',
]
MIDDLEWARE = [
    'corsheaders.middleware.CorsMiddleware',  # Add CORS middleware first
    'django.middleware.security.SecurityMiddleware',
    'django.contrib.sessions.middleware.SessionMiddleware',
    'django.middleware.common.CommonMiddleware',
    'django.middleware.csrf.CsrfViewMiddleware',
    'django.contrib.auth.middleware.AuthenticationMiddleware',
    'django.contrib.messages.middleware.MessageMiddleware',
    'django.middleware.clickjacking.XFrameOptionsMiddleware',
]
# CORS Configuration
CORS_ALLOW_ALL_ORIGINS = True
CORS_ALLOW_METHODS = [
    'DELETE', 'GET', 'OPTIONS', 'PATCH', 'POST', 'PUT',
]
CORS_ALLOW_HEADERS = [
    'accept', 'accept-encoding', 'authorization', 'content-type',
    'dnt', 'origin', 'user-agent', 'x-csrftoken', 'x-requested-with',
]
# Allow requests from your testing domain or Postman
CORS_ALLOWED_ORIGINS = [
    "http://localhost:3000",
    "https://phonebechpk.com",  # Your frontend URL
    "http://phonebechpk.com",
    "http://www.phonebechpk.com",
    "https://www.phonebechpk.com",
    "http://13.60.241.230:8000",  # Your AWS server with port
    "https://www.postman.com",  # Postman requests
    "chrome-extension://mblodabbcapnkgcfnddfpfaamjngknck"  # Postman Chrome extension
]

# Add credentials support
CORS_ALLOW_CREDENTIALS = True

ROOT_URLCONF = 'imageNetProj.urls'
TEMPLATES = [
    {
        'BACKEND': 'django.template.backends.django.DjangoTemplates',
        'DIRS': [os.path.join(BASE_DIR,"template")],
        'APP_DIRS': True,
        'OPTIONS': {
            'context_processors': [
                'django.template.context_processors.debug',
                'django.template.context_processors.request',
                'django.contrib.auth.context_processors.auth',
                'django.contrib.messages.context_processors.messages',
            ],
        },
    },
]
WSGI_APPLICATION = 'imageNetProj.wsgi.application'
# Database
DATABASES = {
    'default': {
        'ENGINE': 'django.db.backends.sqlite3',
        'NAME': BASE_DIR / 'db.sqlite3',
    }
}
# Password validation
AUTH_PASSWORD_VALIDATORS = [
    {'NAME': 'django.contrib.auth.password_validation.UserAttributeSimilarityValidator',},
    {'NAME': 'django.contrib.auth.password_validation.MinimumLengthValidator',},
    {'NAME': 'django.contrib.auth.password_validation.CommonPasswordValidator',},
    {'NAME': 'django.contrib.auth.password_validation.NumericPasswordValidator',},
]
# Internationalization
LANGUAGE_CODE = 'en-us'
TIME_ZONE = 'UTC'
USE_I18N = True
USE_TZ = True
# Static files
STATIC_URL = 'static/'
STATIC_ROOT = os.path.join(BASE_DIR, 'static')
# Media files
MEDIA_URL = '/media/'
MEDIA_ROOT = os.path.join(BASE_DIR, 'media')
# Default primary key field type
DEFAULT_AUTO_FIELD = 'django.db.models.BigAutoField'
