from django.conf.urls import patterns, include, url
from django.contrib import admin

urlpatterns = [
        url(r'', include('spam_classifier.urls')),
        url(r'^admin/', include(admin.site.urls))
]