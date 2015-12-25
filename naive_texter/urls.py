from django.conf.urls import patterns, include, url
from django.contrib import admin

urlpatterns = patterns('',
                       url(r'^spam_classifier/', include('spam_classifier.urls')),
                       url(r'^admin/', include(admin.site.urls)),
                       )