from django.conf.urls import patterns, include, url
from django.contrib import admin

urlpatterns = patterns('',
                       url(r'^text_analysis/', include('text_analysis.urls')),
                       url(r'^admin/', include(admin.site.urls)),
                       )