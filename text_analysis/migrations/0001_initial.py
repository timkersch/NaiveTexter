# -*- coding: utf-8 -*-
from __future__ import unicode_literals

from django.db import models, migrations
import text_analysis.custom_fields


class Migration(migrations.Migration):

    dependencies = [
    ]

    operations = [
        migrations.CreateModel(
            name='SpamData',
            fields=[
                ('id', models.AutoField(verbose_name='ID', serialize=False, auto_created=True, primary_key=True)),
                ('spam_data', text_analysis.custom_fields.SerializedDataField()),
            ],
            options={
            },
            bases=(models.Model,),
        ),
    ]
