#!/usr/bin/env python
# -*- coding: utf-8 -*- #
from __future__ import unicode_literals
import os

AUTHOR = u'Jonathan Arfa'
SITENAME = u'Machine Learning and Tacos'
# SITEURL = 'http://localhost:8000'
SITEURL = 'http://machinelearningandtacos.com'


THEME = os.path.join(os.path.expanduser('~'), "pelican-themes/flex")
#settings for the flex theme
SITETITLE = 'Machine Learning and Tacos'
SITESUBTITLE = 'What else?'
SITEDESCRIPTION = 'Jonathan Arfa\'s Personal Blog'
SITELOGO = "/images/halong_kayak.jpg"

# BROWSER_COLOR = '#333'
# MAIN_MENU = True
# MENUITEMS = (#('Archives', '/archives.html'),
#              #('Categories', '/categories.html'),
#              #('Tags', '/tags.html'),
#              ("Home", "/index.html",),
#              ("About", "/pages/about.html",),
#              )


PATH = 'content'

TIMEZONE = u'America/New_York'
DEFAULT_LANG = u'en'

# Feed generation is usually not desired when developing
FEED_ALL_ATOM = None
CATEGORY_FEED_ATOM = None
TRANSLATION_FEED_ATOM = None
AUTHOR_FEED_ATOM = None
AUTHOR_FEED_RSS = None

# Social widget
SOCIAL = (
    ('twitter', 'https://twitter.com/jonarfa'),
    ('linkedin', 'https://www.linkedin.com/in/jarfa'),
    ('github', 'https://github.com/jarfa/'),
)

DEFAULT_PAGINATION = 10

GOOGLE_ANALYTICS = "UA-78236307-1"

# Uncomment following line if you want document-relative URLs when developing
#RELATIVE_URLS = True
