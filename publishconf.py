#!/usr/bin/env python
# -*- coding: utf-8 -*- #
from __future__ import unicode_literals

# This file is only used if you use `make publish` or
# explicitly specify it as your config file.

import os
import sys
sys.path.append(os.curdir)
from pelicanconf import *

SITEURL = 'http://jarfa.github.io'
RELATIVE_URLS = False

FEED_ALL_ATOM = None #'feeds/all.atom.xml' #un-comment to allow an RSS feed
CATEGORY_FEED_ATOM = None #'feeds/%s.atom.xml' #un-comment to allow an RSS feed

DELETE_OUTPUT_DIRECTORY = True

STATIC_PATHS = ['images', 'extra/CNAME']
EXTRA_PATH_METADATA = {'extra/CNAME': {'path': 'CNAME'},} #this path leads to where jarfa.github.io should redirect to

# Following items are often useful when publishing

#DISQUS_SITENAME = ""
#GOOGLE_ANALYTICS = ""
