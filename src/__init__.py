"""Day 8 RAG Pipeline package."""

import site
import sys


USER_SITE = site.getusersitepackages()
if USER_SITE and USER_SITE not in sys.path:
    sys.path.append(USER_SITE)
