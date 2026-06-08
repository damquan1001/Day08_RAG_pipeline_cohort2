"""Make user-site packages visible when this repo runs under locked Anaconda."""

import site

site.addsitedir(site.getusersitepackages())
