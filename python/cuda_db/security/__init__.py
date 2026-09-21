"""Authentication primitives and API-key administration.

The package deliberately does not import ``keys`` here.  That keeps
``python -m cuda_db.security.keys`` from preloading its target module and
emitting a runpy warning before the one-time token.
"""
