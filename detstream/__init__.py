from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("detstream")
except PackageNotFoundError:
    __version__ = "0.0.0"
