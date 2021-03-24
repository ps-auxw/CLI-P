import warnings

def setupWarningFilters():
    # Not our bugs. (Or: De-clutter test suite terminal output.)
    warnings.filterwarnings("ignore", module="clip", category=ResourceWarning, message="unclosed file")
    warnings.filterwarnings("ignore", module="distutils", category=DeprecationWarning, message="the imp module is deprecated in favour of importlib")
