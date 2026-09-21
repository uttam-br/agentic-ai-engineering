class Colors:
    PURPLE = '\033[95m'
    CYAN = '\033[96m'
    DARKCYAN = '\033[36m'
    BLUE = '\033[94m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'
    END = '\033[0m'


def log_info(message: str, color: str = Colors.CYAN):
    """ log info message with color"""
    print(f"{color} {message} {Colors.END}")


def log_success(message: str, color: str = Colors.GREEN):
    """ log success message with color"""
    print(f"{color} {message} {Colors.END}")


def log_error(message: str, color: str = Colors.RED):
    """ log error message with color"""
    print(f"{color} {message} {Colors.END}")


def log_warning(message: str, color: str = Colors.YELLOW):
    """ log warning message with color"""
    print(f"{color} {message} {Colors.END}")


def log_header(message: str):
    """ log header message with emphasis"""
    print(f"{Colors.BOLD}{Colors.PURPLE} {'=' * 60} {Colors.END}")
    print(f"{Colors.BOLD}{Colors.PURPLE} {message} {Colors.END}")
    print(f"{Colors.BOLD}{Colors.PURPLE} {'=' * 60} {Colors.END}")

