"""Logging helpers."""


class ColorPrint:
    HEADER = "\033[95m"
    BLUE = "\033[94m"
    CYAN = "\033[96m"
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    RED = "\033[91m"
    BOLD = "\033[1m"
    UNDERLINE = "\033[4m"
    RESET = "\033[0m"

    @staticmethod
    def red(text):
        print(f"{ColorPrint.RED}{text}{ColorPrint.RESET}")

    @staticmethod
    def green(*text):
        print(f"{ColorPrint.GREEN}{text}{ColorPrint.RESET}")

    @staticmethod
    def yellow(*text):
        print(f"{ColorPrint.YELLOW}{text}{ColorPrint.RESET}")

    @staticmethod
    def blue(*text):
        print(f"{ColorPrint.BLUE}{text}{ColorPrint.RESET}")

    @staticmethod
    def cyan(*text):
        print(f"{ColorPrint.CYAN}{text}{ColorPrint.RESET}")

    @staticmethod
    def header(*text):
        print(f"{ColorPrint.HEADER}{text}{ColorPrint.RESET}")

    @staticmethod
    def bold(*text):
        print(f"{ColorPrint.BOLD}{text}{ColorPrint.RESET}")

    @staticmethod
    def underline(*text):
        print(f"{ColorPrint.UNDERLINE}{text}{ColorPrint.RESET}")
