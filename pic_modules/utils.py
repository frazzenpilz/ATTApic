import sys
import datetime

def log_messages(message, filename, line_number, level):
    """
    Logs messages with timestamp, file info, and severity level.
    Level: 1 = Info, 2 = Warning, 3 = Error
    """
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    prefix = ""
    if level == 1:
        prefix = "[INFO]"
    elif level == 2:
        prefix = "[WARNING]"
    elif level == 3:
        prefix = "[ERROR]"
    
    print(f"{prefix} {timestamp} - {message} (File: {filename}, Line: {line_number})")
    
    if level == 3:
        print("Exiting due to error.")
        sys.exit(1)

def log_brief(message, level):
    """
    Logs brief messages.
    """
    prefix = ""
    if level == 1:
        prefix = "[INFO]"
    elif level == 2:
        prefix = "[WARNING]"
    elif level == 3:
        prefix = "[ERROR]"
    
    print(f"{prefix} {message}")
