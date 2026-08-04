import os
import platform
import sys

def get_system_info():
    return {
        "OS Name": platform.system(),
        "OS Release": platform.release(),
        "OS Version": platform.version(),
        "Architecture": platform.machine(),
        "Python Version": sys.version.split()[0],
        "CPU Cores": os.cpu_count(),
    }

def print_cli_dashboard():
    print("=" * 45)
    print("         System Diagnostic Report         ")
    print("=" * 45)
    
    info = get_system_info()
    for label, value in info.items():
        print(f"    {label:<16}: {value}")
        
    print("=" * 45)
    
if __name__ == "__main__":
    print_cli_dashboard()