import sys
import os

# Ensure the parent directory is in the system path to allow importing local modules
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import openclaw_client

def main():
    print("==================================================")
    print("Testing OpenClaw Tool Calling (Llama 3.1)")
    print("==================================================")
    
    command = "Tolong catat \"Hari ini tanggal 6/18/2026\" ke file bernama HariIni.txt"
    print(f"\nUser Command: {command}\n")
    
    print("Sending to Llama 3.1...")
    result = openclaw_client.send_message(command)
    
    print("\n--- Result ---")
    print(f"Type: {result.type}")
    print(f"Text Response: {result.text}")
    
    if result.tool_calls:
        print("\n--- Tool Calls Requested ---")
        for idx, tool in enumerate(result.tool_calls):
            print(f"Tool {idx + 1}: {tool['name']}")
            print(f"Parameters: {tool['params']}")
    else:
        print("\nNo tool calls requested by Llama 3.1.")
        
    print("\n==================================================")

if __name__ == "__main__":
    main()
