# ZeroTouch: Executable Build & Usage Guide

This guide provides step-by-step instructions on how to package the ZeroTouch project into a Windows executable and how to operate it as a system extension.

## 1. How to Build the Executable (.exe)

Follow these steps to convert the Python scripts into a standalone app:

1.  **Open Terminal**: Open a terminal (Command Prompt or PowerShell) and navigate to your project directory:
    ```powershell
    cd d:\Protel
    ```
2.  **Activate Virtual Environment**:
    ```powershell
    .\venv\Scripts\activate
    ```
3.  **Install PyInstaller**: (Only if you haven't already)
    ```powershell
    pip install pyinstaller
    ```
4.  **Run Build Command**:
    Execute the following command to start the packaging process:
    ```powershell
    pyinstaller protel_app.spec
    ```
    *Note: This process may take a minute. It will create `build/` and `dist/` folders.*

---

## 2. How to Open and Use the App

Once the build is complete, you can find the application in the `dist` folder.

1.  **Navigate to the App**:
    Go to `d:\Protel\dist\ZeroTouch\`.
2.  **Launch the App**:
    Double-click **`ZeroTouch.exe`**.
3.  **Using the Launcher**:
    - A small menu will appear first.
    - **Debug Mode (Checklist)**:
        - **Checked**: Shows the camera feed and diagnostic info (use this for setup).
        - **Unchecked**: Runs as a clean background extension (no camera feed).
    - **Start Program**: Click this button to begin gesture control.
4.  **Operating the App**:
    - The app will anchor to the **bottom-right** of your screen and stay **Always on Top**.
    - **'q' Key**: Press 'q' while the tracking window is focused to quit.
    - **'r' Key**: Press 'r' to reset the system state to LISTENING.

---

## Troubleshooting

- **Camera Not Opening**: Ensure no other application (like Zoom or Teams) is using your webcam.
- **Missing Models**: The `protel_app.spec` handles model bundling automatically. If you get a "File Not Found" error, ensure you are running the `.exe` from within its `dist\ZeroTouch\` folder.
- **Antivirus Flags**: Sometimes PyInstaller-generated executables are flagged as "Unknown". You may need to click "Run anyway" on the Windows SmartScreen prompt.
