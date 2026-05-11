# Packaging ZeroTouch as a Windows Executable

This guide explains how to convert the ZeroTouch project into a standalone `.exe` file.

## Prerequisites

1.  **Virtual Environment**: Ensure you are using the provided `venv`.
2.  **PyInstaller**: Install it within your virtual environment:
    ```bash
    pip install pyinstaller
    ```

## Packaging Instructions

Run the following command from the root of the project (`d:\Protel`):

```bash
pyinstaller protel_app.spec
```

### What this command does:
- Uses `protel_app.spec` to bundle all model files (`.tflite`, `.task`).
- Sets `launcher.py` as the entry point.
- Uses the `--noconsole` flag to hide the background terminal window.
- Creates a `dist/` folder containing `ZeroTouch.exe`.

## Files Included in Bundle
The following assets are automatically bundled into the executable:
- `hand_landmarker.task`
- `gesture_classifier.tflite`
- `gesture_classifier_lstm.tflite`

## Running the App
After packaging, you can find the executable in:
`d:\Protel\dist\ZeroTouch\ZeroTouch.exe`

You can move this `ZeroTouch` folder anywhere on your computer and run the app without needing Python installed.
