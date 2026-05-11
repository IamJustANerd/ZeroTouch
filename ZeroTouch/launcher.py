import tkinter as tk
from tkinter import ttk
import hand_tracker
import sys

def start_app():
    debug_on = debug_var.get()
    root.destroy()
    try:
        hand_tracker.main(debug=debug_on)
    except Exception as e:
        import tkinter as tk
        from tkinter import messagebox
        error_root = tk.Tk()
        error_root.withdraw()
        messagebox.showerror("Error", f"Application crashed: {str(e)}")
        error_root.destroy()

# GUI Setup
root = tk.Tk()
root.title("ZeroTouch Launcher")
root.geometry("300x200")
root.resizable(False, False)

# Styling
style = ttk.Style()
style.configure("TButton", padding=10, font=('Helvetica', 10))
style.configure("TCheckbutton", font=('Helvetica', 10))

frame = ttk.Frame(root, padding="20")
frame.pack(expand=True, fill="both")

title_label = ttk.Label(frame, text="ZeroTouch OS Extension", font=('Helvetica', 12, 'bold'))
title_label.pack(pady=(0, 20))

debug_var = tk.BooleanVar(value=True)
debug_check = ttk.Checkbutton(frame, text="Enable Debug Mode (Show Camera)", variable=debug_var)
debug_check.pack(pady=10)

start_button = ttk.Button(frame, text="START PROGRAM", command=start_app)
start_button.pack(pady=10)

footer_label = ttk.Label(frame, text="v1.0 - Operating Room Interface", font=('Helvetica', 8))
footer_label.pack(side="bottom", pady=(10, 0))

if __name__ == "__main__":
    root.mainloop()
