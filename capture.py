import time
import json
import os
from pynput import keyboard

# --- Configuration ---
TARGET_PHRASE = "The quick brown fox jumps over the lazy dog."
DATA_DIR = "keystroke_data" 

# --- Global variables ---
session_events = []
key_press_times = {}
start_time = None

def on_press(key):
    """
    This function is called every time a key is pressed.
    """
    global start_time, session_events, key_press_times

    if start_time is None:
        start_time = time.perf_counter()

    current_time = time.perf_counter()
    relative_time_ms = (current_time - start_time) * 1000

    try:
        key_char = key.char
    except AttributeError:
        key_char = str(key)

    event_data = {
        "event": "press",
        "key": key_char,
        "time_ms": relative_time_ms
    }
    session_events.append(event_data)
    
    if key not in key_press_times:
        key_press_times[key] = current_time

    # --- NEW: Print feedback to the terminal ---
    if key == keyboard.Key.space:
        print(' ', end='', flush=True) # Print a real space
    elif key == keyboard.Key.backspace:
        # Move cursor back, print space, move back again
        print('\b \b', end='', flush=True) 
    elif key_char is not None and len(key_char) == 1:
        print(key_char, end='', flush=True) # Print the character

def on_release(key):
    """
    This function is called every time a key is released.
    """
    global start_time, session_events, key_press_times

    if start_time is None:
        return

    current_time = time.perf_counter()
    relative_time_ms = (current_time - start_time) * 1000

    try:
        key_char = key.char
    except AttributeError:
        key_char = str(key)
        
    hold_time_ms = None
    if key in key_press_times:
        hold_time_ms = (current_time - key_press_times[key]) * 1000
        del key_press_times[key]

    event_data = {
        "event": "release",
        "key": key_char,
        "time_ms": relative_time_ms,
        "hold_time_ms": hold_time_ms
    }
    session_events.append(event_data)

    if key == keyboard.Key.enter:
        print("\n--- Session Complete ---")
        return False  # Stops the listener

# --- Main execution ---
print(f"Please type the following phrase and press Enter when finished:\n")
print(f"'{TARGET_PHRASE}'")
print("\nStarting listener... (Begin typing)")
print(">", end=' ', flush=True) # Add a little prompt

# Start the listener
with keyboard.Listener(on_press=on_press, on_release=on_release) as listener:
    listener.join()

# --- Save the data ---
if session_events:
    if not os.path.exists(DATA_DIR):
        os.makedirs(DATA_DIR)

    timestamp = int(time.time() * 1000)
    filename = os.path.join(DATA_DIR, f"session_{timestamp}.json")

    with open(filename, 'w') as f:
        json.dump(session_events, f, indent=4)
        
    print(f"\nSuccessfully captured {len(session_events)} events.")
    print(f"Data saved to: {filename}")
else:
    print("No events were captured.")