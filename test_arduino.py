"""
Arduino connection test
Run this to check if Python can talk to your Arduino
"""
import serial
import serial.tools.list_ports
import time

print("=" * 45)
print("  ARDUINO CONNECTION TEST")
print("=" * 45)
print()

# List all available COM ports
print("Available COM ports:")
ports = serial.tools.list_ports.comports()
if not ports:
    print("  No COM ports found!")
else:
    for p in ports:
        print(f"  {p.device} - {p.description}")

print()

# Try to connect
port = input("Enter COM port to test (e.g. COM5): ").strip()

try:
    print(f"Connecting to {port}...")
    ser = serial.Serial(port, 9600, timeout=3)
    time.sleep(2)  # wait for Arduino reset

    print("Connected! Sending PING...")
    ser.write(b"PING\n")
    time.sleep(1)

    response = ""
    while ser.in_waiting:
        response += ser.readline().decode().strip()

    if response:
        print(f"Arduino replied: {response}")
    else:
        print("No response - check Arduino has the scanner.ino uploaded")

    print()
    print("Listening for button presses for 10 seconds...")
    print("Press the button on your Arduino now!")
    print()

    start = time.time()
    while time.time() - start < 10:
        if ser.in_waiting:
            line = ser.readline().decode().strip()
            print(f"Received: {line}")

    ser.close()
    print()
    print("Test complete!")

except serial.SerialException as e:
    print(f"ERROR: {e}")
    print()
    print("Possible fixes:")
    print("1. Check Arduino is plugged in via USB")
    print("2. Check the COM port number in Device Manager")
    print("3. Make sure scanner.ino is uploaded to the Arduino")
    print("4. Close Arduino IDE Serial Monitor if it's open")

input("\nPress Enter to exit...")
