/*
  Arduino UNO Q - Scanner Button Trigger
  Tries multiple button pins automatically
  
  Upload this, then open Serial Monitor at 9600 baud
  Press each button/pin to find which one works
*/

#include "Arduino_LED_Matrix.h"

ArduinoLEDMatrix matrix;

// Try these pins - UNO Q may use different pin for built-in button
// Check Serial Monitor output to see which pin triggers
const int BUTTON_PINS[] = {2, 3, 4, 7, 8, 12};
const int NUM_PINS = 6;

int  lastStates[6];
int  frameCount = 0;

byte smiley[8][12] = {
  {0,0,0,0,0,0,0,0,0,0,0,0},
  {0,0,1,0,0,0,0,0,1,0,0,0},
  {0,0,1,0,0,0,0,0,1,0,0,0},
  {0,0,0,0,0,0,0,0,0,0,0,0},
  {0,1,0,0,0,0,0,0,0,1,0,0},
  {0,0,1,0,0,0,0,0,1,0,0,0},
  {0,0,0,1,1,1,1,1,0,0,0,0},
  {0,0,0,0,0,0,0,0,0,0,0,0}
};

byte bright[8][12] = {
  {1,1,1,1,1,1,1,1,1,1,1,1},
  {1,1,1,1,1,1,1,1,1,1,1,1},
  {1,1,1,1,1,1,1,1,1,1,1,1},
  {1,1,1,1,1,1,1,1,1,1,1,1},
  {1,1,1,1,1,1,1,1,1,1,1,1},
  {1,1,1,1,1,1,1,1,1,1,1,1},
  {1,1,1,1,1,1,1,1,1,1,1,1},
  {1,1,1,1,1,1,1,1,1,1,1,1}
};

void setup() {
  Serial.begin(9600);

  // Set all test pins as input with pullup
  for (int i = 0; i < NUM_PINS; i++) {
    pinMode(BUTTON_PINS[i], INPUT_PULLUP);
    lastStates[i] = HIGH;
  }

  matrix.begin();
  matrix.renderBitmap(smiley, 8, 12);

  Serial.println("READY");
  Serial.println("INFO:Press your button to find which pin it uses");
}

void loop() {

  // Check all pins
  for (int i = 0; i < NUM_PINS; i++) {
    int state = digitalRead(BUTTON_PINS[i]);

    if (state == LOW && lastStates[i] == HIGH) {
      delay(50); // debounce

      frameCount++;

      // Flash LED
      matrix.renderBitmap(bright, 8, 12);
      delay(150);
      matrix.renderBitmap(smiley, 8, 12);

      // Tell Python which pin triggered and send capture
      Serial.println("INFO:Button on pin " + String(BUTTON_PINS[i]));
      Serial.println("CAPTURE:" + String(frameCount));
    }

    lastStates[i] = state;
  }

  // Listen for Python commands
  if (Serial.available()) {
    String cmd = Serial.readStringUntil('\n');
    cmd.trim();

    if (cmd == "PING")  Serial.println("PONG");
    if (cmd == "START") { frameCount = 0; Serial.println("OK:STARTED"); }
    if (cmd == "STOP")  Serial.println("OK:STOPPED:" + String(frameCount));
    if (cmd == "RESET") { frameCount = 0; Serial.println("OK:RESET"); }
  }

  delay(10);
}
