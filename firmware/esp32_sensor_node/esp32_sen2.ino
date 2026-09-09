#include <Arduino.h>
#include <WiFi.h>
#include <HTTPClient.h>
#include <ArduinoJson.h>
#include "DHT.h"

// -----------------------------------------------------------------------------
// 1. PIN DEFINITIONS
// -----------------------------------------------------------------------------
#define DHTPIN        4       // GPIO4 connected to DHT22 DATA pin
#define DHTTYPE       DHT22   // Sensor model DHT22 (AM2302)
#define SOIL_ANALOG_PIN 34    // GPIO34 (ADC1_CH6) connected to HW-080 AO pin

// -----------------------------------------------------------------------------
// 2. WI-FI & API CONFIGURATION (Configure for your local network)
// -----------------------------------------------------------------------------
const char* WIFI_SSID     = "S15";         // Replace with your local Wi-Fi SSID
const char* WIFI_PASSWORD = "12345678";     // Replace with your Wi-Fi password

// FastAPI server base URL and sensor ingestion endpoint
// Example: "http://192.168.1.100:8000/sensor-data" (Use your host PC's local IP)
const char* API_BASE_URL  = "http://10.244.164.209:8000/sensor-data";
const char* DEVICE_ID     = "SAFE-ESP32-01";

// -----------------------------------------------------------------------------
// 3. SENSOR & TIMING CONSTANTS
// -----------------------------------------------------------------------------
// DHT22 sampling interval (must be >= 2000 ms to respect sensor specifications)
const unsigned long SENSOR_READ_INTERVAL = 3000;   // 3 seconds between reads

// API transmission interval
const unsigned long TRANSMIT_INTERVAL   = 5000;   // 5 seconds between HTTP POSTs

// HTTP client timeout in milliseconds
const int HTTP_TIMEOUT_MS               = 4000;

// -----------------------------------------------------------------------------
// 4. SOIL MOISTURE CALIBRATION CONSTANTS
// -----------------------------------------------------------------------------
/*
 * CALIBRATION NOTE:
 * ESP32 ADC resolution is 12-bit (0 to 4095).
 * The HW-080 resistive sensor outputs HIGHER voltage (near 4095) in dry air
 * and LOWER voltage (near 1000-1500) in saturated water.
 * These values are defaults and MUST be calibrated with your specific probe & soil:
 * 1. Read raw value in dry air -> SOIL_RAW_AIR_DRY
 * 2. Read raw value submerged in water -> SOIL_RAW_WATER_WET
 */
const int SOIL_RAW_AIR_DRY   = 3500;  // 0% moisture benchmark (sensor suspended in air)
const int SOIL_RAW_WATER_WET = 1200;  // 100% moisture benchmark (sensor dipped in water)

// -----------------------------------------------------------------------------
// GLOBAL INSTANCES & VARIABLES
// -----------------------------------------------------------------------------
DHT dht(DHTPIN, DHTTYPE);

unsigned long lastSensorReadTime = 0;
unsigned long lastTransmitTime   = 0;

float currentTemperature       = NAN;
float currentHumidity          = NAN;
int   currentSoilRaw           = -1;
float currentSoilPercent       = NAN;

// -----------------------------------------------------------------------------
// HELPER FUNCTIONS
// -----------------------------------------------------------------------------

/**
 * Print clean ASCII startup banner with pin mappings and system details.
 */
void printSystemBanner() {
  Serial.println("\n=======================================================");
  Serial.println("   S.A.F.E. - Smart Agricultural Forecasting & Edge    ");
  Serial.println("   ESP32-WROOM Microclimate Telemetry Node v1.0.0      ");
  Serial.println("=======================================================");
  Serial.println("Configured Hardware Pinout:");
  Serial.printf("  [+] DHT22 Temperature & Humidity : GPIO %d\n", DHTPIN);
  Serial.printf("  [+] HW-080 Soil Moisture AO      : GPIO %d (ADC1)\n", SOIL_ANALOG_PIN);
  Serial.println("  [+] ESP32-CAM Power Rail         : VIN 5V + GND (Common)");
  Serial.println("  [+] Camera Comm Channel          : Wi-Fi REST (No UART)");
  Serial.printf("  [+] Device Identifier            : %s\n", DEVICE_ID);
  Serial.printf("  [+] Target API Endpoint          : %s\n", API_BASE_URL);
  Serial.println("=======================================================\n");
}

/**
 * Connect to configured Wi-Fi network with non-infinite loop retry logic.
 */
void connectWiFi() {
  if (WiFi.status() == WL_CONNECTED) {
    return;
  }

  Serial.printf("[WiFi] Connecting to SSID: %s ", WIFI_SSID);
  WiFi.mode(WIFI_STA);
  WiFi.begin(WIFI_SSID, WIFI_PASSWORD);

  unsigned long startAttempt = millis();
  // Attempt connection for up to 10 seconds per cycle so sensors continue reading
  while (WiFi.status() != WL_CONNECTED && (millis() - startAttempt < 10000)) {
    delay(500);
    Serial.print(".");
  }

  if (WiFi.status() == WL_CONNECTED) {
    Serial.println("\n[WiFi] Connected successfully!");
    Serial.printf("[WiFi] Assigned IP Address: %s\n", WiFi.localIP().toString().c_str());
    Serial.printf("[WiFi] Signal Strength (RSSI): %d dBm\n\n", WiFi.RSSI());
  } else {
    Serial.println("\n[WiFi] Connection timeout. Operating in offline/retry mode.\n");
  }
}

/**
 * Read DHT22 and HW-080 analog probe with validation and non-blocking timing.
 */
void readSensors() {
  // Read analog soil moisture first (multisampling for ADC noise reduction)
  long adcAccumulator = 0;
  const int NUM_SAMPLES = 8;
  for (int i = 0; i < NUM_SAMPLES; i++) {
    adcAccumulator += analogRead(SOIL_ANALOG_PIN);
    delay(2);
  }
  currentSoilRaw = adcAccumulator / NUM_SAMPLES;

  // Compute moisture percentage using linear interpolation between calibration points
  // Inverse relationship: High raw ADC = Dry soil, Low raw ADC = Wet soil
  float mappedPercent = (float)(SOIL_RAW_AIR_DRY - currentSoilRaw) * 100.0f / (float)(SOIL_RAW_AIR_DRY - SOIL_RAW_WATER_WET);
  currentSoilPercent = constrain(mappedPercent, 0.0f, 100.0f);

  // Read temperature and humidity from DHT22
  float t = dht.readTemperature(); // Default: Celsius
  float h = dht.readHumidity();

  // Validate DHT22 reading
  if (isnan(t) || isnan(h)) {
    Serial.println("[Sensor Error] Failed to read from DHT22 sensor! Check wiring on GPIO4.");
  } else {
    currentTemperature = t;
    currentHumidity    = h;
  }