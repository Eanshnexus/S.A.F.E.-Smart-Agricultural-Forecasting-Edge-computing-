#include <Arduino.h>
#include <WiFi.h>
#include <HTTPClient.h>
#include <ArduinoJson.h>
#include "DHT.h"

#define DHTPIN        4       
#define DHTTYPE       DHT22   
#define SOIL_ANALOG_PIN 34    


const char* WIFI_SSID     = "S15";         
const char* WIFI_PASSWORD = "12345678";     

// FastAPI server base URL and sensor ingestion endpoint
const char* API_BASE_URL  = "http://10.244.164.209:8000/sensor-data";
const char* DEVICE_ID     = "SAFE-ESP32-01";

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

  // Print structured sensor debug status to Serial Monitor
  Serial.println("----------------------------------------");
  if (!isnan(currentTemperature)) {
    Serial.printf("Temperature:   %.1f \xC2\xB0" "C\n", currentTemperature);
  } else {
    Serial.println("Temperature:   ERROR (NaN)");
  }

  if (!isnan(currentHumidity)) {
    Serial.printf("Humidity:      %.1f %%\n", currentHumidity);
  } else {
    Serial.println("Humidity:      ERROR (NaN)");
  }

  Serial.printf("Soil Raw:      %d\n", currentSoilRaw);
  Serial.printf("Soil Moisture: %.1f %%\n", currentSoilPercent);
  Serial.printf("WiFi Status:   %s\n", (WiFi.status() == WL_CONNECTED) ? "Connected" : "Disconnected");
  Serial.println("----------------------------------------");
}

/**
 * Transmit sensor readings to FastAPI backend over HTTP POST.
 * Fails gracefully without blocking if network or server is down.
 */
void transmitSensorData() {
  if (WiFi.status() != WL_CONNECTED) {
    Serial.println("[API] Skipped transmission: Wi-Fi not connected.");
    return;
  }

  // Ensure we have at least one valid reading before transmitting
  if (isnan(currentTemperature) && isnan(currentHumidity) && currentSoilRaw == -1) {
    Serial.println("[API] Skipped transmission: No valid sensor readings yet.");
    return;
  }

  // Build JSON payload conforming to FastAPI /sensor-data schema
  StaticJsonDocument<256> doc;
  doc["device_id"]             = DEVICE_ID;
  doc["temperature"]           = isnan(currentTemperature) ? 25.0 : currentTemperature;
  doc["humidity"]              = isnan(currentHumidity) ? 50.0 : currentHumidity;
  doc["soil_moisture_raw"]     = currentSoilRaw;
  doc["soil_moisture_percent"] = currentSoilPercent;
  doc["timestamp"]             = (unsigned long)(millis() / 1000); // Node uptime in seconds

  String jsonPayload;
  serializeJson(doc, jsonPayload);

  HTTPClient http;
  http.begin(API_BASE_URL);
  http.setTimeout(HTTP_TIMEOUT_MS);
  http.addHeader("Content-Type", "application/json");

  int httpResponseCode = http.POST(jsonPayload);

  if (httpResponseCode > 0) {
    if (httpResponseCode == HTTP_CODE_OK || httpResponseCode == 201) {
      String response = http.getString();
      Serial.println("[API] Sensor data sent successfully (HTTP 200)");
      Serial.printf("[API] Server Response: %s\n", response.c_str());
    } else {
      Serial.printf("[API Warning] Server responded with HTTP status: %d\n", httpResponseCode);
    }
  } else {
    Serial.printf("[API Error] HTTP POST failed, error: %s\n", http.errorToString(httpResponseCode).c_str());
    Serial.println("[API Note] Firmware continuing normal operation; will retry on next cycle.");
  }

  http.end(); // Free connection resources
}

// -----------------------------------------------------------------------------
// SETUP
// -----------------------------------------------------------------------------
void setup() {
  Serial.begin(115200);
  delay(1000); // Allow UART connection to settle

  // Configure ADC attenuation for full 0 - 3.3V scale (11dB gives ~0 to 3.3V measurement)
  analogSetAttenuation(ADC_11db);
  pinMode(SOIL_ANALOG_PIN, INPUT);

  // Initialize DHT sensor
  dht.begin();

  // Print startup overview banner
  printSystemBanner();

  // Initial Wi-Fi connection
  connectWiFi();
}

// -----------------------------------------------------------------------------
// MAIN LOOP (Non-blocking timing using millis())
// -----------------------------------------------------------------------------
void loop() {
  unsigned long currentMillis = millis();

  // 1. Maintain Wi-Fi connection with non-blocking checks
  if (WiFi.status() != WL_CONNECTED && (currentMillis - lastTransmitTime >= TRANSMIT_INTERVAL)) {
    connectWiFi();
  }

  // 2. Periodic Sensor Sampling (every SENSOR_READ_INTERVAL ms)
  if (currentMillis - lastSensorReadTime >= SENSOR_READ_INTERVAL) {
    lastSensorReadTime = currentMillis;
    readSensors();
  }

  // 3. Periodic Telemetry Transmission (every TRANSMIT_INTERVAL ms)
  if (currentMillis - lastTransmitTime >= TRANSMIT_INTERVAL) {
    lastTransmitTime = currentMillis;
    transmitSensorData();
  }

  // Short delay to yield execution to ESP32 RTOS background Wi-Fi tasks
  delay(10);
}


/*
 * ==============================================================================
 * Project: S.A.F.E. (Smart Agricultural Forecasting & Edge-computing)
 * Component: ESP32 Environmental Edge Node Firmware
 * Target Board: ESP32-WROOM-32 DevKit
 *
 * HARDWARE CONNECTIONS:
 * - DHT22 VCC      -> ESP32 3V3
 * - DHT22 DATA     -> GPIO4 (D4)
 * - DHT22 GND      -> ESP32 GND
 * - HW-080 VCC     -> ESP32 3V3
 * - HW-080 GND     -> ESP32 GND
 * - HW-080 AO/A0   -> GPIO34 (D34 - Input Only Analog Pin)
 * - HW-080 DO/D0   -> Unused
 * - ESP32 VIN/5V   -> ESP32-CAM 5V (Shared Power Rail)
 * - ESP32 GND      -> ESP32-CAM GND (Common Ground)
 *
 * NOTE: ESP32-CAM operates as an independent Wi-Fi camera node sending images
 * to the FastAPI backend over Wi-Fi. UART pins are NOT used for communication.
 * ==============================================================================
 */
