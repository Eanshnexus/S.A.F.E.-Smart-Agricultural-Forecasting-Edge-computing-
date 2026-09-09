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
  Serial.pri