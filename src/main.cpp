#include <Arduino.h>
#include <Adafruit_MPU6050.h>
#include <Adafruit_SSD1306.h>
#include <Adafruit_Sensor.h>
#include <Wire.h>

// Definición del pin del LED integrado para la depuración
#define LED_BUILTIN 2

// Objetos para los sensores y la pantalla
Adafruit_MPU6050 mpu;
Adafruit_SSD1306 display = Adafruit_SSD1306(128, 32, &Wire);

// Banderas para verificar si los dispositivos se inicializaron correctamente
bool sensor_ok = false;
bool display_ok = false;

void setup() {
  Serial.begin(115200);
  pinMode(LED_BUILTIN, OUTPUT);
  Serial.println("--- Inicio de Diagnostico ---");
  Wire.begin();

  // 1. Diagnóstico de la pantalla OLED
  if (!display.begin(SSD1306_SWITCHCAPVCC, 0x3C)) {
    Serial.println("Error: No se pudo inicializar la pantalla OLED. Revise las conexiones.");
    for (int i = 0; i < 10; i++) {
      digitalWrite(LED_BUILTIN, HIGH);
      delay(100);
      digitalWrite(LED_BUILTIN, LOW);
      delay(100);
    }
  } else {
    display_ok = true;
    Serial.println("Pantalla OLED inicializada correctamente.");
    display.clearDisplay();
    display.setTextSize(1);
    display.setTextColor(WHITE);
    display.setCursor(0, 0);
    display.println("OLED OK");
    display.display();
    delay(1000);
  }

  // 2. Diagnóstico del sensor MPU6050
  if (!mpu.begin()) {
    Serial.println("Error: No se pudo encontrar el sensor MPU6050.");
    Serial.println("Verifique las conexiones I2C y la alimentacion.");
  } else {
    sensor_ok = true;
    Serial.println("Sensor MPU-6050 encontrado y inicializado.");
    mpu.setAccelerometerRange(MPU6050_RANGE_2_G);
    mpu.setGyroRange(MPU6050_RANGE_500_DEG);
    mpu.setFilterBandwidth(MPU6050_BAND_5_HZ);
  }

  Serial.println("--- Diagnostico Completo ---");
  delay(100);
}

void loop() {
  // Lógica para el LED parpadeante
  static unsigned long lastBlinkTime = 0;
  if (millis() - lastBlinkTime > 1000) {
    digitalWrite(LED_BUILTIN, !digitalRead(LED_BUILTIN));
    lastBlinkTime = millis();
  }

  if (sensor_ok && display_ok) {
    sensors_event_t a, g, temp;
    mpu.getEvent(&a, &g, &temp);

    // Limpiar pantalla y mostrar los valores en dos filas
    display.clearDisplay();
    display.setCursor(0, 0);
    display.print("A (m/s2): ");
    display.print(a.acceleration.x, 1);
    display.print(", ");
    display.print(a.acceleration.y, 1);
    display.print(", ");
    display.println(a.acceleration.z, 1);

    display.print("G (rad/s): ");
    display.print(g.gyro.x, 1);
    display.print(", ");
    display.print(g.gyro.y, 1);
    display.print(", ");
    display.println(g.gyro.z, 1);
    
    display.display();

    // También enviar al Serial Monitor para depuración
    Serial.print("A (m/s2): ");
    Serial.print(a.acceleration.x, 1);
    Serial.print(", ");
    Serial.print(a.acceleration.y, 1);
    Serial.print(", ");
    Serial.println(a.acceleration.z, 1);

    Serial.print("G (rad/s): ");
    Serial.print(g.gyro.x, 1);
    Serial.print(", ");
    Serial.print(g.gyro.y, 1);
    Serial.print(", ");
    Serial.println(g.gyro.z, 1);
    Serial.println("");

  } else if (!display_ok && sensor_ok) {
    // Si solo el sensor funciona, avisar por serial
    Serial.println("ERROR: Pantalla no detectada. Mostrando solo por Serial Monitor.");
    sensors_event_t a, g, temp;
    mpu.getEvent(&a, &g, &temp);
    Serial.print("Acc: "); Serial.print(a.acceleration.x); Serial.print(";"); Serial.print(a.acceleration.y); Serial.print(";"); Serial.println(a.acceleration.z);
  } else if (display_ok && !sensor_ok) {
    // Si solo la pantalla funciona, mostrar el mensaje de error del sensor en ella
    display.clearDisplay();
    display.setCursor(0, 0);
    display.println("ERROR: Sensor MPU6050");
    display.println("no detectado. Revise ");
    display.println("las conexiones.");
    display.display();
  }

  delay(100);
}