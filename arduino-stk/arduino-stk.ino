#include <LiquidCrystal_I2C.h>

LiquidCrystal_I2C lcd(0x27, 16, 2);

#define SENSOR_PIN        A2
#define WET_THRESHOLD     210
#define DRY_THRESHOLD     510
#define BUZZER_PIN        11      // conecta un cable del buzzer a digital 8 y otro a GND
#define HUMIDITY_ALERT    60     // porcentaje a partir del cual suena la alarma

void setup() {
  pinMode(BUZZER_PIN, OUTPUT);
  digitalWrite(BUZZER_PIN, LOW);

  Serial.begin(9600);
  lcd.init();
  lcd.backlight();
}

void loop() {
  int value = analogRead(SENSOR_PIN);
  int pct   = map(value, WET_THRESHOLD, DRY_THRESHOLD, 100, 0);
      pct   = constrain(pct, 0, 100);

  // mostrar en LCD
  lcd.setCursor(1,0);
  lcd.print("Moisture value");
  lcd.setCursor(3,1);
  lcd.print(value);
  lcd.print("   ");
  lcd.setCursor(10,1);
  lcd.print(pct);
  lcd.print("% ");

  // enviar por Serial
  Serial.print("Raw: ");
  Serial.print(value);
  Serial.print("  |  H2O%: ");
  Serial.print(pct);
  Serial.println("%");

  // alarma: si supera el umbral, emite tono; si no, apaga
  if (pct >= HUMIDITY_ALERT) {
    tone(BUZZER_PIN, 3000);   // activa buzzer a 1 kHz
  } else {
    noTone(BUZZER_PIN);       // desactiva buzzer
  }

  delay(500);
}
