/*
 * ElderGuard - ESP8266 transmitter firmware (M2)
 * ECE 310 Group 8
 *
 * The ESP8266 has no CSI API in its SDK, so it is used purely as a dedicated
 * 802.11 traffic source. It brings up a soft-AP and floods UDP packets at a
 * steady rate; the ESP32 receiver joins this soft-AP and captures CSI on the
 * frames it receives.
 *
 * Board: NodeMCU v3 (ESP8266). Core: Arduino-ESP8266.
 * Configure WIFI_SSID / WIFI_PASS / UDP_PORT below.
 */

#include <ESP8266WiFi.h>
#include <WiFiUdp.h>

#define WIFI_SSID   "ElderGuard-TX"
#define WIFI_PASS   "elderguard"
#define UDP_PORT    5000
#define PKT_RATE_HZ 100
#define PKT_SIZE    64

WiFiUDP udp;
IPAddress broadcastAddr(192, 168, 4, 255);
uint8_t payload[PKT_SIZE];

void setup() {
  Serial.begin(115200);
  delay(200);

  WiFi.mode(WIFI_AP);
  WiFi.softAP(WIFI_SSID, WIFI_PASS);
  Serial.print("Soft-AP up. Gateway: ");
  Serial.println(WiFi.softAPIP());

  for (int i = 0; i < PKT_SIZE; i++) payload[i] = (uint8_t)('A' + (i % 26));
  udp.begin(UDP_PORT);
  Serial.println("Transmitter ready.");
}

void loop() {
  udp.beginPacket(broadcastAddr, UDP_PORT);
  udp.write(payload, PKT_SIZE);
  udp.endPacket();
  delay(1000 / PKT_RATE_HZ);   // ~100 packets/s
}
