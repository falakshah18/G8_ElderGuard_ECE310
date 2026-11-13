/*
 * ElderGuard - ESP32-S-series TRANSMITTER firmware (Board A)
 * ECE 310 Group 8 - MVP build
 *
 * Ported from the repo's ESP8266 sketch (M2_Group8/Data/arduino/
 * transmitter_esp8266) to any ESP32 chip (classic ESP32 / S2 / S3),
 * because you have two ESP32-S boards and no ESP8266.
 *
 * Role: create Wi-Fi soft-AP "ElderGuard-TX" on a FIXED channel and flood
 * ~100 UDP packets/s. The receiver board joins this AP and captures CSI
 * from every packet it receives.
 *
 * Arduino IDE: Board = your chip's "Dev Module"; if the board is plugged in
 * over its native USB-C port set "USB CDC On Boot: Enabled".
 */

#include <WiFi.h>
#include <WiFiUdp.h>

#define WIFI_SSID "ElderGuard-TX"
#define WIFI_PASS "elderguard" // must be >= 8 characters (WPA2)
#define WIFI_CHANNEL 1         // fixed so TX and RX never drift apart
#define UDP_PORT 5000
#define PKT_RATE_HZ 100
#define PKT_SIZE 64

WiFiUDP udp;
IPAddress broadcastAddr(192, 168, 4, 255);
IPAddress receiverAddr(192, 168, 4, 2); // unicast backup in case the AP
                                        // drops broadcast forwarding
uint8_t payload[PKT_SIZE];
static uint32_t bcast_ok = 0, bcast_err = 0, uni_ok = 0;
static uint32_t last_reported = 0;

void setup()
{
  Serial.begin(115200);
  delay(200);

  WiFi.mode(WIFI_AP);
  bool ok = WiFi.softAP(WIFI_SSID, WIFI_PASS, WIFI_CHANNEL);
  Serial.print("Soft-AP ");
  Serial.println(ok ? "up" : "FAILED - check name/cable");
  Serial.print("SSID: ");
  Serial.println(WIFI_SSID);
  Serial.print("Channel: ");
  Serial.println(WIFI_CHANNEL);
  Serial.print("Gateway: ");
  Serial.println(WiFi.softAPIP());

  for (int i = 0; i < PKT_SIZE; i++)
    payload[i] = (uint8_t)('A' + (i % 26));
  udp.begin(UDP_PORT);
  Serial.println("Transmitter ready.");
}

void loop()
{
  udp.beginPacket(broadcastAddr, UDP_PORT);
  udp.write(payload, PKT_SIZE);
  if (udp.endPacket())
    bcast_ok++;
  else
    bcast_err++;

  udp.beginPacket(receiverAddr, UDP_PORT);
  udp.write(payload, PKT_SIZE);
  if (udp.endPacket())
    uni_ok++;

  static uint32_t last = 0;
  if (millis() - last >= 10000)
  {
    last = millis();
    Serial.printf("# tx bcast_ok=%lu bcast_err=%lu uni_ok=%lu (~%lu/s total)\n",
                  (unsigned long)bcast_ok, (unsigned long)bcast_err,
                  (unsigned long)uni_ok,
                  (unsigned long)((bcast_ok + uni_ok - last_reported) / 10));
    last_reported = bcast_ok + uni_ok;
  }
  delay(1000 / PKT_RATE_HZ); // ~100 packets/s
}
