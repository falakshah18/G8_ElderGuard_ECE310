/*
 * ElderGuard - ESP32 hardware self-test (M2)
 * ECE 310 Group 8
 *
 * Bring-up check for a single ESP32 board, run before the two-board capture:
 *   1. Serial is alive at 921600 baud.
 *   2. Wi-Fi can scan and find the transmitter soft-AP.
 *   3. The CSI callback fires and reports a non-zero buffer length.
 *
 * Flash, open the serial monitor, and expect the three PASS lines.
 */

#include <WiFi.h>

#define TX_SSID "ElderGuard-TX"

static volatile uint32_t csi_frames = 0;
static volatile int last_len = 0;

void csi_cb(void *ctx, wifi_csi_info_t *info) {
  if (info && info->buf) {
    csi_frames++;
    last_len = info->len;
  }
}

void setup() {
  Serial.begin(921600);
  delay(500);
  Serial.println("ElderGuard self-test starting...");

  // 1. Wi-Fi scan
  WiFi.mode(WIFI_STA);
  int n = WiFi.scanNetworks();
  bool found = false;
  for (int i = 0; i < n; i++) {
    if (WiFi.SSID(i) == TX_SSID) found = true;
  }
  Serial.printf("[%s] Wi-Fi scan: %d networks, transmitter %s\n",
                found ? "PASS" : "WARN", n, found ? "found" : "not found");

  // 2. CSI callback
  wifi_csi_config_t cfg = {
      .lltf_en = true, .htltf_en = true, .stbc_htltf2_en = true,
      .ltf_merge_en = true, .channel_filter_en = false,
      .manu_scale = false, .shift = 0,
  };
  esp_wifi_set_csi_config(&cfg);
  esp_wifi_set_csi_rx_cb(&csi_cb, NULL);
  esp_wifi_set_csi(true);

  WiFi.begin(TX_SSID, "elderguard");
  unsigned long t0 = millis();
  while (WiFi.status() != WL_CONNECTED && millis() - t0 < 15000) delay(500);
  Serial.printf("[%s] Connected to transmitter\n",
                WiFi.status() == WL_CONNECTED ? "PASS" : "FAIL");
}

void loop() {
  delay(2000);
  Serial.printf("[%s] CSI frames=%lu, last_len=%d\n",
                csi_frames > 0 ? "PASS" : "WAIT",
                (unsigned long)csi_frames, last_len);
}
