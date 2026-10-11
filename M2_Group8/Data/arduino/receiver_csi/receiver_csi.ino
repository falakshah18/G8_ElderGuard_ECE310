/*
 * ElderGuard - ESP32 CSI receiver firmware (M2)
 * ECE 310 Group 8
 *
 * Only the ESP32 can capture CSI. This sketch joins the ESP8266 transmitter's
 * soft-AP, enables the CSI callback, and prints one CSV row per received frame
 * in the ESP32-CSI-Tool format consumed by Code/src/parse_csi.py.
 *
 * Board: ESP32-WROOM-32D. Core: Arduino-ESP32 (>= 2.0.x).
 * Flash with the ESP32-CSI-Tool / espressif CSI example enabled.
 *
 * Output row:
 *   type,role,mac,rssi,rate,sig_mode,mcs,bandwidth,...,data
 * where `data` is the bracketed signed-byte CSI buffer (imag,real pairs).
 */

#include <WiFi.h>
#include <WiFiUdp.h>

#define WIFI_SSID   "ElderGuard-TX"
#define WIFI_PASS   "elderguard"

static bool csi_ready = false;

void wifi_csi_callback(void *ctx, wifi_csi_info_t *info) {
  if (!info || !info->buf) return;

  Serial.printf("CSI,%d,%02x:%02x:%02x:%02x:%02x:%02x,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,",
                info->rx_ctrl.channel,
                info->mac[0], info->mac[1], info->mac[2],
                info->mac[3], info->mac[4], info->mac[5],
                info->rx_ctrl.rssi,
                info->rx_ctrl.rate,
                info->rx_ctrl.sig_mode,
                info->rx_ctrl.mcs,
                info->rx_ctrl.cwb,
                info->rx_ctrl.smoothing,
                info->rx_ctrl.not_sounding,
                info->rx_ctrl.aggregation,
                info->rx_ctrl.stbc,
                info->rx_ctrl.fec_coding,
                info->rx_ctrl.sgi,
                info->rx_ctrl.ampdu_cnt,
                info->rx_ctrl.ant,
                info->rx_ctrl.timestamp);
  Serial.print("[");
  for (int i = 0; i < info->len; i++) {
    Serial.print((int8_t)info->buf[i]);
    if (i != info->len - 1) Serial.print(" ");
  }
  Serial.println("]");
}

void setup() {
  Serial.begin(921600);
  delay(500);

  WiFi.mode(WIFI_STA);
  WiFi.begin(WIFI_SSID, WIFI_PASS);
  Serial.print("Connecting to transmitter soft-AP");
  while (WiFi.status() != WL_CONNECTED) {
    delay(500);
    Serial.print(".");
  }
  Serial.println();
  Serial.print("Connected. IP: ");
  Serial.println(WiFi.localIP());

  // Enable CSI on the station interface, 20 MHz, capture on all RX.
  wifi_csi_config_t csi_config = {
      .lltf_en = true,
      .htltf_en = true,
      .stbc_htltf2_en = true,
      .ltf_merge_en = true,
      .channel_filter_en = false,
      .manu_scale = false,
      .shift = 0,
  };
  esp_wifi_set_csi_config(&csi_config);
  esp_wifi_set_csi_rx_cb(&wifi_csi_callback, NULL);
  esp_wifi_set_csi(true);
  csi_ready = true;
  Serial.println("CSI capture enabled.");
}

void loop() {
  delay(1000);
  if (!csi_ready) return;
  // Heartbeat so the serial log shows the board is alive.
  static uint32_t n = 0;
  if (++n % 10 == 0) {
    Serial.printf("# alive, rssi=%d\n", WiFi.RSSI());
  }
}
