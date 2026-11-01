/*
 * ElderGuard - ESP32-S-series CSI RECEIVER firmware (Board B)
 * ECE 310 Group 8 - MVP build
 *
 * Replaces M2_Group8/Data/arduino/receiver_csi for your two ESP32-S boards.
 * Fixed vs the repo version (each bug alone would kill the whole capture):
 *   1) Prints the CSV header line at startup. collect_serial.py discards
 *      every frame until it sees a header, so the repo sketch produced
 *      empty files.
 *   2) Header now has EXACTLY the same column count as the printed rows
 *      (repo header listed 19 names for 18-field rows, so parse_csi.py's
 *      DictReader always got data=None and skipped every row).
 *   3) WiFi.setSleep(false): station power-save batches/delays broadcast
 *      frames and destroys the ~100 Hz sample grid.
 *   4) Emits only frames from the transmitter AP (BSSID filter), not every
 *      neighbour Wi-Fi packet on the channel.
 *   5) Numbers are printed via (int) casts; Print::print(char) would emit
 *      characters instead of values on some cores.
 *   6) Checks esp_wifi_set_csi* return codes and prints PASS/FAIL.
 *
 * Verified against Arduino ESP32 core 3.x headers for ESP32-S3
 * (wifi_csi_config_t + wifi_pkt_rx_ctrl_t fields used here exist on
 * ESP32 / S2 / S3; CONFIG_ESP_WIFI_CSI_ENABLED=y in the core build).
 *
 * Output row (18 columns):
 *   type,channel,mac,rssi,rate,sig_mode,mcs,bandwidth,smoothing,not_sounding,
 *   aggregation,stbc,fec_coding,sgi,ampdu_cnt,ant,timestamp,data
 */

#include <WiFi.h>
#include <WiFiUdp.h>
#include "esp_wifi.h"   // Arduino core 3.x: WiFi.h only brings the types; the
                        // esp_wifi_set_csi* functions live in esp_wifi.h

#define WIFI_SSID "ElderGuard-TX"
#define WIFI_PASS "elderguard"

static uint8_t tx_mac[6]     = {0};
static bool    have_tx_mac   = false;
static volatile uint32_t frames_out = 0;
static volatile uint32_t frames_raw = 0;   // before the MAC filter (diagnostic)

WiFiUDP probe;                             // plain UDP listener (diagnostic)
static volatile uint32_t udp_pkts = 0;     // packets seen outside the CSI path

static const char *CSI_HEADER =
  "type,channel,mac,rssi,rate,sig_mode,mcs,bandwidth,smoothing,not_sounding,"
  "aggregation,stbc,fec_coding,sgi,ampdu_cnt,ant,timestamp,data";

void wifi_csi_callback(void *ctx, wifi_csi_info_t *info) {
  if (!info || !info->buf || info->len < 2) return;
  uint32_t rawid = ++frames_raw;
  if (rawid <= 8) {   // first 8 CSI frames: show who they came from
    Serial.printf("# raw%lu mac=%02x:%02x:%02x:%02x:%02x:%02x len=%d\n",
                  (unsigned long)rawid,
                  info->mac[0], info->mac[1], info->mac[2],
                  info->mac[3], info->mac[4], info->mac[5], (int)info->len);
  }
  if (have_tx_mac && memcmp(info->mac, tx_mac, 6) != 0) return;

  const wifi_pkt_rx_ctrl_t *rx = &info->rx_ctrl;
  Serial.printf("CSI,%d,%02x:%02x:%02x:%02x:%02x:%02x,"
                "%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,",
                (int)rx->channel,
                info->mac[0], info->mac[1], info->mac[2],
                info->mac[3], info->mac[4], info->mac[5],
                (int)rx->rssi, (int)rx->rate, (int)rx->sig_mode,
                (int)rx->mcs, (int)rx->cwb, (int)rx->smoothing,
                (int)rx->not_sounding, (int)rx->aggregation, (int)rx->stbc,
                (int)rx->fec_coding, (int)rx->sgi, (int)rx->ampdu_cnt,
                (int)rx->ant, (int)rx->timestamp);
  Serial.print("[");
  for (uint16_t i = 0; i < info->len; i++) {
    if (i) Serial.print(' ');
    // ESP32-S3 can mark the first four CSI bytes invalid. Preserve the
    // expected buffer shape while preventing those bytes becoming signal.
    Serial.print((int)((info->first_word_invalid && i < 4) ? 0 : info->buf[i]));
  }
  Serial.println("]");
  frames_out++;
}

void setup() {
  Serial.begin(921600);
  delay(500);
  Serial.println(CSI_HEADER);

  WiFi.mode(WIFI_STA);
  Serial.print("Connecting to transmitter AP");
  WiFi.begin(WIFI_SSID, WIFI_PASS);
  while (WiFi.status() != WL_CONNECTED) {
    delay(300);
    Serial.print(".");
  }
  Serial.println(" connected.");
  Serial.print("IP: ");
  Serial.println(WiFi.localIP());

  const uint8_t *bssid = WiFi.BSSID();
  if (bssid) {
    memcpy(tx_mac, bssid, 6);
    have_tx_mac = true;
    Serial.printf("Filtering on TX AP MAC %02x:%02x:%02x:%02x:%02x:%02x\n",
                  tx_mac[0], tx_mac[1], tx_mac[2],
                  tx_mac[3], tx_mac[4], tx_mac[5]);
  }

  WiFi.setSleep(false);   // critical for a steady ~100 Hz CSI rate
  probe.begin(5000);      // listen for the transmitter's UDP packets

  wifi_csi_config_t cfg = {
    .lltf_en           = true,
    .htltf_en          = true,
    .stbc_htltf2_en    = true,
    .ltf_merge_en      = true,
    .channel_filter_en = false,
    .manu_scale        = false,
    .shift             = 0,
  };
  esp_err_t e1 = esp_wifi_set_csi_config(&cfg);
  esp_err_t e2 = esp_wifi_set_csi_rx_cb(&wifi_csi_callback, NULL);
  esp_err_t e3 = esp_wifi_set_csi(true);
  Serial.printf("[%s] CSI enable: config=%d rxcb=%d enable=%d (0 = OK)\n",
                (e1 || e2 || e3) ? "FAIL" : "PASS", (int)e1, (int)e2, (int)e3);
}

void loop() {
  int sz;
  while ((sz = probe.parsePacket()) > 0) {
    while (probe.available()) probe.read();
    udp_pkts++;
  }

  static uint32_t last = 0;
  static uint32_t last_frames = 0;
  uint32_t now = millis();
  if (now - last >= 10000) {
    uint32_t n = frames_out;
    uint32_t r = frames_raw;
    uint32_t u = udp_pkts;
    Serial.printf("# alive rssi=%d udp_total=%lu raw_total=%lu frames_total=%lu (~%lu/s)\n",
                  WiFi.RSSI(), (unsigned long)u, (unsigned long)r, (unsigned long)n,
                  (unsigned long)((n - last_frames) * 1000 / (now - last)));
    last = now;
    last_frames = n;
  }
  delay(100);
}
