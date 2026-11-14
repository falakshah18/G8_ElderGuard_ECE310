

#include <WiFi.h>
#include <WiFiUdp.h>
#include "esp_wifi.h"   
                        

#define WIFI_SSID "ElderGuard-TX"
#define WIFI_PASS "elderguard"

static uint8_t tx_mac[6]     = {0};
static bool    have_tx_mac   = false;
static volatile uint32_t frames_out = 0;
static volatile uint32_t frames_raw = 0;   

WiFiUDP probe;                             
static volatile uint32_t udp_pkts = 0;     

static const char *CSI_HEADER =
  "type,channel,mac,rssi,rate,sig_mode,mcs,bandwidth,smoothing,not_sounding,"
  "aggregation,stbc,fec_coding,sgi,ampdu_cnt,ant,timestamp,data";

void wifi_csi_callback(void *ctx, wifi_csi_info_t *info) {
  if (!info || !info->buf || info->len < 2) return;
  uint32_t rawid = ++frames_raw;
  if (rawid <= 8) {   
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

  WiFi.setSleep(false);   
  probe.begin(5000);      

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
