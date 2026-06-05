#!/usr/bin/env python3
"""
╔══════════════════════════════════════════════════════════════════╗
║   Real-Time Network IDS — Transformer Autoencoder (LiteTAE)     ║
║   Dataset trained on: CSE-CIC-IDS2018                           ║
║                                                                  ║
║   Automated Single-Threshold Detection Logic:                    ║
║     [Phase 1] Calibration: Learns normal traffic behavior        ║
║     [Phase 2] Detection  : error >= Dynamic Threshold → ATTACK║
║                            error <  Dynamic Threshold → NORMAL║
║                                                                  ║
║   Usage:                                                         ║
║     sudo python ids_live.py --model lite_tae_full.pt             ║
║     sudo python ids_live.py --model lite_tae_full.pt --iface eth0║
║     python ids_live.py --model lite_tae_full.pt --pcap test.pcap ║
╚══════════════════════════════════════════════════════════════════╝
"""

import os
import sys
import math
import time
import json
import logging
import argparse
import threading
import warnings
from collections import defaultdict
from datetime import datetime

warnings.filterwarnings('ignore')

import numpy as np

try:
    import torch
    import torch.nn as nn
except ImportError:
    sys.exit("PyTorch :  pip install torch")

try:
    from scapy.all import sniff, IP, TCP, UDP, ICMP
except ImportError:
    sys.exit("❌ Scapy غير مثبتة. شغّل:  pip install scapy")


# ══════════════════════════════════════════════════════════════════
#  ANSI COLORS
# ══════════════════════════════════════════════════════════════════
class C:
    R  = '\033[0m'       # reset
    B  = '\033[1m'       # bold
    RED    = '\033[91m'
    ORANGE = '\033[33m'
    GREEN  = '\033[92m'
    CYAN   = '\033[96m'
    GRAY   = '\033[90m'
    YELLOW = '\033[93m'
    BLUE   = '\033[94m'
    WHITE  = '\033[97m'


NORMAL_PORTS = {
    80   : "HTTP",        8080 : "HTTP-alt",
    443  : "HTTPS",       8443 : "HTTPS-alt",
    21   : "FTP",         20   : "FTP-data",
    22   : "SSH",         2222 : "SSH-alt",
    53   : "DNS",
    25   : "SMTP",        587  : "SMTP-TLS",   465: "SMTPS",
    110  : "POP3",        995  : "POP3S",
    143  : "IMAP",        993  : "IMAPS",
    3306 : "MySQL",       5432 : "PostgreSQL",
    27017: "MongoDB",     6379 : "Redis",
    5672 : "AMQP",        15672: "RabbitMQ",
    123  : "NTP",         67   : "DHCP",       68: "DHCP",
    161  : "SNMP",        162  : "SNMP-trap",
    445  : "SMB",         139  : "NetBIOS",
    3389 : "RDP",         5985 : "WinRM",
    1194 : "OpenVPN",     500  : "IKE-VPN",    4500: "IPSec-NAT",
    8888 : "Jupyter",     6006 : "TensorBoard",
}

HIGH_RISK_PORTS = {
    23   : "Telnet",
    135  : "MS-RPC",
    137  : "NetBIOS-NS",  138: "NetBIOS",
    1433 : "MSSQL",       1521: "Oracle-DB",
    2375 : "Docker-API",  2376: "Docker-TLS",
    4444 : "Metasploit",
    5800 : "VNC-HTTP",    5900: "VNC",
    6666 : "IRC",         6667: "IRC",         6668: "IRC",
    31337: "BackOrifice",
    12345: "NetBus",      12346: "NetBus",
    27374: "SubSeven",
    1080 : "SOCKS",
    3128 : "Squid-Proxy",
    9200 : "Elasticsearch",
    6379 : "Redis-open",
}

PROTO_NAMES = {6: "TCP", 17: "UDP", 1: "ICMP", 2: "IGMP", 47: "GRE", 50: "ESP", 58: "ICMPv6"}

MODEL_FEATURES = [
    'Dst Port', 'Protocol', 'Flow Duration', 'Tot Fwd Pkts', 'Tot Bwd Pkts',
    'TotLen Fwd Pkts', 'TotLen Bwd Pkts', 'Fwd Pkt Len Max', 'Fwd Pkt Len Min',
    'Fwd Pkt Len Mean', 'Fwd Pkt Len Std', 'Bwd Pkt Len Max', 'Bwd Pkt Len Min',
    'Bwd Pkt Len Mean', 'Bwd Pkt Len Std', 'Flow Byts/s', 'Flow Pkts/s',
    'Flow IAT Mean', 'Flow IAT Std', 'Flow IAT Max', 'Flow IAT Min',
    'Fwd IAT Tot', 'Fwd IAT Mean', 'Fwd IAT Std', 'Fwd IAT Max', 'Fwd IAT Min',
    'Bwd IAT Tot', 'Bwd IAT Mean', 'Bwd IAT Std', 'Bwd IAT Max', 'Bwd IAT Min',
    'Fwd PSH Flags', 'Fwd URG Flags', 'Fwd Header Len', 'Bwd Header Len',
    'Fwd Pkts/s', 'Bwd Pkts/s', 'Pkt Len Min', 'Pkt Len Max', 'Pkt Len Mean',
    'Pkt Len Std', 'Pkt Len Var', 'FIN Flag Cnt', 'SYN Flag Cnt', 'RST Flag Cnt',
    'PSH Flag Cnt', 'ACK Flag Cnt', 'URG Flag Cnt', 'CWE Flag Count', 'ECE Flag Cnt',
    'Down/Up Ratio', 'Pkt Size Avg', 'Fwd Seg Size Avg', 'Bwd Seg Size Avg',
    'Subflow Fwd Pkts', 'Subflow Fwd Byts', 'Subflow Bwd Pkts', 'Subflow Bwd Byts',
    'Init Fwd Win Byts', 'Init Bwd Win Byts', 'Fwd Act Data Pkts', 'Fwd Seg Size Min',
    'Active Mean', 'Active Std', 'Active Max', 'Active Min',
    'Idle Mean', 'Idle Std', 'Idle Max', 'Idle Min',
    'Src Port',
]


class LiteTAE(nn.Module):
    def __init__(self, input_dim, d_model, nhead, enc_layers,
                 dec_layers, ffn_dim, latent_dim, group_size, dropout=0.0):
        super().__init__()
        self.g    = group_size
        self.d    = d_model
        self.lat  = latent_dim
        self.pad  = (-input_dim) % group_size
        self.pdim = input_dim + self.pad
        self.seq  = self.pdim // group_size
        self.inp  = input_dim

        self.pos     = nn.Embedding(self.seq, d_model)
        self.tok_enc = nn.Linear(group_size, d_model)

        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=ffn_dim,
            dropout=dropout, activation='gelu', batch_first=True, norm_first=True,
        )
        self.enc    = nn.TransformerEncoder(enc_layer, num_layers=enc_layers,
                                            norm=nn.LayerNorm(d_model))
        self.to_lat = nn.Linear(d_model * self.seq, latent_dim)

        self.from_lat = nn.Linear(latent_dim, d_model * self.seq)
        dec_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=ffn_dim,
            dropout=dropout, activation='gelu', batch_first=True, norm_first=True,
        )
        self.dec     = nn.TransformerEncoder(dec_layer, num_layers=dec_layers,
                                             norm=nn.LayerNorm(d_model))
        self.tok_dec = nn.Linear(d_model, group_size)

    def _tokenize(self, x):
        B = x.size(0)
        if self.pad > 0:
            x = torch.cat([x, x.new_zeros(B, self.pad)], dim=1)
        return x.view(B, self.seq, self.g)

    def encode(self, x):
        B = x.size(0)
        h = self.tok_enc(self._tokenize(x))
        h = h + self.pos(torch.arange(self.seq, device=x.device))
        z = self.to_lat(self.enc(h).reshape(B, -1))
        return z

    def decode(self, z):
        B = z.size(0)
        h = self.from_lat(z).view(B, self.seq, self.d)
        h = h + self.pos(torch.arange(self.seq, device=z.device))
        out = self.tok_dec(self.dec(h)).reshape(B, -1)
        return out[:, :self.inp]

    def forward(self, x):
        z = self.encode(x)
        return self.decode(z), z


# ══════════════════════════════════════════════════════════════════
#  MODEL LOADER
# ══════════════════════════════════════════════════════════════════
class ModelLoader:
    def __init__(self, model_path: str, device: str = 'cpu'):
        self.device = torch.device(device)
        print(f"{C.CYAN}⏳ Loading model: {model_path}{C.R}")

        ck = torch.load(model_path, map_location=self.device, weights_only=False)
        cfg = ck['config']

        saved_cols = ck.get('feat_cols', MODEL_FEATURES)
        if saved_cols != MODEL_FEATURES:
            print(f"{C.YELLOW}⚠  Feature mismatch — using saved feat_cols{C.R}")
        self.feat_cols = saved_cols

        self.scaler_c  = np.array(ck['scaler_c'], dtype=np.float32)
        self.scaler_s  = np.array(ck['scaler_s'], dtype=np.float32)
        self.clip_val  = float(cfg.get('CLIP_VAL', 10.0))
        self.threshold = float(ck['threshold'])
        self.input_dim = int(ck['input_dim'])

        self.model = LiteTAE(
            input_dim  = self.input_dim,
            d_model    = cfg['D_MODEL'],
            nhead      = cfg['NHEAD'],
            enc_layers = cfg['ENC_LAYERS'],
            dec_layers = cfg['DEC_LAYERS'],
            ffn_dim    = cfg['FFN_DIM'],
            latent_dim = cfg['LATENT_DIM'],
            group_size = cfg['GROUP_SIZE'],
            dropout    = 0.0,
        ).to(self.device)

        self.model.load_state_dict(ck['state_dict'])
        self.model.eval()

        print(f"{C.GREEN}✅ Model loaded successfully{C.R}")

    def _safe(self, v):
        if v is None: return 0.0
        try:
            f = float(v)
            return 0.0 if (math.isnan(f) or math.isinf(f)) else f
        except Exception:
            return 0.0

    def preprocess(self, feat_dict: dict) -> np.ndarray:
        vec = np.array([self._safe(feat_dict.get(c, 0.0))
                        for c in self.feat_cols], dtype=np.float32)
        vec = (vec - self.scaler_c) / (self.scaler_s + 1e-9)
        vec = np.clip(vec, -self.clip_val, self.clip_val)
        return vec

    @torch.no_grad()
    def score_batch(self, vecs: list) -> list:
        arr = np.stack(vecs, axis=0)
        t   = torch.from_numpy(arr).to(self.device)
        recon, _ = self.model(t)
        errs = ((recon - t) ** 2).mean(dim=1).cpu().numpy()
        return errs.tolist()


class FlowStats:
    IDLE_THR = 1.0

    def __init__(self, src_ip, dst_ip, src_port, dst_port, proto, ts):
        self.src_ip   = src_ip
        self.dst_ip   = dst_ip
        self.src_port = src_port
        self.dst_port = dst_port
        self.proto    = proto
        self.t_start  = ts
        self.t_last   = ts

        self.fwd_plen = []
        self.bwd_plen = []
        self.fwd_ts   = []
        self.bwd_ts   = []

        self.fwd_hdr  = 0
        self.bwd_hdr  = 0

        self.init_fwd_win = -1
        self.init_bwd_win = -1

        self.flag_fin = self.flag_syn = self.flag_rst = 0
        self.flag_psh = self.flag_ack = self.flag_urg = 0
        self.flag_cwe = self.flag_ece = 0
        self.fwd_psh  = 0
        self.fwd_urg  = 0

        self._act_start   = ts
        self._last_active = ts
        self.active_times = []
        self.idle_times   = []

    def add(self, ts: float, payload_len: int, hdr_len: int,
            is_fwd: bool, tcp_flags: int = 0, window: int = 0):

        gap = ts - self._last_active
        if gap > self.IDLE_THR and (self.fwd_ts or self.bwd_ts):
            self.idle_times.append(gap)
            self.active_times.append(self._last_active - self._act_start)
            self._act_start = ts
        self._last_active = ts
        self.t_last = ts

        if is_fwd:
            self.fwd_plen.append(payload_len)
            self.fwd_ts.append(ts)
            self.fwd_hdr += hdr_len
            if self.init_fwd_win < 0: self.init_fwd_win = window
            if tcp_flags & 0x08: self.fwd_psh += 1
            if tcp_flags & 0x20: self.fwd_urg += 1
        else:
            self.bwd_plen.append(payload_len)
            self.bwd_ts.append(ts)
            self.bwd_hdr += hdr_len
            if self.init_bwd_win < 0: self.init_bwd_win = window

        if tcp_flags & 0x01: self.flag_fin += 1
        if tcp_flags & 0x02: self.flag_syn += 1
        if tcp_flags & 0x04: self.flag_rst += 1
        if tcp_flags & 0x08: self.flag_psh += 1
        if tcp_flags & 0x10: self.flag_ack += 1
        if tcp_flags & 0x20: self.flag_urg += 1
        if tcp_flags & 0x40: self.flag_cwe += 1
        if tcp_flags & 0x80: self.flag_ece += 1

    @staticmethod
    def _stats(lst):
        if not lst:
            return 0.0, 0.0, 0.0, 0.0
        a = np.asarray(lst, dtype=np.float64)
        return float(a.min()), float(a.max()), float(a.mean()), float(a.std())

    @staticmethod
    def _iat(ts_list):
        if len(ts_list) < 2:
            return []
        s = sorted(ts_list)
        return [s[i+1] - s[i] for i in range(len(s)-1)]

    def extract(self) -> dict:
        US = 1e6
        all_plen = self.fwd_plen + self.bwd_plen
        all_ts   = sorted(self.fwd_ts + self.bwd_ts)

        n_fwd = len(self.fwd_plen)
        n_bwd = len(self.bwd_plen)
        n_all = n_fwd + n_bwd

        sum_fwd = sum(self.fwd_plen)
        sum_bwd = sum(self.bwd_plen)
        sum_all = sum_fwd + sum_bwd

        duration  = max(self.t_last - self.t_start, 0.0)
        dur_s     = max(duration, 1e-9)

        flow_byts_s = sum_all / dur_s
        flow_pkts_s = n_all   / dur_s
        fwd_pkts_s  = n_fwd   / dur_s
        bwd_pkts_s  = n_bwd   / dur_s

        fmn, fmx, fmean, fstd = self._stats(self.fwd_plen)
        bmn, bmx, bmean, bstd = self._stats(self.bwd_plen)
        pmn, pmx, pmean, pstd = self._stats(all_plen)

        iat_all = self._iat(all_ts)
        fia_mean, fia_std, fia_mx, fia_mn = self._stats(iat_all)

        iat_fwd = self._iat(self.fwd_ts)
        fia_tot_f = sum(iat_fwd) if iat_fwd else 0.0
        fia_mn_f, fia_mx_f, fia_mean_f, fia_std_f = self._stats(iat_fwd)

        iat_bwd = self._iat(self.bwd_ts)
        bia_tot  = sum(iat_bwd) if iat_bwd else 0.0
        bia_mn, bia_mx, bia_mean, bia_std = self._stats(iat_bwd)

        self.active_times.append(self._last_active - self._act_start)
        act_mn, act_mx, act_mean, act_std = self._stats(self.active_times)
        idl_mn, idl_mx, idl_mean, idl_std = self._stats(self.idle_times)

        down_up = (n_bwd / n_fwd) if n_fwd > 0 else 0.0

        return {
            'Dst Port'        : self.dst_port,
            'Src Port'        : self.src_port,
            'Protocol'        : self.proto,
            'Flow Duration'   : duration * US,
            'Tot Fwd Pkts'    : n_fwd,
            'Tot Bwd Pkts'    : n_bwd,
            'TotLen Fwd Pkts' : sum_fwd,
            'TotLen Bwd Pkts' : sum_bwd,
            'Fwd Pkt Len Max' : fmx,
            'Fwd Pkt Len Min' : fmn,
            'Fwd Pkt Len Mean': fmean,
            'Fwd Pkt Len Std' : fstd,
            'Bwd Pkt Len Max' : bmx,
            'Bwd Pkt Len Min' : bmn,
            'Bwd Pkt Len Mean': bmean,
            'Bwd Pkt Len Std' : bstd,
            'Flow Byts/s'     : flow_byts_s,
            'Flow Pkts/s'     : flow_pkts_s,
            'Flow IAT Mean'   : fia_mean  * US,
            'Flow IAT Std'    : fia_std   * US,
            'Flow IAT Max'    : fia_mx    * US,
            'Flow IAT Min'    : fia_mn    * US,
            'Fwd IAT Tot'     : fia_tot_f * US,
            'Fwd IAT Mean'    : fia_mean_f* US,
            'Fwd IAT Std'     : fia_std_f * US,
            'Fwd IAT Max'     : fia_mx_f  * US,
            'Fwd IAT Min'     : fia_mn_f  * US,
            'Bwd IAT Tot'     : bia_tot   * US,
            'Bwd IAT Mean'    : bia_mean  * US,
            'Bwd IAT Std'     : bia_std   * US,
            'Bwd IAT Max'     : bia_mx    * US,
            'Bwd IAT Min'     : bia_mn    * US,
            'Fwd PSH Flags'   : self.fwd_psh,
            'Fwd URG Flags'   : self.fwd_urg,
            'Fwd Header Len'  : self.fwd_hdr,
            'Bwd Header Len'  : self.bwd_hdr,
            'Fwd Pkts/s'      : fwd_pkts_s,
            'Bwd Pkts/s'      : bwd_pkts_s,
            'Pkt Len Min'     : pmn,
            'Pkt Len Max'     : pmx,
            'Pkt Len Mean'    : pmean,
            'Pkt Len Std'     : pstd,
            'Pkt Len Var'     : pstd ** 2,
            'FIN Flag Cnt'    : self.flag_fin,
            'SYN Flag Cnt'    : self.flag_syn,
            'RST Flag Cnt'    : self.flag_rst,
            'PSH Flag Cnt'    : self.flag_psh,
            'ACK Flag Cnt'    : self.flag_ack,
            'URG Flag Cnt'    : self.flag_urg,
            'CWE Flag Count'  : self.flag_cwe,
            'ECE Flag Cnt'    : self.flag_ece,
            'Down/Up Ratio'   : down_up,
            'Pkt Size Avg'    : (sum_all / n_all) if n_all > 0 else 0.0,
            'Fwd Seg Size Avg': fmean,
            'Bwd Seg Size Avg': bmean,
            'Subflow Fwd Pkts': n_fwd,
            'Subflow Fwd Byts': sum_fwd,
            'Subflow Bwd Pkts': n_bwd,
            'Subflow Bwd Byts': sum_bwd,
            'Init Fwd Win Byts': max(0, self.init_fwd_win),
            'Init Bwd Win Byts': max(0, self.init_bwd_win),
            'Fwd Act Data Pkts': sum(1 for l in self.fwd_plen if l > 0),
            'Fwd Seg Size Min' : fmn,
            'Active Mean'      : act_mean * US,
            'Active Std'       : act_std  * US,
            'Active Max'       : act_mx   * US,
            'Active Min'       : act_mn   * US,
            'Idle Mean'        : idl_mean * US,
            'Idle Std'         : idl_std  * US,
            'Idle Max'         : idl_mx   * US,
            'Idle Min'         : idl_mn   * US,
        }

    @property
    def n_pkts(self):
        return len(self.fwd_plen) + len(self.bwd_plen)


class FlowTracker:
    def __init__(self, timeout: float = 5.0):
        self.timeout  = timeout
        self._flows   = {}
        self._ready   = []
        self._lock    = threading.Lock()

    @staticmethod
    def _key(src_ip, dst_ip, src_port, dst_port, proto):
        a, b = (src_ip, src_port), (dst_ip, dst_port)
        if a > b: a, b = b, a
        return (a[0], b[0], a[1], b[1], proto)

    def ingest(self, ts, src_ip, dst_ip, src_port, dst_port,
               proto, payload_len, hdr_len, tcp_flags=0, window=0):
        key    = self._key(src_ip, dst_ip, src_port, dst_port, proto)
        is_fwd = ((src_ip, src_port) >= (dst_ip, dst_port))

        with self._lock:
            if key not in self._flows:
                self._flows[key] = FlowStats(src_ip, dst_ip, src_port, dst_port, proto, ts)

            flow = self._flows[key]
            flow.add(ts, payload_len, hdr_len, is_fwd, tcp_flags, window)

            if tcp_flags & 0x04:
                self._ready.append(self._flows.pop(key))
            elif (tcp_flags & 0x01) and (tcp_flags & 0x10):
                self._ready.append(self._flows.pop(key))

    def expire(self, now: float) -> int:
        with self._lock:
            expired_keys = [k for k, f in self._flows.items()
                            if (now - f.t_last) >= self.timeout]
            for k in expired_keys:
                self._ready.append(self._flows.pop(k))
        return len(expired_keys)

    def pop_ready(self) -> list:
        with self._lock:
            out = self._ready[:]
            self._ready.clear()
        return out



class ThresholdCalibrator:
    def __init__(self, k_factor: float = 3.0):
        self.errors = []
        self.k_factor = k_factor

    def add_error(self, err: float):
        self.errors.append(err)

    def calculate(self) -> float:
        if len(self.errors) < 20:
            return 0.0
        mu = np.mean(self.errors)
        sigma = np.std(self.errors)
        return float(mu + (self.k_factor * sigma))

class Classifier:
    def __init__(self, threshold: float = 0.0):
        self.threshold = threshold

    def _port_factor(self, flow: FlowStats) -> float:
        dp = flow.dst_port
        sp = flow.src_port
        n  = flow.n_pkts
        factor = 1.0

        if dp in NORMAL_PORTS or sp in NORMAL_PORTS:
            factor *= 0.85

        if dp in HIGH_RISK_PORTS or sp in HIGH_RISK_PORTS:
            factor *= 1.5

        if n > 4:
            syn_r = flow.flag_syn / n
            rst_r = flow.flag_rst / n
            if syn_r > 0.7: factor *= 1.4
            if rst_r > 0.5: factor *= 1.3

        if (dp == 53 or sp == 53) and flow.proto == 17:
            if sum(flow.bwd_plen) > 10 * sum(flow.fwd_plen) and n > 3:
                factor *= 1.6

        if (dp in (80, 443, 8080, 8443)) and n < 200:
            factor *= 0.9

        if dp == 22 and flow.flag_syn > 5:
            factor *= 1.3

        return factor

    def classify(self, error: float, flow: FlowStats):
        factor   = self._port_factor(flow)
        adj      = error * factor

        if adj >= self.threshold:
            verdict = 'ATTACK'
        else:
            verdict = 'NORMAL'

        return verdict, adj, factor


class Reporter:
    _COLORS = {
        'NORMAL': C.GREEN,
        'ATTACK': C.RED + C.B,
    }
    _ICONS = {
        'NORMAL': '✅', 'ATTACK': '🚨',
    }

    def __init__(self, log_path: str, show_normal: bool = False):
        self.show_normal = show_normal
        self.counts      = {'NORMAL': 0, 'ATTACK': 0}
        self.t_start     = time.time()

        self._log = logging.getLogger('IDS')
        if log_path:
            fh = logging.FileHandler(log_path)
            fh.setFormatter(logging.Formatter('%(asctime)s | %(message)s'))
            self._log.addHandler(fh)
            self._log.setLevel(logging.INFO)

    def report(self, flow: FlowStats, raw_err: float,
               verdict: str, adj_err: float, factor: float):
        self.counts[verdict] += 1

        if verdict == 'NORMAL' and not self.show_normal:
            return

        ts    = datetime.now().strftime('%H:%M:%S.%f')[:-3]
        proto = PROTO_NAMES.get(flow.proto, str(flow.proto))
        svc   = NORMAL_PORTS.get(flow.dst_port,
                HIGH_RISK_PORTS.get(flow.dst_port,
                NORMAL_PORTS.get(flow.src_port,
                HIGH_RISK_PORTS.get(flow.src_port, ''))))
        svc_s  = f" [{svc}]" if svc else ''
        n_pkts = flow.n_pkts
        color  = self._COLORS[verdict]
        icon   = self._ICONS[verdict]
        dur_ms = (flow.t_last - flow.t_start) * 1000

        line = (
            f"{color}[{ts}] {icon} {verdict:10s}│"
            f" err={raw_err:.5f} adj={adj_err:.5f} (×{factor:.2f})"
            f"│ {proto} {flow.src_ip}:{flow.src_port}"
            f" → {flow.dst_ip}:{flow.dst_port}{svc_s}"
            f"│ pkts={n_pkts} dur={dur_ms:.0f}ms{C.R}"
        )
        print(line)

        self._log.info(
            f"{verdict} | raw={raw_err:.6f} adj={adj_err:.6f} factor={factor:.3f}"
            f"| {proto} {flow.src_ip}:{flow.src_port}"
            f"→{flow.dst_ip}:{flow.dst_port}{svc_s}"
            f"| pkts={n_pkts} dur={dur_ms:.0f}ms"
        )

    def print_summary(self):
        total   = sum(self.counts.values())
        elapsed = time.time() - self.t_start
        fps     = total / max(elapsed, 1.0)

        print(f"\n{C.CYAN}{'═'*58}{C.R}")
        print(f"{C.B}  IDS Summary  ({elapsed:.1f}s | {fps:.1f} flows/s){C.R}")
        print(f"{C.CYAN}{'─'*58}")
        for v in ['NORMAL', 'ATTACK']:
            n   = self.counts[v]
            pct = 100 * n / total if total else 0
            bar = '█' * int(30 * n / total) if total else ''
            c   = self._COLORS[v]
            print(f"  {c}{v:10s}{C.R} : {n:7,}  ({pct:5.1f}%)  {C.CYAN}{bar}{C.R}")
        print(f"{C.CYAN}{'─'*58}")
        print(f"  Total flows processed : {total:,}{C.R}")
        print(f"{C.CYAN}{'═'*58}{C.R}\n")



class LiveIDS:
    def __init__(self, args):
        dev = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.loader = ModelLoader(args.model, device=dev)

        
        self.is_calibrating = True
        self.cal_limit = args.cal_flows
        self.calibrator = ThresholdCalibrator(k_factor=args.k_factor)
        
        print(f"\n{C.YELLOW}⚙️  [System State] Calibration Mode Active.")
        print(f"   Waiting for the first {self.cal_limit} flows to dynamically adapt to network traffic...{C.R}\n")

        self.tracker    = FlowTracker(timeout=args.timeout)
        self.classifier = Classifier(threshold=0.0) # سيتحدد تلقائياً بعد المعايرة
        self.reporter   = Reporter(args.log, show_normal=args.show_normal)

        self.min_pkts   = args.min_pkts
        self.batch_size = 64
        self._running   = True

    def _on_packet(self, pkt):
        try:
            if not pkt.haslayer(IP): return
            ip      = pkt[IP]
            ts      = float(pkt.time)
            src_ip  = ip.src
            dst_ip  = ip.dst
            proto   = int(ip.proto)
            ip_hlen = int(ip.ihl) * 4 if hasattr(ip, 'ihl') else 20
            total   = int(ip.len) if hasattr(ip, 'len') else len(bytes(pkt))

            tcp_flags = src_port = dst_port = t_hlen = window = 0

            if pkt.haslayer(TCP):
                tcp       = pkt[TCP]
                src_port  = int(tcp.sport)
                dst_port  = int(tcp.dport)
                tcp_flags = int(tcp.flags)
                window    = int(tcp.window)
                t_hlen    = int(tcp.dataofs) * 4 if hasattr(tcp, 'dataofs') else 20
            elif pkt.haslayer(UDP):
                udp      = pkt[UDP]
                src_port = int(udp.sport)
                dst_port = int(udp.dport)
                t_hlen   = 8
            elif pkt.haslayer(ICMP):
                t_hlen = 8

            hdr_len     = ip_hlen + t_hlen
            payload_len = max(0, total - hdr_len)

            self.tracker.ingest(
                ts=ts, src_ip=src_ip, dst_ip=dst_ip,
                src_port=src_port, dst_port=dst_port,
                proto=proto, payload_len=payload_len,
                hdr_len=hdr_len, tcp_flags=tcp_flags, window=window,
            )
        except Exception:
            pass

    def _score_flows(self, flows: list):
        vecs  = []
        valid = []

        for f in flows:
            if f.n_pkts < self.min_pkts: continue
            feat = f.extract()
            vec  = self.loader.preprocess(feat)
            vecs.append(vec)
            valid.append(f)

        if not vecs: return

        for i in range(0, len(vecs), self.batch_size):
            batch_v = vecs[i:i + self.batch_size]
            batch_f = valid[i:i + self.batch_size]
            try:
                errors = self.loader.score_batch(batch_v)
            except Exception as e:
                print(f"{C.GRAY}[score error] {e}{C.R}")
                continue

            for flow, err in zip(batch_f, errors):
                if self.is_calibrating:
                    self.calibrator.add_error(err)
                    current_count = len(self.calibrator.errors)
                    print(f"{C.YELLOW}⏳ [Calibration] Logged {current_count}/{self.cal_limit} flows...{C.R}", end="\r")
                    
                    if current_count >= self.cal_limit:
                        dyn_thr = self.calibrator.calculate()
                        if dyn_thr > 0:
                            self.classifier.threshold = dyn_thr
                            self.is_calibrating = False
                            print(f"\n\n{C.GREEN} Calibration finished! Dynamic Threshold established: {dyn_thr:.6f}{C.R}")
                            print(f"{C.GRAY}   Traffic below this limit is NORMAL, above is an ATTACK.{C.R}\n")
                        else:
                            # Fallback safe limit
                            self.classifier.threshold = self.loader.threshold
                            self.is_calibrating = False
                else:
                    verdict, adj, factor = self.classifier.classify(err, flow)
                    self.reporter.report(flow, err, verdict, adj, factor)

    def _expiry_loop(self):
        while self._running:
            time.sleep(1.0)
            self.tracker.expire(time.time())
            ready = self.tracker.pop_ready()
            if ready:
                self._score_flows(ready)

    def run_live(self, iface=None):
        print(f"{C.B}{C.BLUE}{'═'*58}")
        print(f"  🔍  Live Network IDS — Adaptive Autoencoder Engine")
        print(f"{'═'*58}{C.R}")
        print(f"{C.CYAN}  Interface         : {iface or 'system default'}")
        print(f"  Calibration Limit : {self.cal_limit} flows")
        print(f"  Sensitivity (K)   : {self.calibrator.k_factor}")
        print(f"\n{C.GRAY}  Press Ctrl+C to stop...{C.R}\n")

        t = threading.Thread(target=self._expiry_loop, daemon=True)
        t.start()

        try:
            sniff(iface=iface, prn=self._on_packet, store=False, filter="ip")
        except KeyboardInterrupt:
            print(f"\n{C.YELLOW}Stopping capture loop...{C.R}")
        finally:
            self._running = False
            if self.is_calibrating:
                # إذا تم الإغلاق قبل انتهاء عدد التدفقات المطلوبة، احسب العتبة بما لديك
                dyn_thr = self.calibrator.calculate() or self.loader.threshold
                self.classifier.threshold = dyn_thr
                print(f"\n{C.YELLOW}⚠ Interrupted early. Set Threshold from partial calibration: {dyn_thr:.6f}{C.R}")
            self.tracker.expire(time.time() + 1e9)
            self._score_flows(self.tracker.pop_ready())
            self.reporter.print_summary()

    def run_pcap(self, pcap_path: str):
        from scapy.all import rdpcap
        print(f"{C.CYAN}📂 Reading PCAP file: {pcap_path}{C.R}")
        pkts = rdpcap(pcap_path)
        print(f"   Loaded {len(pkts):,} packets\n")

        for pkt in pkts:
            self._on_packet(pkt)

        max_ts = max((float(p.time) for p in pkts if hasattr(p, 'time')), default=0)
        self.tracker.expire(max_ts + 1e9)
        self._score_flows(self.tracker.pop_ready())
        self.reporter.print_summary()


def build_parser():
    p = argparse.ArgumentParser(
        prog='ids_live.py',
        description='Adaptive Real-Time IDS — Transformer Autoencoder',
    )
    p.add_argument('--model',    required=True, help=' pt')
    p.add_argument('--iface',    default=None, help='اسم الـ interface ')
    p.add_argument('--pcap',     default=None, help='analysis pcap ')
    p.add_argument('--timeout',  type=float, default=5.0, help=' ـflow ')
    p.add_argument('--min-pkts', type=int,   default=3, help='ـ flow ')
    p.add_argument('--cal-flows', type=int,  default=200, help='number of flows to gain perfect threshold')
    p.add_argument('--k-factor',  type=float, default=3.0, help=' (the highest = false positives the slowest)')
    p.add_argument('--log',      default='ids_detections.log', help='')
    p.add_argument('--show-normal', action='store_true', help='ـ terminal')
    return p


def main():
    parser = build_parser()
    args   = parser.parse_args()

    if not os.path.isfile(args.model):
        sys.exit(f"model not exist: {args.model}")

    ids = LiveIDS(args)

    if args.pcap:
        if not os.path.isfile(args.pcap):
            sys.exit(f" pcap not exsit: {args.pcap}")
        ids.run_pcap(args.pcap)
    else:
        ids.run_live(iface=args.iface)


if __name__ == '__main__':
    main()
