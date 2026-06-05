#!/usr/bin/env python3
"""
Zero-Day Attack Detection System
================================
This script converts pfSense pcap captures to the exact same data format as the trained model,
runs inference to detect attacks, and sends results to a user-provided API.

المهم: بيحول الـ pcap لنفس شكل الداتا اللي ادرب عليها المودل - بنفس الأعمدة والترتيب

Author: MiniMax Agent
"""

import numpy as np
import pandas as pd
import pickle
import os
import json
import requests
from datetime import datetime
from scapy.all import rdpcap, IP, TCP, UDP, ICMP, Ether
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# CONFIGURATION - عدل القيم دي حسب اللي عندك
# ============================================================================
MODEL_DIR = "user_input_files"
MODEL_PATH = os.path.join(MODEL_DIR, "stm_full_model.keras")
SCALER_PATH = os.path.join(MODEL_DIR, "scaler.pkl")
THRESHOLD_PATH = os.path.join(MODEL_DIR, "threshold.npy")

# API Configuration - استبدل بالرابط الحقيقي بتاعك
API_URL = "https://your-api-endpoint.com/api/detection-results"
API_KEY = ""  # لو عندك API key حطه هنا

# Default pcap file
DEFAULT_PCAP = "user_input_files/packetcapture-em0-20260507233338.pcap"

# ============================================================================
# CICIDS2018 EXACT FEATURES - نفس الترتيب اللي اتدرب عليه المودل
# ============================================================================
FEATURE_NAMES = [
    'Dst Port', 'Protocol', 'Flow Duration', 'Tot Fwd Pkts', 'Tot Bwd Pkts',
    'TotLen Fwd Pkts', 'TotLen Bwd Pkts', 'Fwd Pkt Len Max', 'Fwd Pkt Len Min',
    'Fwd Pkt Len Mean', 'Fwd Pkt Len Std', 'Bwd Pkt Len Max', 'Bwd Pkt Len Min',
    'Bwd Pkt Len Mean', 'Bwd Pkt Len Std', 'Flow Byts/s', 'Flow Pkts/s',
    'Flow IAT Mean', 'Flow IAT Std', 'Flow IAT Max', 'Flow IAT Min',
    'Fwd IAT Tot', 'Fwd IAT Mean', 'Fwd IAT Std', 'Fwd IAT Max', 'Fwd IAT Min',
    'Bwd IAT Tot', 'Bwd IAT Mean', 'Bwd IAT Std', 'Bwd IAT Max', 'Bwd IAT Min',
    'Fwd PSH Flags', 'Bwd PSH Flags', 'Fwd URG Flags', 'Bwd URG Flags',
    'Fwd Header Len', 'Bwd Header Len', 'Fwd Pkts/s', 'Bwd Pkts/s', 'Pkt Len Min',
    'Pkt Len Max', 'Pkt Len Mean', 'Pkt Len Std', 'Pkt Len Var', 'FIN Flag Cnt',
    'SYN Flag Cnt', 'RST Flag Cnt', 'PSH Flag Cnt', 'ACK Flag Cnt',
    'URG Flag Cnt', 'CWE Flag Count', 'ECE Flag Cnt', 'Down/Up Ratio',
    'Pkt Size Avg', 'Fwd Seg Size Avg', 'Bwd Seg Size Avg', 'Fwd Byts/b Avg',
    'Fwd Pkts/b Avg', 'Fwd Blk Rate Avg', 'Bwd Byts/b Avg', 'Bwd Pkts/b Avg',
    'Bwd Blk Rate Avg', 'Subflow Fwd Pkts', 'Subflow Fwd Byts',
    'Subflow Bwd Pkts', 'Subflow Bwd Byts', 'Init Fwd Win Byts',
    'Init Bwd Win Byts', 'Fwd Act Data Pkts', 'Fwd Seg Size Min', 'Active Mean',
    'Active Std', 'Active Max', 'Active Min', 'Idle Mean', 'Idle Std', 'Idle Max',
    'Idle Min'
]

def extract_flow_features_from_packets(packets):
    """
    تحويل الـ pcap لبيانات بنفس شكل CICFlowMeter output - نفس الـ features
    كل حزمة هتتحول لصف واحد في الـ DataFrame بنفس الأعمدة المطلوبة
    """
    records = []

    for i, pkt in enumerate(packets):
        if not pkt.haslayer(IP):
            continue

        ip_layer = pkt['IP']

        # استخراج المنافذ
        src_port = 0
        dst_port = 0
        protocol = 0

        if pkt.haslayer(TCP):
            src_port = pkt['TCP'].sport
            dst_port = pkt['TCP'].dport
            protocol = 6  # TCP
        elif pkt.haslayer(UDP):
            src_port = pkt['UDP'].sport
            dst_port = pkt['UDP'].dport
            protocol = 17  # UDP
        elif pkt.haslayer(ICMP):
            protocol = 1  # ICMP

        # طول الحزمة
        pkt_len = len(pkt)

        # استخراج TCP flags
        fin_cnt = 0
        syn_cnt = 0
        rst_cnt = 0
        psh_cnt = 0
        ack_cnt = 0
        urg_cnt = 0

        if pkt.haslayer(TCP):
            tcp = pkt['TCP']
            flags_str = str(tcp.flags)
            fin_cnt = 1 if 'F' in flags_str else 0
            syn_cnt = 1 if 'S' in flags_str else 0
            rst_cnt = 1 if 'R' in flags_str else 0
            psh_cnt = 1 if 'P' in flags_str else 0
            ack_cnt = 1 if 'A' in flags_str else 0
            urg_cnt = 1 if 'U' in flags_str else 0

        # حساب الـ header length
        fwd_header_len = 0
        bwd_header_len = 0
        if pkt.haslayer(TCP):
            fwd_header_len = pkt['TCP'].dataofs * 4 if hasattr(pkt['TCP'], 'dataofs') else 20

        # الـ window size
        init_fwd_win = 0
        init_bwd_win = 0
        if pkt.haslayer(TCP) and hasattr(pkt['TCP'], 'window'):
            init_fwd_win = pkt['TCP'].window

        # بناء الـ record بالأعمدة المطلوبة بالتحديد
        record = {
            # الأعمدة الثابتة
            'Dst Port': dst_port,
            'Protocol': protocol,
            'Flow Duration': 0,  # حزمة واحدة - لا يوجد duration
            'Tot Fwd Pkts': 1,
            'Tot Bwd Pkts': 0,
            'TotLen Fwd Pkts': pkt_len,
            'TotLen Bwd Pkts': 0,

            # Forward packet lengths
            'Fwd Pkt Len Max': pkt_len,
            'Fwd Pkt Len Min': pkt_len,
            'Fwd Pkt Len Mean': pkt_len,
            'Fwd Pkt Len Std': 0,

            # Backward packet lengths
            'Bwd Pkt Len Max': 0,
            'Bwd Pkt Len Min': 0,
            'Bwd Pkt Len Mean': 0,
            'Bwd Pkt Len Std': 0,

            # Flow rates
            'Flow Byts/s': pkt_len,
            'Flow Pkts/s': 1,

            # Flow IAT
            'Flow IAT Mean': 0,
            'Flow IAT Std': 0,
            'Flow IAT Max': 0,
            'Flow IAT Min': 0,

            # Forward IAT
            'Fwd IAT Tot': 0,
            'Fwd IAT Mean': 0,
            'Fwd IAT Std': 0,
            'Fwd IAT Max': 0,
            'Fwd IAT Min': 0,

            # Backward IAT
            'Bwd IAT Tot': 0,
            'Bwd IAT Mean': 0,
            'Bwd IAT Std': 0,
            'Bwd IAT Max': 0,
            'Bwd IAT Min': 0,

            # Flags
            'Fwd PSH Flags': psh_cnt,
            'Bwd PSH Flags': 0,
            'Fwd URG Flags': urg_cnt,
            'Bwd URG Flags': 0,
            'Fwd Header Len': fwd_header_len,
            'Bwd Header Len': bwd_header_len,
            'Fwd Pkts/s': 1,
            'Bwd Pkts/s': 0,

            # Packet length stats
            'Pkt Len Min': pkt_len,
            'Pkt Len Max': pkt_len,
            'Pkt Len Mean': pkt_len,
            'Pkt Len Std': 0,
            'Pkt Len Var': 0,

            # TCP Flags counts
            'FIN Flag Cnt': fin_cnt,
            'SYN Flag Cnt': syn_cnt,
            'RST Flag Cnt': rst_cnt,
            'PSH Flag Cnt': psh_cnt,
            'ACK Flag Cnt': ack_cnt,
            'URG Flag Cnt': urg_cnt,
            'CWE Flag Count': 0,
            'ECE Flag Cnt': 0,

            # Ratios
            'Down/Up Ratio': pkt_len if syn_cnt == 0 else 0,
            'Pkt Size Avg': pkt_len,
            'Fwd Seg Size Avg': pkt_len,
            'Bwd Seg Size Avg': 0,
            'Fwd Byts/b Avg': 0,
            'Fwd Pkts/b Avg': 0,
            'Fwd Blk Rate Avg': 0,
            'Bwd Byts/b Avg': 0,
            'Bwd Pkts/b Avg': 0,
            'Bwd Blk Rate Avg': 0,

            # Subflows
            'Subflow Fwd Pkts': 1,
            'Subflow Fwd Byts': pkt_len,
            'Subflow Bwd Pkts': 0,
            'Subflow Bwd Byts': 0,

            # Window sizes
            'Init Fwd Win Byts': init_fwd_win,
            'Init Bwd Win Byts': init_bwd_win,

            # Active/Idle
            'Fwd Act Data Pkts': psh_cnt,
            'Fwd Seg Size Min': pkt_len,
            'Active Mean': 0,
            'Active Std': 0,
            'Active Max': 0,
            'Active Min': 0,
            'Idle Mean': 0,
            'Idle Std': 0,
            'Idle Max': 0,
            'Idle Min': 0,
        }

        records.append(record)

    # تحويل لـ DataFrame مع ترتيب الأعمدة الصحيح
    df = pd.DataFrame(records)
    df = df[FEATURE_NAMES]  # تأكد من نفس الترتيب
    return df


def clean_and_preprocess_data(df, scaler):
    """
    تنظيف البيانات - نفس اللي عملته في الـ training
    """
    df = df.copy()

    # تحويل الأعمدة لـ numeric
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    # استبدال الـ infinite values
    df = df.replace([np.inf, -np.inf], np.nan)

    # ملء الـ NaN بالـ median
    for col in df.columns:
        if df[col].isna().any():
            median_val = df[col].median()
            if pd.isna(median_val):
                median_val = 0
            df[col] = df[col].fillna(median_val)

    # عمل scale باستخدام الـ scaler اللي اتدرب عليه
    X_scaled = scaler.transform(df)

    return X_scaled


def load_model_and_scaler():
    """تحميل المودل والـ scaler والـ threshold"""
    from tensorflow.keras.models import load_model

    # تحميل scaler
    with open(SCALER_PATH, 'rb') as f:
        scaler = pickle.load(f)

    # تحميل المودل
    model = load_model(MODEL_PATH)

    # تحميل threshold
    threshold = np.load(THRESHOLD_PATH)

    return model, scaler, threshold


def detect_anomalies(model, X_scaled, threshold):
    """تشغيل المودل للكشف عن الهجمات"""
    # reshaping للـ LSTM
    X_reshaped = X_scaled.reshape(-1, 1, X_scaled.shape[1])

    # التنبؤ
    reconstructions = model.predict(X_reshaped, verbose=0, batch_size=256)

    # حساب الـ reconstruction error
    mse = np.mean(np.power(X_reshaped - reconstructions, 2), axis=(1, 2))

    # تصنيف بناء على الـ threshold
    predictions = (mse > threshold).astype(int)

    return predictions, mse


def analyze_attack_type(prediction, mse, pkt_info, threshold):
    """تحليل نوع الهجوم المحتمل"""
    if prediction == 0:
        return "Benign"

    # heuristic analysis بناءً على خصائص الحزمة
    attack_type = "Suspicious Activity"

    if pkt_info:
        syn = pkt_info.get('syn_count', 0)
        rst = pkt_info.get('rst_count', 0)
        psh = pkt_info.get('psh_count', 0)
        pkt_len = pkt_info.get('length', 0)

        if syn == 1 and rst == 0:
            attack_type = "Possible SYN Scan"
        elif rst == 1:
            attack_type = "Possible RST Attack"
        elif psh == 1 and pkt_len > 1000:
            attack_type = "Possible Data Exfiltration"
        elif mse > threshold * 100:
            attack_type = "High Anomaly Score"

    return attack_type


def generate_report(predictions, mse, packets, df_features, threshold):
    """توليد تقرير مفصل"""
    total = len(predictions)
    attacks = int(np.sum(predictions))
    benign = total - attacks

    report = {
        "timestamp": datetime.now().isoformat(),
        "model_info": {
            "threshold": float(threshold),
            "total_features": len(FEATURE_NAMES)
        },
        "summary": {
            "total_packets": total,
            "attacks_detected": attacks,
            "benign_packets": benign,
            "attack_rate_percent": round(attacks/total*100, 2) if total > 0 else 0
        },
        "detections": [],
        "overall_status": "CLEAN" if attacks == 0 else "ALERT"
    }

    # تفاصيل كل هجوم
    for i, (pred, err) in enumerate(zip(predictions, mse)):
        if pred == 1:
            pkt = packets[i] if i < len(packets) else None

            pkt_info = {}
            if pkt and pkt.haslayer(IP):
                pkt_info = {
                    "src_ip": pkt['IP'].src,
                    "dst_ip": pkt['IP'].dst,
                    "length": len(pkt),
                    "protocol": pkt['IP'].proto
                }
                if pkt.haslayer(TCP):
                    pkt_info["sport"] = pkt['TCP'].sport
                    pkt_info["dport"] = pkt['TCP'].dport
                    pkt_info["flags"] = str(pkt['TCP'].flags)

            # تحليل نوع الهجوم
            attack_type = analyze_attack_type(pred, err, pkt_info, threshold)

            # حساب confidence
            confidence = min(err / (threshold * 10), 1.0) if threshold > 0 else 0.5

            detection = {
                "packet_index": i,
                "reconstruction_error": float(err),
                "threshold": float(threshold),
                "confidence_percent": round(confidence * 100, 1),
                "attack_type": attack_type,
                "packet_info": pkt_info
            }
            report["detections"].append(detection)

    return report


def send_to_api(results, api_url, api_key):
    """إرسال النتائج للـ API"""
    if not api_url or api_url == "https://your-api-endpoint.com/api/detection-results":
        print("[WARNING] API URL not configured. Skipping API call.")
        return False

    headers = {'Content-Type': 'application/json'}
    if api_key:
        headers['Authorization'] = f'Bearer {api_key}'

    try:
        response = requests.post(api_url, json=results, headers=headers, timeout=30)
        if response.status_code in [200, 201]:
            print(f"[SUCCESS] Results sent to API successfully!")
            print(f"Response: {response.text[:200]}")
            return True
        else:
            print(f"[ERROR] API error: {response.status_code} - {response.text[:200]}")
            return False
    except Exception as e:
        print(f"[ERROR] Failed to send to API: {str(e)}")
        return False


def main(pcap_file, api_url=None, api_key=None):
    """الدالة الرئيسية"""
    print("=" * 60)
    print("  Zero-Day Attack Detection System")
    print("  تحويل pcap للكشف عن الهجمات الجديدة")
    print("=" * 60)

    # التحقق من وجود الملف
    if not os.path.exists(pcap_file):
        print(f"[ERROR] الملف مش موجود: {pcap_file}")
        return None

    # تحميل المودل
    print("\n[1/5] تحميل المودل والـ scaler...")
    model, scaler, threshold = load_model_and_scaler()
    print(f"   ✓ Model loaded (threshold: {threshold:.6f})")

    # قراءة الـ pcap
    print("\n[2/5] قراءة ملف الـ pcap...")
    packets = rdpcap(pcap_file)
    print(f"   ✓ تم قراءة {len(packets)} حزمة")

    # استخراج الـ features
    print("\n[3/5] استخراج الـ features بنفس شكل الداتا المدربة...")
    df_features = extract_flow_features_from_packets(packets)
    print(f"   ✓ تم استخراج {len(df_features)} سجل")
    print(f"   ✓ عدد الأعمدة: {len(df_features.columns)} (مطابق للتدريب)")

    # تنظيف البيانات
    print("\n[4/5] تنظيف البيانات وتشغيل التنبؤ...")
    X_scaled = clean_and_preprocess_data(df_features, scaler)
    print(f"   ✓ تم تنظيف البيانات ومقياستها")

    # تشغيل الكشف
    predictions, mse = detect_anomalies(model, X_scaled, threshold)
    print(f"   ✓ تم تشغيل الكشف على {len(predictions)} حزمة")

    # توليد التقرير
    print("\n[5/5] توليد التقرير...")
    report = generate_report(predictions, mse, packets, df_features, threshold)

    # عرض النتائج
    print("\n" + "=" * 60)
    print("  RESULTS / النتائج")
    print("=" * 60)
    print(f"  Total Packets Analyzed: {report['summary']['total_packets']}")
    print(f"  Attacks Detected: {report['summary']['attacks_detected']}")
    print(f"  Benign Packets: {report['summary']['benign_packets']}")
    print(f"  Attack Rate: {report['summary']['attack_rate_percent']}%")
    print(f"  Overall Status: {report['overall_status']}")

    if report['detections']:
        print("\n" + "-" * 40)
        print("  Suspicious Packets:")
        print("-" * 40)
        for i, detection in enumerate(report['detections'][:10]):
            print(f"\n  [{i+1}] Packet #{detection['packet_index']}")
            print(f"      Type: {detection['attack_type']}")
            print(f"      Reconstruction Error: {detection['reconstruction_error']:.6f}")
            print(f"      Confidence: {detection['confidence_percent']}%")
            if detection['packet_info']:
                pkt = detection['packet_info']
                print(f"      Source: {pkt.get('src_ip', 'N/A')}")
                print(f"      Destination: {pkt.get('dst_ip', 'N/A')}")

    # حفظ التقرير
    report_file = "detection_report.json"
    with open(report_file, 'w') as f:
        json.dump(report, f, indent=2)
    print(f"\n[✓] Report saved: {report_file}")

    # إرسال للـ API
    if api_url:
        print("\n[Sending to API...]")
        send_to_api(report, api_url, api_key)

    return report


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description='Zero-Day Attack Detection - تحويل pcap للكشف عن الهجمات',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
مثال الاستخدام:
  python pcap_detector.py --pcap capture.pcap --api-url https://my-api.com/results
  python pcap_detector.py -p capture.pcap -u https://api.com/endpoint
        """
    )
    parser.add_argument('--pcap', '-p', default=None, help='Path to pcap file')
    parser.add_argument('--api-url', '-u', default=None, help='API endpoint URL')
    parser.add_argument('--api-key', '-k', default=None, help='API key')

    args = parser.parse_args()

    # استخدام القيم من args أو defaults
    pcap_file = args.pcap or DEFAULT_PCAP
    api_url = args.api_url or API_URL
    api_key = args.api_key or API_KEY

    # تشغيل
    results = main(pcap_file, api_url, api_key)