import streamlit as st
import pandas as pd
import numpy as np
import math
import re
import os
import zipfile
import io
import base64
from collections import Counter
import datetime

import matplotlib.pyplot as plt

# --- 確実な文字化け対策 ---
# クラウド（japanize_matplotlib有）とローカル（無）を自動判別し、設定を上書きしない
try:
    import japanize_matplotlib
except ImportError:
    plt.rcParams['font.family'] = ['Meiryo', 'Yu Gothic', 'MS Gothic', 'sans-serif']

# ==========================================
# 状態管理（セッションステート）の初期化
# ==========================================
if "uploader_key" not in st.session_state:
    st.session_state.uploader_key = 0

if "calc_done" not in st.session_state:
    st.session_state.calc_done = False


class VirtualFile:
    def __init__(self, name, data):
        self.name = name
        self.data = data

    def getvalue(self):
        return self.data


# ==========================================
# 各種関数
# ==========================================
def natural_sort_key(file):
    return [int(text) if text.isdigit() else text.lower() for text in re.split(r'(\d+)', file.name)]


def parse_memo_file(file):
    info = {}
    try:
        content = file.getvalue().decode('utf-8', errors='replace').splitlines()
        for line in content:
            parts = [p.strip() for p in line.split(',')]
            if len(parts) == 5:
                time_str, meas_type, folder_num, src_room, recv_room = parts
                mode = "軽量床衝撃音"
                if "タイヤ" in meas_type:
                    mode = "重量床衝撃音(1)"
                elif "ボール" in meas_type:
                    mode = "重量床衝撃音(2)"
                elif "軽量" in meas_type:
                    mode = "軽量床衝撃音"
                elif "壁" in meas_type:
                    mode = "室間音圧レベル差(壁)"
                elif "床" in meas_type:
                    mode = "室間音圧レベル差(床)"
                info[folder_num] = {'mode': mode, 'src': src_room, 'recv': recv_room}
    except Exception:
        pass
    return info


def get_mode_from_folder(folder_num):
    prefix = folder_num[:2]
    if prefix == "01": return "重量床衝撃音(1)"
    if prefix == "02": return "重量床衝撃音(2)"
    if prefix == "03": return "軽量床衝撃音"
    if prefix == "11": return "室間音圧レベル差(壁)"
    if prefix == "12": return "室間音圧レベル差(床)"
    return "軽量床衝撃音"


def extract_rnd_values(file_bytes, target_row):
    lines = file_bytes.decode('shift_jis', errors='replace').splitlines()
    for line in lines:
        if line.startswith(target_row + ","):
            parts = [p.strip() for p in line.split(",")]
            if len(parts) >= 12:
                return np.array([float(x) for x in parts[4:12]])
    return None


def energy_mean(values_list):
    if not values_list: return np.zeros(8)
    arr = np.array(values_list)
    mean_energy = np.mean(10 ** (arr / 10), axis=0)
    return 10 * np.log10(mean_energy)


def round_half_up(n, decimals=0):
    multiplier = 10 ** decimals
    return np.floor(n * multiplier + 0.5) / multiplier


def clear_data():
    st.session_state.uploader_key += 1
    st.session_state.calc_done = False
    st.session_state.excel_data = None
    st.session_state.html_content = None
    st.session_state.macro_data = None
    if "comparison_figs" in st.session_state: del st.session_state.comparison_figs
    if "individual_figs" in st.session_state: del st.session_state.individual_figs


def split_room_name(room_str):
    if not room_str:
        return "", ""
    if "号室" in room_str:
        parts = room_str.split("号室", 1)
        return parts[0], parts[1]
    else:
        return room_str, ""


def map_macro_category(mode):
    if mode == "重量床衝撃音(1)": return "重量_タイヤ"
    if mode == "重量床衝撃音(2)": return "重量_ボール"
    if mode == "軽量床衝撃音": return "軽量"
    if mode == "室間音圧レベル差(壁)": return "室間_壁"
    if mode == "室間音圧レベル差(床)": return "室間_床"
    return mode


# --- HTML・画面表示用の個別グラフ画像生成関数 ---
def generate_graph_base64(mode, is_jis, case_name, src_room, recv_room, values, sn_mask, grade_str, det_freq, eval_num):
    fig, ax = plt.subplots(figsize=(6, 8))
    is_floor = "床衝撃音" in mode
    sn_mask = np.array(sn_mask)

    if is_floor:
        x_labels = ['31.5', '63', '125', '250', '500', '1k', '2k', '4k']
        x_ticks = np.arange(8)
        x_ticks_curve = np.arange(1, 8)
        ref_50 = np.array([73, 63, 56, 50, 47, 46, 46])
        m_intervals = 10
        n_intervals = 7
        y_min, y_max = 10, 110
        curve_label_prefix = 'Lr-' if is_jis else 'L-'
        line_color = 'red'
        ax.set_ylabel('床衝撃音レベル (dB)')

        max_v, min_v = max(values), min(values)
        if max_v > 100:
            y_max = math.ceil(max_v / 10) * 10
            y_min = y_max - 100
        elif min_v < 20:
            y_min = math.floor(min_v / 10) * 10
            y_max = y_min + 100

        y_shift = y_min - 10
        curve_start = 35 + y_shift
        curve_end = 85 + y_shift

        for val in range(curve_start, curve_end + 5, 5):
            c_vals = ref_50 + (val - 50)
            if np.any((c_vals >= y_min) & (c_vals <= y_max)):
                ax.plot(x_ticks_curve, c_vals, color='black', linewidth=0.8)
                ax.text(x_ticks_curve[-1] + 0.1, c_vals[-1], f"{curve_label_prefix}{val}", va='center', ha='left',
                        fontsize=10)

    else:
        x_labels = ['63', '125', '250', '500', '1k', '2k', '4k']
        x_ticks = np.arange(7)
        x_ticks_curve = np.arange(1, 7)
        ref_50 = np.array([35, 42.5, 50, 55, 60, 60])
        n_intervals = 6
        line_color = 'blue'
        ax.set_ylabel('音圧レベル差 (dB)')

        values = values[1:]
        sn_mask = sn_mask[1:]

        if is_jis:
            m_intervals = 8
            y_min, y_max = 0, 80
            curve_label_prefix = 'Dr-'
            for val in range(30, 65, 5):
                c_vals = ref_50 + (val - 50)
                if np.any((c_vals >= y_min) & (c_vals <= y_max)):
                    ax.plot(x_ticks_curve, c_vals, color='black', linewidth=0.8)
                    ax.text(x_ticks_curve[-1] + 0.1, c_vals[-1], f"{curve_label_prefix}{val}", va='center', ha='left',
                            fontsize=10)
        else:
            m_intervals = 9
            y_min, y_max = 0, 90
            curve_label_prefix = 'D-'
            aij_curves = {
                'D-15': np.array([10., 12.5, 15., 15., 15., 15.]),
                'D-20': np.array([10., 15., 20., 20., 20., 20.]),
                'D-25': np.array([10., 17.5, 25., 25., 25., 25.]),
                'D-30-Ⅱ': np.array([15., 22.5, 30., 30., 30., 30.]),
                'D-30-Ⅰ': np.array([15., 22.5, 30., 35., 35., 35.]),
                'D-30': np.array([15., 22.5, 30., 35., 40., 40.]),
                'D-35': np.array([20., 27.5, 35., 40., 45., 45.]),
                'D-40': np.array([25., 32.5, 40., 45., 50., 50.]),
                'D-45': np.array([30., 37.5, 45., 50., 55., 55.]),
                'D-50': np.array([35., 42.5, 50., 55., 60., 60.]),
                'D-55': np.array([40., 47.5, 55., 60., 65., 65.]),
                'D-60': np.array([45., 52.5, 60., 65., 70., 70.]),
                'D-65': np.array([50., 57.5, 65., 70., 75., 75.]),
                'D-70': np.array([55., 62.5, 70., 75., 80., 80.]),
                'D-75': np.array([60., 67.5, 75., 80., 85., 85.]),
                'D-80': np.array([65., 72.5, 80., 85., 85., 85.]),
                'D-85': np.array([70., 77.5, 85., 85., 85., 85.])
            }
            for lbl, c_vals in aij_curves.items():
                if np.any((c_vals >= y_min) & (c_vals <= y_max)):
                    ax.plot(x_ticks_curve, c_vals, color='black', linewidth=0.8)
                    label_y = c_vals[-1]
                    if lbl in ['D-80', 'D-85']:
                        ax.text(0.5, c_vals[0], lbl, va='center', ha='center', fontsize=10, fontweight='bold')
                    else:
                        if lbl == 'D-75': label_y = 83
                        ax.text(x_ticks_curve[-1] + 0.1, label_y, lbl, va='center', ha='left', fontsize=10)

    valid_mask = ~sn_mask
    invalid_mask = sn_mask
    vals_arr = np.array(values)

    ax.plot(x_ticks, values, color=line_color, linewidth=2.0, zorder=2)

    if np.any(valid_mask):
        ax.plot(x_ticks[valid_mask], vals_arr[valid_mask], marker='o', linestyle='', color=line_color,
                markerfacecolor=line_color, markersize=6, zorder=3)
    if np.any(invalid_mask):
        ax.plot(x_ticks[invalid_mask], vals_arr[invalid_mask], marker='o', linestyle='', color=line_color,
                markerfacecolor='white', markeredgewidth=1.5, markersize=6, zorder=3)

    for i, v in enumerate(values):
        ax.text(i + 0.15, v, f"{int(round_half_up(v))}", color=line_color, fontsize=10, va='center')

    ax.set_xticks(x_ticks)
    ax.set_xticklabels(x_labels)
    ax.set_xlim(0, n_intervals + 0.8)
    ax.set_ylim(y_min, y_max)
    ax.set_yticks(np.arange(y_min, y_max + 1, 10))
    ax.grid(True, which='both', axis='both', color='gray', linestyle='-', linewidth=0.5)
    ax.set_xlabel('オクターブバンド中心周波数(Hz)')
    ax.set_box_aspect((m_intervals * 4) / (n_intervals * 3))

    if grade_str != "計算不可":
        grade_num = re.search(r'\d+', grade_str).group()
        graph_grade_str = f"{curve_label_prefix}{grade_num}"
    else:
        graph_grade_str = "計算不可"

    room_info = f" (音源室: {src_room} / 受音室: {recv_room})" if src_room or recv_room else ""
    num_label = "L数" if is_floor else "D数"
    title_str = f"測定番号: {case_name}{room_info}\n判定: {graph_grade_str}    {num_label}: {eval_num} [{det_freq}]"
    ax.set_title(title_str, pad=15, fontweight='bold')

    plt.tight_layout()
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=100)
    plt.close(fig)
    buf.seek(0)  # 修正: 画像データの巻き戻し
    return base64.b64encode(buf.read()).decode('utf-8')


# --- 現場確認用の比較グラフ生成関数 ---
def generate_comparison_graph_base64(mode, is_jis, items_list):
    fig, ax = plt.subplots(figsize=(8, 8))
    is_floor = "床衝撃音" in mode

    if is_floor:
        x_labels = ['31.5', '63', '125', '250', '500', '1k', '2k', '4k']
        x_ticks = np.arange(8)
        x_ticks_curve = np.arange(1, 8)
        ref_50 = np.array([73, 63, 56, 50, 47, 46, 46])
        m_intervals = 10
        n_intervals = 7
        y_min, y_max = 10, 110
        curve_label_prefix = 'Lr-' if is_jis else 'L-'
        ax.set_ylabel('床衝撃音レベル (dB)')

        all_vals = []
        for item in items_list:
            all_vals.extend(item["_raw_values"])

        if all_vals:
            max_v, min_v = max(all_vals), min(all_vals)
            if max_v > 100:
                y_max = math.ceil(max_v / 10) * 10
                y_min = y_max - 100
            elif min_v < 20:
                y_min = math.floor(min_v / 10) * 10
                y_max = y_min + 100

        y_shift = y_min - 10
        curve_start = 35 + y_shift
        curve_end = 85 + y_shift

        for val in range(curve_start, curve_end + 5, 5):
            c_vals = ref_50 + (val - 50)
            if np.any((c_vals >= y_min) & (c_vals <= y_max)):
                ax.plot(x_ticks_curve, c_vals, color='black', linewidth=0.8)
                ax.text(x_ticks_curve[-1] + 0.1, c_vals[-1], f"{curve_label_prefix}{val}", va='center', ha='left',
                        fontsize=10)
    else:
        x_labels = ['63', '125', '250', '500', '1k', '2k', '4k']
        x_ticks = np.arange(7)
        x_ticks_curve = np.arange(1, 7)
        ref_50 = np.array([35, 42.5, 50, 55, 60, 60])
        n_intervals = 6
        ax.set_ylabel('音圧レベル差 (dB)')

        if is_jis:
            m_intervals = 8
            y_min, y_max = 0, 80
            curve_label_prefix = 'Dr-'
            for val in range(30, 65, 5):
                c_vals = ref_50 + (val - 50)
                if np.any((c_vals >= y_min) & (c_vals <= y_max)):
                    ax.plot(x_ticks_curve, c_vals, color='black', linewidth=0.8)
                    ax.text(x_ticks_curve[-1] + 0.1, c_vals[-1], f"{curve_label_prefix}{val}", va='center', ha='left',
                            fontsize=10)
        else:
            m_intervals = 9
            y_min, y_max = 0, 90
            curve_label_prefix = 'D-'
            aij_curves = {
                'D-15': np.array([10., 12.5, 15., 15., 15., 15.]),
                'D-20': np.array([10., 15., 20., 20., 20., 20.]),
                'D-25': np.array([10., 17.5, 25., 25., 25., 25.]),
                'D-30-Ⅱ': np.array([15., 22.5, 30., 30., 30., 30.]),
                'D-30-Ⅰ': np.array([15., 22.5, 30., 35., 35., 35.]),
                'D-30': np.array([15., 22.5, 30., 35., 40., 40.]),
                'D-35': np.array([20., 27.5, 35., 40., 45., 45.]),
                'D-40': np.array([25., 32.5, 40., 45., 50., 50.]),
                'D-45': np.array([30., 37.5, 45., 50., 55., 55.]),
                'D-50': np.array([35., 42.5, 50., 55., 60., 60.]),
                'D-55': np.array([40., 47.5, 55., 60., 65., 65.]),
                'D-60': np.array([45., 52.5, 60., 65., 70., 70.]),
                'D-65': np.array([50., 57.5, 65., 70., 75., 75.]),
                'D-70': np.array([55., 62.5, 70., 75., 80., 80.]),
                'D-75': np.array([60., 67.5, 75., 80., 85., 85.]),
                'D-80': np.array([65., 72.5, 80., 85., 85., 85.]),
                'D-85': np.array([70., 77.5, 85., 85., 85., 85.])
            }
            for lbl, c_vals in aij_curves.items():
                if np.any((c_vals >= y_min) & (c_vals <= y_max)):
                    ax.plot(x_ticks_curve, c_vals, color='black', linewidth=0.8)
                    label_y = c_vals[-1]
                    if lbl in ['D-80', 'D-85']:
                        ax.text(0.5, c_vals[0], lbl, va='center', ha='center', fontsize=10, fontweight='bold')
                    else:
                        if lbl == 'D-75': label_y = 83
                        ax.text(x_ticks_curve[-1] + 0.1, label_y, lbl, va='center', ha='left', fontsize=10)

    # 複数データの折れ線描画
    cmap = plt.get_cmap("tab10")
    for i, item in enumerate(items_list):
        vals = item["_raw_values"]
        sn_mask = np.array(item["_sn_mask"])

        if not is_floor:
            vals = vals[1:]
            sn_mask = sn_mask[1:]

        valid_mask = ~sn_mask
        invalid_mask = sn_mask
        vals_arr = np.array(vals)
        color = cmap(i % 10)

        # 凡例用のラベル
        line_label = f"{item['測定番号']}"
        ax.plot(x_ticks, vals, color=color, linewidth=2.0, label=line_label, zorder=2)

        if np.any(valid_mask):
            ax.plot(x_ticks[valid_mask], vals_arr[valid_mask], marker='o', linestyle='', color=color,
                    markerfacecolor=color, markersize=6, zorder=3)
        if np.any(invalid_mask):
            ax.plot(x_ticks[invalid_mask], vals_arr[invalid_mask], marker='o', linestyle='', color=color,
                    markerfacecolor='white', markeredgewidth=1.5, markersize=6, zorder=3)

        for j, v in enumerate(vals):
            ax.text(j + 0.15, v, f"{int(round_half_up(v))}", color=color, fontsize=9, va='center')

    ax.set_xticks(x_ticks)
    ax.set_xticklabels(x_labels)
    ax.set_xlim(0, n_intervals + 0.8)
    ax.set_ylim(y_min, y_max)
    ax.set_yticks(np.arange(y_min, y_max + 1, 10))
    ax.grid(True, which='both', axis='both', color='gray', linestyle='-', linewidth=0.5)
    ax.set_xlabel('オクターブバンド中心周波数(Hz)')
    ax.set_box_aspect((m_intervals * 4) / (n_intervals * 3))

    ax.set_title(f"【比較】{mode}", pad=15, fontweight='bold')
    # 凡例をグラフの外側右上に配置
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)

    plt.tight_layout()
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=100)
    plt.close(fig)
    buf.seek(0)  # 修正: 画像データの巻き戻し
    return base64.b64encode(buf.read()).decode('utf-8')


# ==========================================
# メイン画面（UI）
# ==========================================
st.set_page_config(page_title="遮音性能判定アプリ", layout="wide")
st.title("遮音性能・床衝撃音 判定アプリ")

st.markdown("""
### 📌 自動判定のルール
* **現場メモがある場合**：カンマ区切りで「`時間,測定種別,フォルダ4桁,音源室,受音室`」の5列になっている行のみを読み取り、設定を自動反映します。
* **現場メモがない場合**：
  * **測定種別**: フォルダ名の先頭2桁（01=重(1)、02=重(2)、03=軽量、11=壁、12=床）で判別します。
  * **ファイルの役割・除外**: 
    * **床衝撃音測定**（基本26ファイル）: 1〜25番目を「音源位置1〜5」、26番目を「暗騒音」に割り当て、27番目以降は「除外」にチェックします。
    * **室間レベル差測定**（基本11ファイル）: 1番目を「暗騒音」、2〜6番目を「受音室」、7〜11番目を「音源室」に割り当て、12番目以降は「除外」にチェックします。
""")

col_msg, col_btn = st.columns([3, 1])
with col_msg:
    st.write("対象の `.RND` フォルダ群（ZIP）とメモ `.txt` または `.csv` をアップロードしてください。")
with col_btn:
    if st.button("🗑️ アップロードファイルを一括クリア", use_container_width=True):
        clear_data()
        st.rerun()

uploaded_files = st.file_uploader(
    "ファイルをアップロード (.zip / .rnd / .txt / .csv)",
    accept_multiple_files=True,
    type=['rnd', 'txt', 'csv', 'zip'],
    key=f"uploader_{st.session_state.uploader_key}"
)

if uploaded_files:
    all_files = []
    for file in uploaded_files:
        if file.name.lower().endswith('.zip'):
            with zipfile.ZipFile(file, 'r') as z:
                for name in z.namelist():
                    if not name.endswith('/') and '__MACOSX' not in name:
                        file_data = z.read(name)
                        base_name = os.path.basename(name)
                        if base_name.lower().endswith(('.rnd', '.txt', '.csv')):
                            all_files.append(VirtualFile(base_name, file_data))
        else:
            all_files.append(file)

    memo_info = {}
    rnd_files = []
    for file in all_files:
        if file.name.lower().endswith(('.txt', '.csv')):
            memo_info.update(parse_memo_file(file))
        else:
            rnd_files.append(file)

    measurements = {}
    for file in rnd_files:
        content = file.getvalue().decode('shift_jis', errors='replace').splitlines()
        store_name = "Unknown"
        for line in content[:5]:
            if line.startswith("Store Name"):
                store_name = line.split(",")[1].strip()
                break
        if store_name not in measurements:
            measurements[store_name] = []
        measurements[store_name].append(file)

    if not measurements:
        st.warning("有効な .RND ファイルが見つかりませんでした。")
    else:
        st.success(
            f"読み込み完了！ メモ情報と {len(rnd_files)} 個のデータを統合し、 {len(measurements)} 件の測定を自動設定しました。")
        st.header("測定データの一覧と設定")

        mode_counters = {"重量床衝撃音(1)": 1, "重量床衝撃音(2)": 1, "軽量床衝撃音": 1, "室間音圧レベル差(壁)": 1,
                         "室間音圧レベル差(床)": 1}
        mode_abbr = {"重量床衝撃音(1)": "重(1)", "重量床衝撃音(2)": "重(2)", "軽量床衝撃音": "軽",
                     "室間音圧レベル差(壁)": "壁", "室間音圧レベル差(床)": "床"}
        modes_list = list(mode_counters.keys())

        for store_name in sorted(measurements.keys()):
            folder_num = store_name.replace("MAN_", "")
            mode = memo_info.get(folder_num, {}).get('mode', get_mode_from_folder(folder_num))
            case_key = f"case_{store_name}"
            if case_key not in st.session_state:
                st.session_state[case_key] = f"{mode_abbr.get(mode, '不明')}{mode_counters.get(mode, 1)}"
            if mode in mode_counters:
                mode_counters[mode] += 1

        active_cases = [st.session_state.get(f"case_{s}", "") for s in measurements.keys() if
                        not st.session_state.get(f"skip_{s}", False)]
        case_counts = Counter(active_cases)
        has_duplicate = any(count > 1 for count in case_counts.values())

        edited_dfs = {}

        for store_name in sorted(measurements.keys()):
            files = measurements[store_name]
            files.sort(key=natural_sort_key)
            file_count = len(files)
            folder_num = store_name.replace("MAN_", "")

            mode = memo_info.get(folder_num, {}).get('mode', get_mode_from_folder(folder_num))
            src_room = memo_info.get(folder_num, {}).get('src', "")
            recv_room = memo_info.get(folder_num, {}).get('recv', "")

            case_key = f"case_{store_name}"
            current_case_name = st.session_state[case_key]
            is_duplicate = not st.session_state.get(f"skip_{store_name}", False) and case_counts[current_case_name] > 1
            title_icon = "⚠️" if is_duplicate else "📁"

            with st.expander(f"{title_icon} 【{current_case_name}】 フォルダ: {folder_num} (ファイル数: {file_count}件)",
                             expanded=is_duplicate):
                col1, col2, col3, col4, col5 = st.columns(5)
                with col1:
                    st.selectbox("測定種別", modes_list, index=modes_list.index(mode) if mode in modes_list else 0,
                                 key=f"mode_{store_name}")
                with col2:
                    st.text_input("ケース名", key=case_key)
                    if is_duplicate: st.error("重複しています")
                with col3:
                    st.text_input("音源室", value=src_room, key=f"src_{store_name}")
                with col4:
                    st.text_input("受音室", value=recv_room, key=f"recv_{store_name}")
                with col5:
                    st.write("")
                    st.checkbox("この測定を除外", key=f"skip_{store_name}")

                file_names = [f.name for f in files]
                file_roles, exclude_flags = [], []

                for i in range(file_count):
                    role, exclude = "測定データ", False
                    if "室間" in mode:
                        if i == 0:
                            role = "暗騒音"
                        elif 1 <= i <= 5:
                            role = "受音室"
                        elif 6 <= i <= 10:
                            role = "音源室"
                        else:
                            exclude = True
                    else:
                        if 0 <= i <= 4:
                            role = "音源位置1"
                        elif 5 <= i <= 9:
                            role = "音源位置2"
                        elif 10 <= i <= 14:
                            role = "音源位置3"
                        elif 15 <= i <= 19:
                            role = "音源位置4"
                        elif 20 <= i <= 24:
                            role = "音源位置5"
                        elif i == 25:
                            role = "暗騒音"
                        else:
                            exclude = True
                    file_roles.append(role)
                    exclude_flags.append(exclude)

                role_options = ["音源位置1", "音源位置2", "音源位置3", "音源位置4", "音源位置5", "受音室", "音源室",
                                "暗騒音", "測定データ"]
                df_files = pd.DataFrame({"ファイル名": file_names, "役割": file_roles, "除外": exclude_flags})
                edited_dfs[store_name] = st.data_editor(
                    df_files,
                    column_config={"役割": st.column_config.SelectboxColumn("役割", options=role_options),
                                   "除外": st.column_config.CheckboxColumn("欠測(除外)")},
                    hide_index=True, key=f"editor_{store_name}"
                )

        # ==========================================
        # フェーズ1: 計算・速報出力
        # ==========================================
        st.markdown("---")
        col_eval1, col_eval2 = st.columns(2)
        with col_eval1:
            eval_method = st.radio("📈 評価方法", ["AIJ（日本建築学会）", "JIS（日本産業規格）"], horizontal=True)
        with col_eval2:
            st.write("")
            st.checkbox("2dBの超過/不足を許容する（現場測定用）", value=True, key="allow_2db")

        if st.button("計算を実行して結果を出力", type="primary"):
            if has_duplicate:
                st.error(
                    "⚠️ ケース名が重複している箇所があります。名前を変更するか、「この測定を除外」にチェックを入れてから再実行してください。")
            else:
                with st.spinner("データを集計し、グラフを生成しています..."):
                    results_data = []
                    macro_list_data = []

                    freq_labels = ["31.5Hz", "63Hz", "125Hz", "250Hz", "500Hz", "1kHz", "2kHz", "4kHz"]
                    allowance = 2 if st.session_state["allow_2db"] else 0
                    is_jis = "JIS" in eval_method
                    st.session_state.is_jis = is_jis  # 比較グラフ生成用に保持
                    REF_L50 = [73, 63, 56, 50, 47, 46, 46]
                    REF_D50 = [35, 42.5, 50, 55, 60, 60]

                    for store_name in sorted(measurements.keys()):
                        folder_num = store_name.replace("MAN_", "")
                        if st.session_state.get(f"skip_{store_name}", False): continue

                        case_name = st.session_state[f"case_{store_name}"]
                        mode = st.session_state[f"mode_{store_name}"]
                        src_room = st.session_state.get(f"src_{store_name}", "")
                        recv_room = st.session_state.get(f"recv_{store_name}", "")
                        editor_data = edited_dfs[store_name]

                        # --- マクロ用データ収集 ---
                        src_1, src_2 = split_room_name(src_room)
                        recv_1, recv_2 = split_room_name(recv_room)
                        macro_list_data.append({
                            "フォルダ番号": folder_num,
                            "項目": map_macro_category(mode),
                            "ケース名": case_name,
                            "音源室_1": src_1,
                            "音源室_2": src_2,
                            "受音室_1": recv_1,
                            "受音室_2": recv_2
                        })

                        # --- 計算処理 ---
                        target_row = "Lmax" if "重量" in mode else "Leq"
                        measured_vals, bg_vals, src_vals, recv_vals = [], [], [], []

                        for idx, row in editor_data.iterrows():
                            if row["除外"]: continue
                            file_obj = measurements[store_name][idx]
                            role = row["役割"]
                            if role == "暗騒音":
                                bg_v = extract_rnd_values(file_obj.getvalue(), "Leq")
                                if bg_v is not None: bg_vals.append(bg_v)
                            else:
                                vals = extract_rnd_values(file_obj.getvalue(), target_row)
                                if vals is not None:
                                    if "音源位置" in role or role == "測定データ":
                                        measured_vals.append((role, vals))
                                    elif role == "音源室":
                                        src_vals.append(vals)
                                    elif role == "受音室":
                                        recv_vals.append(vals)

                        result_row = {"測定種別": mode, "測定番号": case_name, "音源室": src_room, "受音室": recv_room,
                                      "遮音等級": "計算不可", "評価数": "", "決定周波数": ""}
                        bg_mean = energy_mean(bg_vals) if bg_vals else np.zeros(8)


                        def apply_bg_correction(p_vals, b_vals):
                            corr_p = []
                            for m, b in zip(p_vals, b_vals):
                                if b <= 0:
                                    corr_p.append(m)
                                else:
                                    diff = m - b
                                    if diff >= 15:
                                        corr_p.append(m)
                                    elif diff >= 6:
                                        corr_p.append(10 * math.log10(10 ** (m / 10) - 10 ** (b / 10)))
                                    else:
                                        corr_p.append(m)
                            return np.array(corr_p)


                        if "床衝撃音" in mode:
                            if not measured_vals: continue

                            raw_src_groups = {}
                            source_groups = {}
                            for role, vals in measured_vals:
                                if role not in raw_src_groups:
                                    raw_src_groups[role] = []
                                    source_groups[role] = []
                                raw_src_groups[role].append(vals)
                                source_groups[role].append(apply_bg_correction(vals, bg_mean))

                            raw_src_means = [round_half_up(energy_mean(pts), 1) for pts in raw_src_groups.values()]
                            raw_arith_mean = np.mean(raw_src_means, axis=0)
                            sn_mask_full = (raw_arith_mean - bg_mean) < 6

                            source_energy_means = [round_half_up(energy_mean(pts), 1) for pts in source_groups.values()]
                            arith_mean = np.mean(source_energy_means, axis=0)
                            eval_vals = round_half_up(arith_mean, 1)
                            comp_vals = round_half_up(eval_vals, 0)

                            if is_jis:
                                if "重量" in mode:
                                    eval_comp, eval_ref = comp_vals[1:5], REF_L50[0:4]
                                    f_names = ["63Hz", "125Hz", "250Hz", "500Hz"]
                                    prefix = "Li,Fmax,r,H(1)-"
                                    sn_mask_sliced = sn_mask_full[1:5]
                                else:
                                    eval_comp, eval_ref = comp_vals[2:7], REF_L50[1:6]
                                    f_names = ["125Hz", "250Hz", "500Hz", "1kHz", "2kHz"]
                                    prefix = "Li,r,L-"
                                    sn_mask_sliced = sn_mask_full[2:7]
                            else:
                                if "重量床衝撃音(1)" in mode:
                                    eval_comp, eval_ref = comp_vals[1:5], REF_L50[0:4]
                                    f_names = ["63Hz", "125Hz", "250Hz", "500Hz"]
                                    prefix = "LH-"
                                    sn_mask_sliced = sn_mask_full[1:5]
                                else:
                                    eval_comp, eval_ref = comp_vals[1:], REF_L50
                                    f_names = ["63Hz", "125Hz", "250Hz", "500Hz", "1kHz", "2kHz", "4kHz"]
                                    prefix = "LL-"
                                    sn_mask_sliced = sn_mask_full[1:]

                            unrounded_L_nums = [m - ref + 50 for m, ref in zip(eval_comp, eval_ref)]
                            L_nums = [round_half_up(x) for x in unrounded_L_nums]

                            valid_L_nums = [num for i, num in enumerate(L_nums) if not sn_mask_sliced[i]]

                            if valid_L_nums:
                                L_value = max(valid_L_nums)
                                max_unrounded = float('-inf')
                                det_idx = 0
                                for i in range(len(L_nums)):
                                    if not sn_mask_sliced[i] and L_nums[i] == L_value:
                                        if unrounded_L_nums[i] > max_unrounded:
                                            max_unrounded = unrounded_L_nums[i]
                                            det_idx = i
                            else:
                                L_value = max(L_nums)
                                max_unrounded = float('-inf')
                                det_idx = 0
                                for i in range(len(L_nums)):
                                    if L_nums[i] == L_value:
                                        if unrounded_L_nums[i] > max_unrounded:
                                            max_unrounded = unrounded_L_nums[i]
                                            det_idx = i

                            grade_val = math.ceil((L_value - allowance) / 5) * 5
                            result_row["遮音等級"] = f"{prefix}{int(grade_val)}"
                            result_row["評価数"] = int(L_value)
                            result_row["決定周波数"] = f_names[det_idx]
                            result_row["_raw_values"] = eval_vals.tolist()
                            result_row["_sn_mask"] = sn_mask_full.tolist()
                            for f_label, val in zip(freq_labels, eval_vals): result_row[f_label] = val

                        elif "室間" in mode:
                            if not src_vals or not recv_vals: continue

                            raw_recv_mean = energy_mean(recv_vals)
                            sn_mask_full = (raw_recv_mean - bg_mean) < 6

                            src_mean = energy_mean(src_vals)
                            corr_recv = [apply_bg_correction(v, bg_mean) for v in recv_vals]
                            recv_mean = energy_mean(corr_recv)
                            d_val_raw = src_mean - recv_mean
                            eval_vals = round_half_up(d_val_raw, 1)
                            comp_vals = round_half_up(eval_vals, 0)

                            if is_jis:
                                eval_comp, eval_ref = comp_vals[2:7], REF_D50[0:5]
                                f_names = ["125Hz", "250Hz", "500Hz", "1kHz", "2kHz"]
                                prefix = "Dr-"
                                sn_mask_sliced = sn_mask_full[2:7]
                            else:
                                eval_comp, eval_ref = comp_vals[2:], REF_D50
                                f_names = ["125Hz", "250Hz", "500Hz", "1kHz", "2kHz", "4kHz"]
                                prefix = "D-"
                                sn_mask_sliced = sn_mask_full[2:]

                            unrounded_D_nums = [m - ref + 50 for m, ref in zip(eval_comp, eval_ref)]
                            D_nums = [round_half_up(x) for x in unrounded_D_nums]

                            valid_D_nums = [num for i, num in enumerate(D_nums) if not sn_mask_sliced[i]]

                            if valid_D_nums:
                                D_value = min(valid_D_nums)
                                min_unrounded = float('inf')
                                det_idx = 0
                                for i in range(len(D_nums)):
                                    if not sn_mask_sliced[i] and D_nums[i] == D_value:
                                        if unrounded_D_nums[i] < min_unrounded:
                                            min_unrounded = unrounded_D_nums[i]
                                            det_idx = i
                            else:
                                D_value = min(D_nums)
                                min_unrounded = float('inf')
                                det_idx = 0
                                for i in range(len(D_nums)):
                                    if D_nums[i] == D_value:
                                        if unrounded_D_nums[i] < min_unrounded:
                                            min_unrounded = unrounded_D_nums[i]
                                            det_idx = i

                            grade_val = math.floor((D_value + allowance) / 5) * 5
                            result_row["遮音等級"] = f"{prefix}{int(grade_val)}"
                            result_row["評価数"] = int(D_value)
                            result_row["決定周波数"] = f_names[det_idx]
                            result_row["_raw_values"] = eval_vals.tolist()
                            result_row["_sn_mask"] = sn_mask_full.tolist()
                            for f_label, val in zip(freq_labels, eval_vals): result_row[f_label] = val

                        results_data.append(result_row)

                    if results_data:
                        mode_order = [
                            ("室間音圧レベル差(壁)", "空気遮音性能結果 (壁)", "D数"),
                            ("室間音圧レベル差(床)", "空気遮音性能結果 (床)", "D数"),
                            ("軽量床衝撃音", "床衝撃音遮断性能 (軽量)", "L数"),
                            ("重量床衝撃音(1)", "床衝撃音遮断性能 (重量[タイヤ])", "L数"),
                            ("重量床衝撃音(2)", "床衝撃音遮断性能 (重量[ボール])", "L数")
                        ]

                        df_results = pd.DataFrame(results_data)
                        summary_dfs = {}
                        for mode_key, title, val_label in mode_order:
                            df_group = df_results[df_results["測定種別"] == mode_key].copy()
                            if not df_group.empty:
                                df_group["音源室/受音室"] = df_group["音源室"] + " / " + df_group["受音室"]
                                df_group = df_group.rename(columns={"遮音等級": "遮音等級", "評価数": val_label})
                                cols_to_show = ["測定番号", "音源室/受音室", "遮音等級", val_label,
                                                "決定周波数"] + freq_labels
                                summary_dfs[title] = df_group[cols_to_show]

                        # --- 画像生成（HTML埋め込み・画面表示用） ---
                        individual_figs = {}
                        for r in results_data:
                            if "_raw_values" in r:
                                mode_key = r["測定種別"]
                                b64_img = generate_graph_base64(
                                    mode=mode_key,
                                    is_jis=is_jis,
                                    case_name=r["測定番号"],
                                    src_room=r["音源室"],
                                    recv_room=r["受音室"],
                                    values=r["_raw_values"],
                                    sn_mask=r["_sn_mask"],
                                    grade_str=r["遮音等級"],
                                    det_freq=r["決定周波数"],
                                    eval_num=r["評価数"]
                                )
                                if mode_key not in individual_figs:
                                    individual_figs[mode_key] = []
                                individual_figs[mode_key].append(b64_img)

                        # 速報Excel生成
                        output_excel = io.BytesIO()
                        with pd.ExcelWriter(output_excel, engine='xlsxwriter') as writer:
                            workbook = writer.book
                            worksheet = workbook.add_worksheet('速報一覧')
                            format_title = workbook.add_format({'bold': True, 'align': 'center', 'font_size': 11})
                            format_header = workbook.add_format(
                                {'bottom': 1, 'top': 1, 'bold': True, 'align': 'center'})
                            format_center = workbook.add_format({'align': 'center'})

                            current_row = 1
                            for title, df_summary in summary_dfs.items():
                                num_cols = len(df_summary.columns)
                                worksheet.merge_range(current_row, 1, current_row, num_cols, title, format_title)
                                current_row += 1
                                for col_idx, col_name in enumerate(df_summary.columns):
                                    worksheet.write(current_row, col_idx + 1, col_name, format_header)
                                current_row += 1
                                for _, row in df_summary.iterrows():
                                    for col_idx, val in enumerate(row):
                                        worksheet.write(current_row, col_idx + 1, val, format_center)
                                    current_row += 1
                                for col_idx in range(num_cols):
                                    worksheet.write(current_row, col_idx + 1, "", workbook.add_format({'top': 1}))
                                current_row += 2

                            worksheet.set_column('B:B', 12)
                            worksheet.set_column('C:C', 25)
                            worksheet.set_column('D:F', 10)
                            worksheet.set_column('G:N', 8)

                        # 速報HTML生成
                        html_content = """<html><head><meta charset="utf-8"><title>遮音性能 速報レポート</title><style>
                            body { font-family: "Noto Sans CJK JP", "Meiryo", "MS Gothic", sans-serif; padding: 20px; color: #333; font-size: 12px; }
                            h2 { text-align: center; border-bottom: 2px solid #333; padding-bottom: 10px; margin-bottom: 30px; font-size: 18px; }
                            h3 { margin-top: 40px; text-align: center; font-size: 16px; border-left: 5px solid #333; padding-left: 10px; }
                            table { border-collapse: collapse; width: 100%; margin: 0 auto 30px auto; table-layout: fixed; }
                            th, td { border-top: 1px solid #333; border-bottom: 1px solid #333; padding: 8px 4px; text-align: center; word-wrap: break-word; }
                            th { background-color: #f9f9f9; }
                            .graph-container { display: flex; flex-wrap: wrap; justify-content: center; gap: 20px; margin-bottom: 40px; }
                            .graph-card { border: 1px solid #ccc; padding: 10px; background: #fff; box-shadow: 2px 2px 5px rgba(0,0,0,0.1); }
                            @media print { body { padding: 0; } h3 { page-break-after: avoid; } table { page-break-inside: avoid; } .graph-card { page-break-inside: avoid; box-shadow: none; border: none; } }
                        </style></head><body><h2>遮音性能 速報レポート</h2>"""

                        for mode_key, title, val_label in mode_order:
                            if title in summary_dfs:
                                df_summary = summary_dfs[title]
                                html_content += f"<h3>{title}</h3>\n"
                                html_content += df_summary.to_html(index=False, border=0, justify='center')

                                html_content += "<div class='graph-container'>"
                                if mode_key in individual_figs:
                                    for b64_img in individual_figs[mode_key]:
                                        html_content += f"<div class='graph-card'><img src='data:image/png;base64,{b64_img}' style='width: 450px; max-width: 100%;'></div>"
                                html_content += "</div>\n"

                        html_content += "<p style='text-align:center; font-size:12px; color:#555;'>※ グラフ中のプロットが白抜きの帯域は、暗騒音との差（S/N比）が6dB未満であり、評価対象から除外したことを示します。</p>"
                        html_content += "</body></html>"

                        # マクロ用テキスト (TXT) 生成
                        df_macro = pd.DataFrame(macro_list_data)
                        df_macro["フォルダ番号"] = df_macro["フォルダ番号"].astype(str).str.zfill(4)
                        macro_csv_data = df_macro.to_csv(index=False, sep=',', encoding='cp932')

                        # セッションへ保存
                        st.session_state.results_data = results_data
                        st.session_state.summary_dfs = summary_dfs
                        st.session_state.individual_figs = individual_figs
                        st.session_state.excel_data = output_excel.getvalue()
                        st.session_state.html_content = html_content.encode('utf-8')
                        st.session_state.macro_data = macro_csv_data.encode('cp932')
                        st.session_state.calc_done = True
                        st.rerun()

        # --- 計算結果・グラフの画面表示 ---
        if st.session_state.calc_done:
            st.write("### 📊 計算結果（速報）")
            for title, df_summary in st.session_state.summary_dfs.items():
                st.write(f"#### {title}")
                st.dataframe(df_summary, hide_index=True)

            st.markdown("---")
            st.write("### 📈 現場確認用 比較グラフ（画面表示のみ）")

            mode_order = [
                ("室間音圧レベル差(壁)", "空気遮音性能結果 (壁)", "D数"),
                ("室間音圧レベル差(床)", "空気遮音性能結果 (床)", "D数"),
                ("軽量床衝撃音", "床衝撃音遮断性能 (軽量)", "L数"),
                ("重量床衝撃音(1)", "床衝撃音遮断性能 (重量[タイヤ])", "L数"),
                ("重量床衝撃音(2)", "床衝撃音遮断性能 (重量[ボール])", "L数")
            ]

            comparison_modes = []
            for mode_key, title, _ in mode_order:
                items = [r for r in st.session_state.results_data if r["測定種別"] == mode_key and "_raw_values" in r]
                if items:
                    comparison_modes.append((mode_key, items))

            if comparison_modes:
                tabs = st.tabs([mode for mode, _ in comparison_modes])
                for i, (mode_key, items) in enumerate(comparison_modes):
                    with tabs[i]:
                        case_names = [item["測定番号"] for item in items]
                        selected_cases = st.multiselect(
                            f"【{mode_key}】 グラフに表示する測定データを選択",
                            options=case_names,
                            default=case_names,
                            key=f"ms_{mode_key}"
                        )

                        selected_items = [item for item in items if item["測定番号"] in selected_cases]

                        if selected_items:
                            comp_img = generate_comparison_graph_base64(mode_key, st.session_state.is_jis,
                                                                        selected_items)
                            col1, col2, col3 = st.columns([1, 2, 1])
                            with col2:
                                st.image(f"data:image/png;base64,{comp_img}", use_container_width=True)
                        else:
                            st.info("データが選択されていません。上の入力欄から表示したいデータを選択してください。")

            st.markdown("---")
            with st.expander("📉 個別グラフのプレビュー（※HTMLに出力されるグラフと同じものです）"):
                if "individual_figs" in st.session_state:
                    for mode_key, _, _ in mode_order:
                        if mode_key in st.session_state.individual_figs:
                            st.write(f"**{mode_key}**")
                            cols = st.columns(3)
                            for i, b64_img in enumerate(st.session_state.individual_figs[mode_key]):
                                with cols[i % 3]:
                                    st.image(f"data:image/png;base64,{b64_img}", use_container_width=True)

            st.markdown("---")
            st.write("### 📥 ダウンロード")
            col_dl1, col_dl2, col_dl3 = st.columns(3)
            date_str = datetime.datetime.now().strftime("%y%m%d")
            with col_dl1:
                st.download_button("📊 速報一覧をExcelでダウンロード", data=st.session_state.excel_data,
                                   file_name=f"{date_str}_遮音性能_速報一覧.xlsx",
                                   mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                                   use_container_width=True)
            with col_dl2:
                st.download_button("📄 グラフ付きHTMLでダウンロード（PDF印刷用）", data=st.session_state.html_content,
                                   file_name=f"{date_str}_遮音性能_速報一覧.html", mime="text/html",
                                   use_container_width=True)
            with col_dl3:
                st.download_button("⚙️ マクロ連携用 リスト (.txt)", data=st.session_state.macro_data,
                                   file_name=f"{date_str}_マクロ連携用リスト.txt", mime="text/plain",
                                   use_container_width=True)