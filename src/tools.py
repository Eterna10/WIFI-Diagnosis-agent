# tools.py
"""
Wi-Fi 诊断 Agent 的工具层。

每个 Tool 都是一个普通 Python 函数：只负责查询数据或完成计算，返回 dict，
不直接给出"根本原因"——原因由 Agent（LLM）根据工具返回的证据推理得出。

TOOL_SCHEMAS 是提供给 LLM 的 function calling 描述，
execute_tool() 是统一的调用入口，负责参数解析和异常处理。
"""
import json
import math
from difflib import get_close_matches
from pathlib import Path
from typing import Any, Dict, Optional

DATA_FILE = Path(__file__).resolve().parent.parent / "data" / "sample_data.json"

# 802.11ax 单空间流、0.8µs GI 的 PHY 速率（Mbps），索引为 MCS 0~11
HE_PHY_RATE_1SS = {
    20: [8.6, 17.2, 25.8, 34.4, 51.6, 68.8, 77.4, 86.0, 103.2, 114.7, 129.0, 143.4],
    40: [17.2, 34.4, 51.6, 68.8, 103.2, 137.6, 154.9, 172.1, 206.5, 229.4, 258.1, 286.8],
    80: [36.0, 72.1, 108.1, 144.1, 216.2, 288.2, 324.3, 360.3, 432.4, 480.4, 540.4, 600.5],
    160: [72.1, 144.1, 216.2, 288.2, 432.4, 576.5, 648.5, 720.6, 864.7, 960.8, 1080.9, 1201.0],
}
MCS_MODULATION = [
    "BPSK 1/2", "QPSK 1/2", "QPSK 3/4", "16-QAM 1/2", "16-QAM 3/4", "64-QAM 2/3",
    "64-QAM 3/4", "64-QAM 5/6", "256-QAM 3/4", "256-QAM 5/6", "1024-QAM 3/4", "1024-QAM 5/6",
]
# 各 MCS 稳定工作所需的最低 SNR（dB），为工程经验近似值
MCS_MIN_SNR_DB = [5, 8, 11, 14, 18, 22, 24, 26, 30, 32, 35, 38]
# MAC 层效率：扣除帧头、ACK、竞争退避等开销后，实际吞吐量约为 PHY 速率的 60%~70%
MAC_EFFICIENCY = 0.65


class ToolInputError(ValueError):
    """工具输入不合法"""


def _load_data() -> Dict[str, Any]:
    with open(DATA_FILE, encoding="utf-8") as f:
        return json.load(f)


def _find_ap(ap_id: str) -> Dict[str, Any]:
    data = _load_data()
    aps = {ap["ap_id"].upper(): ap for ap in data["access_points"]}
    key = (ap_id or "").strip().upper()
    if key not in aps:
        suggestions = get_close_matches(key, list(aps), n=3, cutoff=0.4)
        hint = f"，是否想查询：{', '.join(suggestions)}" if suggestions else ""
        raise ToolInputError(f"AP 编号 '{ap_id}' 不存在{hint}。可先调用 list_access_points 按位置查找 AP。")
    return aps[key]


def list_access_points(keyword: str = "") -> Dict[str, Any]:
    """按位置、楼宇或 AP 编号关键词查找接入点"""
    data = _load_data()
    kw = (keyword or "").strip().lower()
    matches = [
        {"ap_id": ap["ap_id"], "location": ap["location"], "building": ap["building"], "status": ap["status"]}
        for ap in data["access_points"]
        if not kw or kw in ap["location"].lower() or kw in ap["building"].lower() or kw in ap["ap_id"].lower()
    ]
    result = {"keyword": keyword, "match_count": len(matches), "access_points": matches}
    if not matches:
        result["note"] = "没有匹配的 AP，请换一个更短的关键词（例如楼宇名称），或传入空字符串列出全部 AP。"
    return result


def get_ap_metrics(ap_id: str) -> Dict[str, Any]:
    """查询某个 AP 的无线侧监控指标，并计算 SNR"""
    ap = _find_ap(ap_id)
    fields = ["band", "channel", "bandwidth_mhz", "spatial_streams", "rssi_dbm", "noise_floor_dbm",
              "connected_clients", "active_clients", "channel_utilization_pct", "retry_rate_pct",
              "same_channel_neighbor_aps"]
    metrics = {k: ap.get(k) for k in fields}

    # SNR(dB) = RSSI(dBm) - 底噪(dBm)
    if ap.get("rssi_dbm") is not None and ap.get("noise_floor_dbm") is not None:
        metrics["snr_db"] = ap["rssi_dbm"] - ap["noise_floor_dbm"]
    else:
        metrics["snr_db"] = None

    missing = [k for k, v in metrics.items() if v is None]
    result = {
        "ap_id": ap["ap_id"],
        "location": ap["location"],
        "status": ap["status"],
        "collected_at": _load_data()["collected_at"],
        "metrics": metrics,
        "data_quality": "complete" if not missing else "incomplete",
    }
    if ap.get("status_detail"):
        result["status_detail"] = ap["status_detail"]
    if missing:
        result["missing_fields"] = missing
    return result


def get_uplink_status(ap_id: str) -> Dict[str, Any]:
    """查询 AP 有线上联链路的状态，计算链路利用率"""
    ap = _find_ap(ap_id)
    up = ap.get("uplink")
    if not up:
        return {"ap_id": ap["ap_id"], "data_quality": "unavailable",
                "note": "该 AP 没有可用的上联链路监控数据（可能采集失败）。"}

    speed = up["negotiated_speed_mbps"]
    # 全双工链路，两个方向分别计算利用率，取较大者
    util_rx = up["rx_mbps"] / speed * 100
    util_tx = up["tx_mbps"] / speed * 100
    return {
        "ap_id": ap["ap_id"],
        "switch": up["switch"],
        "port": up["port"],
        "negotiated_speed_mbps": speed,
        "rx_mbps": up["rx_mbps"],
        "tx_mbps": up["tx_mbps"],
        "utilization_rx_pct": round(util_rx, 1),
        "utilization_tx_pct": round(util_tx, 1),
        "peak_direction_utilization_pct": round(max(util_rx, util_tx), 1),
        "rtt_to_gateway_ms": up["rtt_to_gateway_ms"],
        "packet_loss_pct": up["packet_loss_pct"],
        "crc_errors_last_hour": up["crc_errors_last_hour"],
        "data_quality": "complete",
    }


def analyze_wireless_capacity(
    bandwidth_mhz: float,
    snr_db: float,
    spatial_streams: int = 1,
    active_clients: Optional[int] = None,
    channel_utilization_pct: Optional[float] = None,
) -> Dict[str, Any]:
    """
    根据带宽和 SNR 计算：
    1. 香农理论容量 C = B * log2(1 + SNR_linear)，SNR_linear = 10^(SNR_dB/10)
    2. 按 SNR 估计可稳定使用的 802.11ax MCS、调制方式和 PHY 速率
    3. 扣除 MAC 开销后的实际吞吐量估计，以及多用户共享时的人均带宽
    """
    try:
        bandwidth_mhz = float(bandwidth_mhz)
        snr_db = float(snr_db)
        spatial_streams = int(spatial_streams)
    except (TypeError, ValueError):
        raise ToolInputError("bandwidth_mhz、snr_db 必须是数字，spatial_streams 必须是整数。")
    if bandwidth_mhz <= 0:
        raise ToolInputError(f"带宽必须为正数，收到 {bandwidth_mhz} MHz。")
    if not -10 <= snr_db <= 60:
        raise ToolInputError(f"SNR={snr_db} dB 超出合理范围（-10~60 dB），请检查输入是否把线性值当成了 dB。")
    if not 1 <= spatial_streams <= 4:
        raise ToolInputError(f"空间流数应为 1~4，收到 {spatial_streams}。")
    if active_clients is not None and int(active_clients) < 0:
        raise ToolInputError("active_clients 不能为负数。")
    if channel_utilization_pct is not None and not 0 <= float(channel_utilization_pct) <= 100:
        raise ToolInputError("channel_utilization_pct 应在 0~100 之间。")

    snr_linear = 10 ** (snr_db / 10)
    shannon_per_stream = bandwidth_mhz * 1e6 * math.log2(1 + snr_linear) / 1e6
    result: Dict[str, Any] = {
        "input": {"bandwidth_mhz": bandwidth_mhz, "snr_db": snr_db, "spatial_streams": spatial_streams},
        "snr_linear": round(snr_linear, 2),
        "shannon_capacity_per_stream_mbps": round(shannon_per_stream, 1),
        # 多流容量按理想 MIMO（各流独立、SNR 相同）估算，是上限
        "shannon_capacity_total_mbps": round(shannon_per_stream * spatial_streams, 1),
        "note": "香农容量是理论上限，不等于实际吞吐量。",
    }

    bw_key = int(bandwidth_mhz) if bandwidth_mhz in HE_PHY_RATE_1SS else None
    supported = [i for i, th in enumerate(MCS_MIN_SNR_DB) if snr_db >= th]
    if bw_key is None:
        result["mcs_estimate"] = None
        result["mcs_note"] = "802.11 信道带宽只有 20/40/80/160 MHz，非标准带宽不估计 MCS。"
    elif not supported:
        result["mcs_estimate"] = None
        result["mcs_note"] = f"SNR={snr_db} dB 低于 MCS0 所需的约 {MCS_MIN_SNR_DB[0]} dB，链路难以稳定工作。"
    else:
        mcs = supported[-1]
        phy = HE_PHY_RATE_1SS[bw_key][mcs] * spatial_streams
        throughput = phy * MAC_EFFICIENCY
        result["mcs_estimate"] = {
            "standard": "802.11ax (Wi-Fi 6), 0.8µs GI",
            "mcs": mcs,
            "modulation": MCS_MODULATION[mcs],
            "min_snr_required_db": MCS_MIN_SNR_DB[mcs],
            "phy_rate_mbps": round(phy, 1),
            "max_phy_rate_at_mcs11_mbps": round(HE_PHY_RATE_1SS[bw_key][11] * spatial_streams, 1),
            "phy_rate_ratio_vs_best_pct": round(phy / (HE_PHY_RATE_1SS[bw_key][11] * spatial_streams) * 100, 1),
            "estimated_max_throughput_mbps": round(throughput, 1),
            "mac_efficiency_assumed": MAC_EFFICIENCY,
        }
        if active_clients:
            share = throughput / int(active_clients)
            result["per_active_client_share_mbps"] = round(share, 1)
            result["sharing_note"] = f"空口为 {int(active_clients)} 个活跃终端共享（理想公平分配的估计值）。"
        if channel_utilization_pct is not None:
            result["channel_idle_airtime_pct"] = round(100 - float(channel_utilization_pct), 1)
    return result


# ---------- 提供给 LLM 的工具描述（OpenAI function calling 格式） ----------
TOOL_SCHEMAS = [
    {
        "type": "function",
        "function": {
            "name": "list_access_points",
            "description": "按位置、楼宇名称或 AP 编号关键词查找校园无线接入点(AP)，返回 AP 编号、位置和在线状态。用户只描述了地点时先用它找到 AP 编号。",
            "parameters": {
                "type": "object",
                "properties": {
                    "keyword": {"type": "string", "description": "关键词，如 '图书馆'、'宿舍5号楼'；空字符串表示列出全部 AP"}
                },
                "required": ["keyword"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_ap_metrics",
            "description": "查询指定 AP 的无线侧监控指标：频段、信道、带宽、空间流、RSSI、底噪、SNR、连接/活跃终端数、信道利用率、重传率、同信道邻居 AP 数。",
            "parameters": {
                "type": "object",
                "properties": {"ap_id": {"type": "string", "description": "AP 编号，例如 AP-LIB-3F-01"}},
                "required": ["ap_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_uplink_status",
            "description": "查询指定 AP 的有线上联链路状态：协商速率、收发流量、链路利用率、到网关 RTT、丢包率、CRC 错误数。用于判断瓶颈是否在有线侧。",
            "parameters": {
                "type": "object",
                "properties": {"ap_id": {"type": "string", "description": "AP 编号"}},
                "required": ["ap_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "analyze_wireless_capacity",
            "description": "无线链路容量计算：用香农公式计算理论容量（自动把 SNR 从 dB 换算为线性值），按 SNR 估计 802.11ax 可用 MCS/调制方式/PHY 速率，估算扣除 MAC 开销后的实际吞吐量和每个活跃终端可分到的带宽。",
            "parameters": {
                "type": "object",
                "properties": {
                    "bandwidth_mhz": {"type": "number", "description": "信道带宽 MHz，如 20/40/80/160"},
                    "snr_db": {"type": "number", "description": "信噪比，单位 dB"},
                    "spatial_streams": {"type": "integer", "description": "空间流数，1~4，默认 1"},
                    "active_clients": {"type": "integer", "description": "可选，同时活跃的终端数"},
                    "channel_utilization_pct": {"type": "number", "description": "可选，信道利用率 0~100"},
                },
                "required": ["bandwidth_mhz", "snr_db"],
            },
        },
    },
]

TOOL_FUNCTIONS = {
    "list_access_points": list_access_points,
    "get_ap_metrics": get_ap_metrics,
    "get_uplink_status": get_uplink_status,
    "analyze_wireless_capacity": analyze_wireless_capacity,
}


def execute_tool(name: str, arguments: str) -> Dict[str, Any]:
    """统一的工具调用入口：解析 LLM 给出的 JSON 参数，执行工具，捕获所有异常"""
    if name not in TOOL_FUNCTIONS:
        return {"error": f"未知工具 '{name}'，可用工具：{', '.join(TOOL_FUNCTIONS)}"}
    try:
        args = json.loads(arguments) if arguments else {}
    except json.JSONDecodeError as e:
        return {"error": f"工具参数不是合法 JSON：{e}"}
    try:
        return TOOL_FUNCTIONS[name](**args)
    except ToolInputError as e:
        return {"error": str(e)}
    except TypeError as e:
        return {"error": f"参数错误：{e}"}
    except Exception as e:  # 数据文件损坏等意外情况，不让整个 Agent 崩溃
        return {"error": f"工具执行失败：{type(e).__name__}: {e}"}
