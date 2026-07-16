# -*- coding: utf-8 -*-
"""
Phase 13 - P13-T1: Gate.io 真实连接只读预检（Level 1，零资金风险）

8 项只读验证：
  1. API 密钥加载（环境变量 / 配置文件）
  2. API 权限验证（需要密钥）
  3. 账户余额查询（需要密钥）
  4. 合约信息查询（公共 API）
  5. 真实 ticker 查询（公共 API）
  6. 真实 K线查询（公共 API）
  7. 真实订单簿查询（公共 API）
  8. 限流探测（公共 API）

硬边界：零资金风险（只读，不下单）；无密钥项 BLOCKED（非降级，非跳过）。
范围：仅 Gate.io USDT 本位永续合约（数字货币合约）。
"""

from __future__ import annotations

import json
import os
import sys
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

ARTIFACTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "phase13_artifacts")
os.makedirs(ARTIFACTS_DIR, exist_ok=True)

GATEIO_API_BASE = "https://api.gateio.ws/api/v4"
SYMBOL = "BTC_USDT"


def _http_get(url: str, timeout: float = 10.0) -> Tuple[Optional[Any], Optional[str], float]:
    t0 = time.time()
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "Hermes-Phase13/1.0"})
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            return data, None, (time.time() - t0) * 1000.0
    except urllib.error.HTTPError as e:
        return None, f"HTTP {e.code}: {e.reason}", (time.time() - t0) * 1000.0
    except Exception as e:
        return None, str(e), (time.time() - t0) * 1000.0


def _load_api_keys() -> Tuple[str, str]:
    """从环境变量和 .env 文件加载 API 密钥"""
    api_key = os.environ.get("GATE_API_KEY", "")
    api_secret = os.environ.get("GATE_API_SECRET", "")

    env_file = "/home/lmy/hermes_v6/.env"
    if os.path.exists(env_file):
        with open(env_file, "r") as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    k, v = line.split("=", 1)
                    k = k.strip().upper()
                    v = v.strip().strip('"').strip("'")
                    if k in ("GATE_API_KEY", "GATEIO_API_KEY", "GATE_KEY") and not api_key:
                        api_key = v
                    elif k in ("GATE_API_SECRET", "GATEIO_API_SECRET", "GATE_SECRET") and not api_secret:
                        api_secret = v
    return api_key, api_secret


def check1_api_key_loading() -> Dict[str, Any]:
    result = {"name": "check1_api_key_loading", "status": "BLOCKED", "details": {}}
    api_key, api_secret = _load_api_keys()
    env_key = os.environ.get("GATE_API_KEY", "")
    env_secret = os.environ.get("GATE_API_SECRET", "")

    result["details"] = {
        "env_var_GATE_API_KEY": "YES" if env_key else "NO",
        "env_var_GATE_API_SECRET": "YES" if env_secret else "NO",
        "has_key": bool(api_key),
        "has_secret": bool(api_secret),
    }
    if api_key and api_secret:
        result["status"] = "PASS"
        result["details"]["source"] = "env_var" if env_key else "env_file"
    else:
        result["details"]["reason"] = "Gate.io API 密钥未配置（需要在 .env 中添加 GATE_API_KEY 和 GATE_API_SECRET）"
    return result


def check2_api_permissions(api_key: str, api_secret: str) -> Dict[str, Any]:
    result = {"name": "check2_api_permissions", "status": "BLOCKED", "details": {}}
    if not api_key or not api_secret:
        result["details"]["reason"] = "API 密钥未配置，无法验证权限"
        return result
    try:
        import ccxt
        exchange = ccxt.gate({"apiKey": api_key, "secret": api_secret, "enableRateLimit": True})
        balance = exchange.fetch_balance()
        result["status"] = "PASS"
        result["details"] = {"permissions": "read"}
    except Exception as e:
        result["status"] = "BLOCKED"
        result["details"]["reason"] = f"API 权限验证失败: {e}"
    return result


def check3_account_balance(api_key: str, api_secret: str) -> Dict[str, Any]:
    result = {"name": "check3_account_balance", "status": "BLOCKED", "details": {}}
    if not api_key or not api_secret:
        result["details"]["reason"] = "API 密钥未配置，无法查询余额"
        return result
    try:
        import ccxt
        exchange = ccxt.gate({"apiKey": api_key, "secret": api_secret, "enableRateLimit": True})
        balance = exchange.fetch_balance({"type": "swap"})
        usdt_free = balance.get("USDT", {}).get("free", 0.0)
        usdt_total = balance.get("USDT", {}).get("total", 0.0)
        result["details"] = {"USDT_free": usdt_free, "USDT_total": usdt_total, "sufficient": usdt_free >= 10.0}
        if usdt_free >= 10.0:
            result["status"] = "PASS"
        else:
            result["details"]["reason"] = f"USDT 余额不足: {usdt_free} < 10 USDT"
    except Exception as e:
        result["status"] = "BLOCKED"
        result["details"]["reason"] = f"余额查询失败: {e}"
    return result


def check4_contract_info() -> Dict[str, Any]:
    result = {"name": "check4_contract_info", "status": "FAIL", "details": {}}
    url = f"{GATEIO_API_BASE}/futures/usdt/contracts/{SYMBOL}"
    data, error, latency = _http_get(url)
    if error:
        result["details"]["error"] = error
        return result
    required = ["name", "type", "quanto_multiplier", "maintenance_rate", "order_size_min"]
    missing = [f for f in required if f not in data]
    result["details"] = {
        "symbol": data.get("name"),
        "type": data.get("type"),
        "quanto_multiplier": data.get("quanto_multiplier"),
        "order_size_min": data.get("order_size_min"),
        "leverage_max": data.get("leverage_max"),
        "latency_ms": round(latency, 2),
        "missing_fields": missing,
    }
    if not missing and data.get("name") == SYMBOL:
        result["status"] = "PASS"
    else:
        result["details"]["reason"] = f"字段缺失: {missing}"
    return result


def check5_real_ticker() -> Dict[str, Any]:
    result = {"name": "check5_real_ticker", "status": "FAIL", "details": {}}
    url = f"{GATEIO_API_BASE}/futures/usdt/tickers?contract={SYMBOL}"
    data, error, latency = _http_get(url)
    if error:
        result["details"]["error"] = error
        return result
    if not isinstance(data, list) or len(data) == 0:
        result["details"]["reason"] = "ticker 返回空列表"
        return result
    t = data[0]
    last = float(t.get("last", 0) or 0)
    mark = float(t.get("mark_price", 0) or 0)
    result["details"] = {
        "last_price": last,
        "mark_price": mark,
        "funding_rate": float(t.get("funding_rate", 0) or 0),
        "volume_24h_usd": float(t.get("volume_24h_settle", 0) or 0),
        "latency_ms": round(latency, 2),
    }
    if last > 0 and mark > 0:
        result["status"] = "PASS"
    else:
        result["details"]["reason"] = f"价格异常: last={last}, mark={mark}"
    return result


def check6_real_klines() -> Dict[str, Any]:
    result = {"name": "check6_real_klines", "status": "FAIL", "details": {}}
    url = f"{GATEIO_API_BASE}/futures/usdt/candlesticks?contract={SYMBOL}&interval=1h&limit=100"
    data, error, latency = _http_get(url)
    if error:
        result["details"]["error"] = error
        return result
    if not isinstance(data, list) or len(data) == 0:
        result["details"]["reason"] = "K线返回空列表"
        return result
    last = data[-1]
    # Gate.io v4 futures candlesticks 返回字典: {t, v, c, h, l, o, n}
    if isinstance(last, dict):
        o = float(last.get("o", 0))
        h = float(last.get("h", 0))
        l = float(last.get("l", 0))
        c = float(last.get("c", 0))
        v = float(last.get("v", 0))
        t = last.get("t", 0)
    else:
        # 兼容列表格式 [t, v, c, h, l, o, n]
        o = float(last[5]) if len(last) > 5 else 0
        h = float(last[3]) if len(last) > 3 else 0
        l = float(last[4]) if len(last) > 4 else 0
        c = float(last[2]) if len(last) > 2 else 0
        v = float(last[1]) if len(last) > 1 else 0
        t = last[0] if len(last) > 0 else 0
    result["details"] = {
        "count": len(data),
        "last_timestamp": t,
        "last_close": c,
        "last_high": h,
        "last_low": l,
        "last_open": o,
        "last_volume": v,
        "latency_ms": round(latency, 2),
    }
    if len(data) >= 10:
        if o > 0 and h >= l and h >= o and h >= c and l <= o and l <= c:
            result["status"] = "PASS"
        else:
            result["details"]["reason"] = f"OHLC错误: o={o} h={h} l={l} c={c}"
    else:
        result["details"]["reason"] = f"K线不足: {len(data)}"
    return result


def check7_real_orderbook() -> Dict[str, Any]:
    result = {"name": "check7_real_orderbook", "status": "FAIL", "details": {}}
    url = f"{GATEIO_API_BASE}/futures/usdt/order_book?contract={SYMBOL}&limit=20"
    data, error, latency = _http_get(url)
    if error:
        result["details"]["error"] = error
        return result
    bids = data.get("bids", [])
    asks = data.get("asks", [])
    if not bids or not asks:
        result["details"]["reason"] = "订单簿空"
        return result
    # Gate.io v4 返回 {"p": price, "s": size} 字典格式
    if isinstance(bids[0], dict):
        bp = [float(b.get("p", 0)) for b in bids]
        ap = [float(a.get("p", 0)) for a in asks]
    else:
        bp = [float(b[0]) for b in bids]
        ap = [float(a[0]) for a in asks]
    bd = all(bp[i] >= bp[i + 1] for i in range(len(bp) - 1))
    aa = all(ap[i] <= ap[i + 1] for i in range(len(ap) - 1))
    spread_pct = (ap[0] - bp[0]) / ap[0] if ap[0] > 0 else 0
    result["details"] = {
        "bids": len(bids),
        "asks": len(asks),
        "best_bid": bp[0],
        "best_ask": ap[0],
        "spread_pct": round(spread_pct, 6),
        "bids_desc": bd,
        "asks_asc": aa,
        "latency_ms": round(latency, 2),
    }
    if bd and aa and spread_pct < 0.005 and bp[0] > 0 and ap[0] > 0:
        result["status"] = "PASS"
    else:
        result["details"]["reason"] = f"异常: bd={bd} aa={aa} spread={spread_pct:.6f}"
    return result


def check8_rate_limit_probe() -> Dict[str, Any]:
    result = {"name": "check8_rate_limit_probe", "status": "FAIL", "details": {}}
    url = f"{GATEIO_API_BASE}/futures/usdt/tickers?contract={SYMBOL}"
    latencies = []
    errors = []
    rl = False
    for i in range(10):
        data, error, latency = _http_get(url, timeout=5.0)
        if error:
            errors.append(f"req{i}: {error}")
            if "429" in error or "rate" in error.lower():
                rl = True
        else:
            latencies.append(latency)
        time.sleep(0.2)
    avg = sum(latencies) / len(latencies) if latencies else 0
    result["details"] = {
        "successful": len(latencies),
        "failed": len(errors),
        "rate_limited": rl,
        "avg_ms": round(avg, 2),
        "min_ms": round(min(latencies), 2) if latencies else 0,
        "max_ms": round(max(latencies), 2) if latencies else 0,
    }
    if len(latencies) >= 8 and not rl:
        result["status"] = "PASS"
    elif rl:
        result["status"] = "BLOCKED"
        result["details"]["reason"] = "触发限流"
    else:
        result["details"]["reason"] = f"成功率不足: {len(latencies)}/10"
    return result


def main() -> int:
    print("=" * 72)
    print("Phase 13 - P13-T1: Gate.io 真实连接只读预检")
    print(f"时间: {datetime.now(timezone.utc).isoformat()}")
    print(f"交易所: Gate.io | 合约: {SYMBOL} (USDT 永续)")
    print(f"范围: 仅数字货币合约 | 风险: Level 1 (零资金)")
    print("=" * 72)

    api_key, api_secret = _load_api_keys()

    checks = []
    checks.append(check1_api_key_loading())
    checks.append(check2_api_permissions(api_key, api_secret))
    checks.append(check3_account_balance(api_key, api_secret))
    checks.append(check4_contract_info())
    checks.append(check5_real_ticker())
    checks.append(check6_real_klines())
    checks.append(check7_real_orderbook())
    checks.append(check8_rate_limit_probe())

    p = sum(1 for c in checks if c["status"] == "PASS")
    b = sum(1 for c in checks if c["status"] == "BLOCKED")
    fl = sum(1 for c in checks if c["status"] == "FAIL")

    print()
    print("-" * 72)
    print("验证结果:")
    print("-" * 72)
    for c in checks:
        e = "\u2705" if c["status"] == "PASS" else ("\u26d4" if c["status"] == "BLOCKED" else "\u274c")
        print(f"  {e} {c['name']}: {c['status']}")
        if c["status"] != "PASS" and "reason" in c.get("details", {}):
            print(f"     原因: {c['details']['reason']}")
        for k, v in c.get("details", {}).items():
            if k != "reason" and v is not None and isinstance(v, (int, float, str, bool)):
                print(f"     {k}: {v}")
    print("-" * 72)
    print(f"PASS: {p}/8 | BLOCKED: {b}/8 | FAIL: {fl}/8")
    print("-" * 72)

    report = {
        "phase": "Phase 13",
        "task": "P13-T1",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "exchange": "Gate.io",
        "symbol": SYMBOL,
        "contract_type": "USDT perpetual",
        "scope": "crypto contracts only",
        "risk_level": "Level 1",
        "checks": checks,
        "summary": {"total": 8, "pass": p, "blocked": b, "fail": fl, "all_pass": p == 8, "has_api_key": bool(api_key)},
    }
    report_path = os.path.join(ARTIFACTS_DIR, "connection_preflight_report.json")
    with open(report_path, "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2, ensure_ascii=False)
    print(f"\n报告: {report_path}")
    return 2 if fl > 0 else (1 if b > 0 else 0)


if __name__ == "__main__":
    sys.exit(main())
