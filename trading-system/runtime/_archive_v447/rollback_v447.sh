#!/bin/bash
# v447 冗余归档回滚脚本
# 生成时间: 2026-06-28T20:18:41
# 归档文件数: 8
set -e
BASE="/mnt/c/Users/Administrator.SK-20260514VARK/Documents/trae_projects/2026/hermes_v6/sandbox_trading"
ARCHIVE="/mnt/c/Users/Administrator.SK-20260514VARK/Documents/trae_projects/2026/hermes_v6/sandbox_trading/_archive_v447"
echo "开始回滚 v447 归档..."
if [ -f "$ARCHIVE/risk_parity_signal.py" ]; then mv "$ARCHIVE/risk_parity_signal.py" "$BASE/risk_parity_signal.py"; echo "已回滚: risk_parity_signal.py"; fi
if [ -f "$ARCHIVE/cppi_portfolio_insurance.py" ]; then mv "$ARCHIVE/cppi_portfolio_insurance.py" "$BASE/cppi_portfolio_insurance.py"; echo "已回滚: cppi_portfolio_insurance.py"; fi
if [ -f "$ARCHIVE/seasonality_cycle.py" ]; then mv "$ARCHIVE/seasonality_cycle.py" "$BASE/seasonality_cycle.py"; echo "已回滚: seasonality_cycle.py"; fi
if [ -f "$ARCHIVE/ml_liquidation_predictor.py" ]; then mv "$ARCHIVE/ml_liquidation_predictor.py" "$BASE/ml_liquidation_predictor.py"; echo "已回滚: ml_liquidation_predictor.py"; fi
if [ -f "$ARCHIVE/event_driven_trading.py" ]; then mv "$ARCHIVE/event_driven_trading.py" "$BASE/event_driven_trading.py"; echo "已回滚: event_driven_trading.py"; fi
if [ -f "$ARCHIVE/mean_reversion_ml.py" ]; then mv "$ARCHIVE/mean_reversion_ml.py" "$BASE/mean_reversion_ml.py"; echo "已回滚: mean_reversion_ml.py"; fi
if [ -f "$ARCHIVE/macro_liquidity_cycle.py" ]; then mv "$ARCHIVE/macro_liquidity_cycle.py" "$BASE/macro_liquidity_cycle.py"; echo "已回滚: macro_liquidity_cycle.py"; fi
if [ -f "$ARCHIVE/sentiment_driven_trading.py" ]; then mv "$ARCHIVE/sentiment_driven_trading.py" "$BASE/sentiment_driven_trading.py"; echo "已回滚: sentiment_driven_trading.py"; fi
echo "v447 归档回滚完成."
