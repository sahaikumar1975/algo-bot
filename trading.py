"""Trading automation template.

This module provides helper functions for position sizing and placeholders to integrate with brokers.
DO NOT use in production without adding order safeguards, logging, retries, and paper testing.
"""
from typing import Dict


def compute_quantity_from_risk(capital: float, price: float, risk_per_trade: float, stop_loss_price: float) -> int:
    """Compute position size (quantity) given capital, risk per trade (fraction), and stop-loss price.

    risk_per_trade: e.g. 0.01 for 1% of capital
    stop_loss_price: absolute stop price; assumes long trade
    Returns integer quantity (rounded down)
    """
    if price <= stop_loss_price:
        return 0
    risk_amount = capital * risk_per_trade
    sl_distance = abs(price - stop_loss_price)
    qty = int(risk_amount // sl_distance)
    return max(qty, 0)


def place_order_broker_side_placeholder(symbol: str, qty: int, side: str, price: float = None) -> Dict:
    """Placeholder: implement order placement with broker SDK (Kite/Upstox/IB).

    side: 'BUY' or 'SELL'
    """
    raise NotImplementedError('Integrate with broker SDK here - this is a placeholder')


def trailing_stop_logic(entry_price: float, current_price: float, trailing_pct: float, last_trail_price: float) -> float:
    """Compute updated trailing stop price.

    trailing_pct: e.g. 0.02 for 2% trailing
    last_trail_price: previous trailing stop level
    Returns updated trailing stop price (won't reduce on adverse moves).
    """
    new_trail = current_price * (1 - trailing_pct)
    # only move trailing stop up for long positions
    return max(last_trail_price, new_trail)


if __name__ == '__main__':
    print('trading templates loaded')
