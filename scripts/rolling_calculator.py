#!/usr/bin/env python3
"""
Rolling Position Calculator
Calculates safe add sizes that keep liquidation price below a defined safety level.
"""

def calculate_rolling_add(
    entry_price: float,
    current_price: float,
    position_btc: float,
    initial_margin: float,
    target_leverage: float,
    max_liq_price: float,  # Must stay below this (e.g., previous swing low)
    maintenance_margin_rate: float = 0.005,  # Bybit's 0.5% for BTC
):
    """Calculate max safe add size at current price."""
    
    # Current state
    unrealised_pnl = position_btc * (current_price - entry_price)
    total_equity = initial_margin + unrealised_pnl
    current_liq = entry_price - (initial_margin / position_btc) * (1 - maintenance_margin_rate)
    effective_leverage = (position_btc * current_price) / total_equity
    
    print(f"=== CURRENT STATE ===")
    print(f"Entry: ${entry_price:,.0f}")
    print(f"Current: ${current_price:,.0f}")
    print(f"Position: {position_btc:.4f} BTC (${position_btc * current_price:,.0f} notional)")
    print(f"Initial Margin: ${initial_margin:,.0f}")
    print(f"Unrealised PnL: ${unrealised_pnl:,.0f}")
    print(f"Total Equity: ${total_equity:,.0f}")
    print(f"Current Liq Price: ${current_liq:,.0f}")
    print(f"Effective Leverage: {effective_leverage:.1f}x")
    print(f"Max Allowed Liq Price: ${max_liq_price:,.0f}")
    print()
    
    # Binary search for max safe add size
    # After adding X BTC at current_price:
    # New position = position_btc + X
    # New avg entry = (position_btc * entry_price + X * current_price) / (position_btc + X)
    # New total margin = total_equity (cross margin uses all available equity)
    # New liq price = new_avg_entry - (total_equity / new_position) * (1 - mmr)
    
    lo, hi = 0.0, position_btc * 10  # search range
    max_safe_add = 0.0
    
    for _ in range(100):  # binary search iterations
        mid = (lo + hi) / 2
        new_pos = position_btc + mid
        new_avg = (position_btc * entry_price + mid * current_price) / new_pos
        new_liq = new_avg - (total_equity / new_pos) * (1 - maintenance_margin_rate)
        
        if new_liq < max_liq_price:
            max_safe_add = mid
            lo = mid
        else:
            hi = mid
    
    if max_safe_add < 0.001:
        print("⚠️  Cannot safely add — liquidation price already too close to safety level!")
        return 0
    
    # Calculate recommended adds using inverted pyramid
    # But cap each at what's safely possible
    new_pos = position_btc + max_safe_add
    new_avg = (position_btc * entry_price + max_safe_add * current_price) / new_pos
    new_liq = new_avg - (total_equity / new_pos) * (1 - maintenance_margin_rate)
    new_leverage = (new_pos * current_price) / total_equity
    
    print(f"=== MAX SAFE ADD ===")
    print(f"Max add: {max_safe_add:.4f} BTC (${max_safe_add * current_price:,.0f})")
    print(f"After add: {new_pos:.4f} BTC total")
    print(f"New avg entry: ${new_avg:,.0f}")
    print(f"New liq price: ${new_liq:,.0f} (safety: ${max_liq_price:,.0f})")
    print(f"New leverage: {new_leverage:.1f}x")
    print()
    
    # Inverted pyramid recommendations
    # Add 1: 150%, Add 2: 200%, Add 3: 250% of base
    base_btc = position_btc
    adds = [
        ("Add 1 (150%)", base_btc * 1.5),
        ("Add 2 (200%)", base_btc * 2.0),
        ("Add 3 (250%)", base_btc * 2.5),
    ]
    
    print(f"=== INVERTED PYRAMID PLAN ===")
    cumulative = position_btc
    cumulative_margin = total_equity
    
    for name, target_add in adds:
        safe = min(target_add, max_safe_add * 0.33)  # Split max across 3 adds
        actual_pos = cumulative + safe
        actual_avg = (cumulative * entry_price + safe * current_price) / actual_pos
        actual_liq = actual_avg - (cumulative_margin / actual_pos) * (1 - maintenance_margin_rate)
        
        status = "✅ SAFE" if actual_liq < max_liq_price else "❌ UNSAFE"
        print(f"{name}: {safe:.4f} BTC (${safe * current_price:,.0f}) | "
              f"Liq: ${actual_liq:,.0f} | {status}")
        cumulative = actual_pos
    
    return max_safe_add


if __name__ == "__main__":
    import sys
    
    print("=" * 60)
    print("  ROLLING POSITION CALCULATOR")
    print("=" * 60)
    print()
    
    # Scenario 1: Current market conditions
    print("### Scenario: Enter now at $68K, target add at $75K ###")
    print()
    calculate_rolling_add(
        entry_price=68000,
        current_price=75000,
        position_btc=1.5,  # ~$10K margin at 10x
        initial_margin=10200,
        target_leverage=10,
        max_liq_price=60000,  # Previous swing low
    )
    
    print()
    print("### Scenario: Enter now at $68K, target add at $85K ###")
    print()
    calculate_rolling_add(
        entry_price=68000,
        current_price=85000,
        position_btc=1.5,
        initial_margin=10200,
        target_leverage=10,
        max_liq_price=60000,
    )
    
    print()
    print("### Scenario: Already added once, add again at $100K ###")
    print()
    calculate_rolling_add(
        entry_price=72000,  # avg after first add
        current_price=100000,
        position_btc=3.0,  # after first add
        initial_margin=10200,
        target_leverage=10,
        max_liq_price=68000,  # raise safety to previous add level
    )
