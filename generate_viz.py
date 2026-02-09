import matplotlib.pyplot as plt
import matplotlib.patches as patches

def draw_candle(ax, x, open_p, high, low, close, color):
    # Wick
    ax.plot([x, x], [low, high], color='black', linewidth=1)
    # Body
    height = close - open_p
    rect = patches.Rectangle((x - 0.25, open_p), 0.5, height, linewidth=1, edgecolor='black', facecolor=color)
    ax.add_patch(rect)

fig, ax = plt.subplots(figsize=(10, 6))

# Mock 9EMA Line
ema_y = [100, 100.2, 100.5, 100.8, 101.0, 101.2]
ax.plot(range(6), ema_y, label='9 EMA', color='blue', linewidth=2)

# Candle 1: Crossing Up (Green)
# "Direction from below towards 9EMA" -> Crossing?
# "Candle closed 9ema" -> Crossover
draw_candle(ax, 1, 99.5, 101.0, 99.2, 100.8, 'green') # Closes above (100.8 > 100.2)

# Candle 2: Holding (Green)
draw_candle(ax, 2, 100.9, 101.5, 100.9, 101.4, 'green') # Closes above (101.4 > 100.5)

# Candle 3: The Red Candle (Pullback)
# "One candle amongst the three should be red"
# "Should not close below 9EMA" (9EMA is 100.8 here)
draw_candle(ax, 3, 101.4, 101.6, 101.1, 101.2, 'red') # Closes 101.2 > 100.8. Open 101.4.

# Entry Line: "Red candle close shall be the reference line"
# Red Candle Close is 101.2.
ax.axhline(y=101.2, color='orange', linestyle='--', label='Entry Ref (Red Close)')

# Stop Loss: "Low of the three candles"
# Lows: C1=99.2, C2=100.9, C3=101.1.
# SL = 99.2 (Low of C1).
ax.axhline(y=99.2, color='red', linestyle=':', label='Stop Loss (Lowest Low)')

# Annotations
ax.text(1, 98.5, "1. Crossing Candle\n(Closes Above)", ha='center')
ax.text(3, 102.0, "3. Red Candle\n(Closes Above)", ha='center')
ax.text(3.5, 101.2, " Entry > Red Close?", color='orange', va='center')
ax.text(3.5, 99.2, " SL (Lowest Low)", color='red', va='center')

ax.set_title("Visualizing Your Strategy (Buying CE)")
ax.set_xlim(0, 5)
ax.set_ylim(98, 103)
ax.legend()
ax.grid(True, alpha=0.3)

plt.savefig("strategy_viz.png")
print("Chart generated: strategy_viz.png")
