
with open("dashboard.py", "r") as f:
    lines = f.readlines()

# Block 1: Start to end of Day Trading/Backtest Indices (Line 1 to 1066)
# Python list slice [0:1066] gets lines 0..1065 (which are lines 1..1066 in 1-based)
part1 = lines[0:1066] 

# Block 2: Sector Analysis (Line 1881 to 2154)
# Line 1881 is index 1880.
# Line 2154 is index 2153.
# We want up to 2154 inclusive.
# So slice [1880:2154] goes up to 2153. Correct.
part2 = lines[1880:2154]

# Block 3: End of file (Line 2335 to End)
# Line 2335 is index 2334.
part3 = lines[2334:]

new_content = "".join(part1 + part2 + part3)

with open("dashboard.py", "w") as f:
    f.write(new_content)

print(f"Successfully stitched dashboard.py. New length: {len(part1)+len(part2)+len(part3)} lines.")
