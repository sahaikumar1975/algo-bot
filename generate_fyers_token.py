import os
import sys
from fyers_integration import FyersApp
from dotenv import load_dotenv

# Load existing .env
env_path = "/Users/sahaikumar/Projects/SMA2150/.env"
load_dotenv(env_path)

# Parameters
CLIENT_ID = os.environ.get("FYERS_APP_ID") or os.environ.get("FYERS_CLIENT_ID")
SECRET_KEY = os.environ.get("FYERS_SECRET")

if not CLIENT_ID or not SECRET_KEY:
    print("❌ Error: FYERS_APP_ID or FYERS_SECRET missing in .env")
    print("Please inspect .env file.")
    sys.exit(1)

print(f"🔹 Client ID: {CLIENT_ID}")

# Init App
app = FyersApp(CLIENT_ID, SECRET_KEY)

# 1. Get Login URL
url = app.get_login_url()

if len(sys.argv) < 2:
    print("\n👉 TO LOGIN:")
    print("1. Click this URL: ")
    print(f"\n{url}\n")
    print("2. Login to Fyers.")
    print("3. You will be redirected to a URL like 'https://trade.fyers.in...?auth_code=...'")
    print("4. COPY that entire URL (or just the auth_code part).")
    print("5. Tell me the code/URL.")
    sys.exit(0)

# 2. Get Auth Code from Args
auth_code = sys.argv[1].strip()

# Cleanup Code (if full URL pasted)
if "auth_code=" in auth_code:
    auth_code = auth_code.split("auth_code=")[1].split("&")[0]

print(f"\nUsing Auth Code: {auth_code[:10]}...")

# 3. Generate Token
try:
    token = app.generate_access_token(auth_code)
    print("\n✅ SUCCESS! Access Token Generated.")
    print(f"Token: {token[:20]}...{token[-10:]}")
    
    # 4. Save to .env
    print("\n💾 Updating .env file...")
    
    with open(env_path, 'r') as f:
        lines = f.readlines()
    
    new_lines = []
    token_saved = False
    client_id_saved = False
    
    for line in lines:
        if line.startswith("FYERS_TOKEN="):
            new_lines.append(f"FYERS_TOKEN={token}\n")
            token_saved = True
        elif line.startswith("FYERS_CLIENT_ID="):
            new_lines.append(f"FYERS_CLIENT_ID={CLIENT_ID}\n")
            client_id_saved = True
        else:
            new_lines.append(line)
            
    if not token_saved:
        new_lines.append(f"FYERS_TOKEN={token}\n")
    if not client_id_saved:
        new_lines.append(f"FYERS_CLIENT_ID={CLIENT_ID}\n")
        
    with open(env_path, 'w') as f:
        f.writelines(new_lines)
        
    print("✅ .env updated successfully.")
    print("You can now run the trade test script.")

except Exception as e:
    print(f"\n❌ FAILED to generate token: {e}")
