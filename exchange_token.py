
import os
from fyers_apiv3 import fyersModel
from dotenv import load_dotenv

# Load Env for Secret/Client ID
load_dotenv()

client_id = os.getenv("FYERS_CLIENT_ID") or "SWO2VFH9PP-100"
secret_key = os.getenv("FYERS_SECRET") or "2XQJ9KWTLR"
redirect_uri = "https://trade.fyers.in/api-login/redirect-url"
response_type = "code" 
grant_type = "authorization_code"

# The Auth Code provided by user
auth_code = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJhcHBfaWQiOiJTV08yVkZIOVBQIiwidXVpZCI6IjcxM2Q4MWQ2Yjg5NjQyM2M5ZDcyNmE2OTIwNTNmZWM3IiwiaXBBZGRyIjoiIiwibm9uY2UiOiIiLCJzY29wZSI6IiIsImRpc3BsYXlfbmFtZSI6IlhTOTU0OTMiLCJvbXMiOiJLMSIsImhzbV9rZXkiOiJkNGIxNWRlN2RjYzBlZTRiYTIwMjY1ODhiYmNhMTEwZDQ0ODEzNjUyMzIyMDYxYzE0MzUyODJlZCIsImlzRGRwaUVuYWJsZWQiOiJZIiwiaXNNdGZFbmFibGVkIjoiWSIsImF1ZCI6IltcImQ6MVwiLFwiZDoyXCIsXCJ4OjBcIixcIng6MVwiLFwieDoyXCJdIiwiZXhwIjoxNzcwMzg0MjUzLCJpYXQiOjE3NzAzNTQyNTMsImlzcyI6ImFwaS5sb2dpbi5meWVycy5pbiIsIm5iZiI6MTc3MDM1NDI1Mywic3ViIjoiYXV0aF9jb2RlIn0.M7a-0kVvlnsam2LBtUraU5Tr6kdVD_mUdYBbXHjvYT8"

session = fyersModel.SessionModel(
    client_id=client_id,
    secret_key=secret_key,
    redirect_uri=redirect_uri,
    response_type=response_type,
    grant_type=grant_type
)

# Set the auth code
session.set_token(auth_code)

# Generate Access Token
try:
    response = session.generate_token()
    print(f"Response: {response}")
    
    if "access_token" in response:
        print(f"ACCESS_TOKEN_FOUND: {response['access_token']}")
    else:
        print("Failed to get access token.")
        
except Exception as e:
    print(f"Error: {e}")
