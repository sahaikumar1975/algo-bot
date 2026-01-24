import urllib.parse

def extract_auth_code(input_string):
    print(f"Input: {input_string}")
    if "auth_code=" in input_string:
        try:
            parsed = urllib.parse.urlparse(input_string)
            print(f"Parsed Query: {parsed.query}")
            params = urllib.parse.parse_qs(parsed.query)
            if 'auth_code' in params:
                code = params['auth_code'][0]
                print(f"Extracted Code: {code}")
                return code
        except Exception as e:
            print(f"Error: {e}")
    return input_string.strip()

# Test Case 1: Full URL with other params
url1 = "https://trade.fyers.in/api-login/redirect-uri/index.html?s=ok&code=200&auth_code=eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.samplecode&state=None"
print("--- Test 1 ---")
extract_auth_code(url1)

# Test Case 2: Just the code
code2 = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.samplecode"
print("\n--- Test 2 ---")
extract_auth_code(code2)

# Test Case 3: URL with fragment (unlikely but possible)
url3 = "https://trade.fyers.in/index.html#auth_code=failed_code"
print("\n--- Test 3 ---")
extract_auth_code(url3)
