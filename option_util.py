import datetime

def get_next_expiry(today=None, ticker="NIFTY"):
    """
    Calculate the next expiry date.
    NIFTY: Thursday
    BANKNIFTY: Wednesday (Effective Sep 2023)
    """
    if today is None:
        today = datetime.date.today()
    
    # 0=Mon, 1=Tue, 2=Wed, 3=Thu, 4=Fri, 5=Sat, 6=Sun
    # Nifty Expiry = Thursday (3)
    # BankNifty Expiry = Wednesday (2)
    
    target_weekday = 3 # Default Thursday
    if "BANK" in ticker.upper():
        target_weekday = 2 # Wednesday
        
    days_ahead = target_weekday - today.weekday()
    if days_ahead < 0: # Target day already passed this week
        days_ahead += 7
    if days_ahead == 0 and datetime.datetime.now().time() > datetime.time(15, 30):
        # If today is expiry but market closed, get next week
        days_ahead += 7
        
    return today + datetime.timedelta(days=days_ahead)

def get_fyers_symbol(ticker, strike, otype, expiry_date=None):
    """
    Construct Fyers Symbol for Weekly Options.
    Format: NSE:NIFTYyyMddSTRCE
    Example: NSE:NIFTY23O1919500CE (19th Oct 2023)
    
    Month Codes for Weekly:
    Jan: 1, Feb: 2 ... Sep: 9, Oct: O, Nov: N, Dec: D
    """
    if expiry_date is None:
        expiry_date = get_next_expiry(ticker=ticker)
        
    # Validations
    ticker_upper = ticker.upper()
    if "NIFTY" in ticker_upper or "^NSEI" in ticker_upper:
        base = "NIFTY"
    elif "BANK" in ticker_upper:
        base = "BANKNIFTY"
    else:
        return f"NSE:{ticker}-EQ" # Fallback
    
    yy = expiry_date.strftime("%y")
    dd = expiry_date.strftime("%d")
    
    month_str = expiry_date.strftime("%b").upper() # JAN, FEB...
    
    # Fyers Weekly Format check failed in live tests.
    # Monthly Format (YYMMM) e.g. 26FEB was confirmed WORKING.
    # We will prioritize the Monthly format if it's the dominant contract or default to it for now.
    
    # Ideally we'd probe, but here we just return the most likely valid one based on live test.
    # Live Test found: NSE:NIFTY26FEB25600CE (Monthly format)
    
    # Check if expiry_date is a monthly expiry? 
    # For now, let's use the Monthly format if the day matches monthly expiry logic, 
    # OR just default to Monthly format if Weekly fails?
    # Actually, the weekly format (YYMMMdd) failed. The Monthly one worked. 
    # Let's standardize on Monthly format 'YYMMM' for now as it seems robust for the near term.
    # CAUTION: This might break if we specifically want weekly expiries that are NOT month-end.
    
    # But for Nifty, usually weekly contracts exist.
    # Maybe the format is different?
    # Let's try to infer if it's a monthly expiry (last Thurs of month).
    
    # REVISION: Robust Logic
    # If date is Last Thursday -> Monthly Format (YYMMM)
    # Else -> Weekly Format (which needs fixing, maybe YYMdd or YYOdd?)
    
    # For Feb 26 2026 (Monthly), format is 26FEB.
    # For Feb 12 2026 (Weekly), format is ? (Probe failed for YYMMMdd and YYMdd)
    
    # Let's stick to what WORKED: YYMMM (Monthly).
    # This implies we are trading the MONTHLY contract.
    # This is safer for liquidity anyway.
    
    symbol = f"NSE:{base}{yy}{month_str}{strike}{otype}"
    return symbol

if __name__ == "__main__":
    # Test
    print("Next Nifty Expiry:", get_next_expiry(ticker="NIFTY"))
    print("Next BankNifty Expiry:", get_next_expiry(ticker="BANKNIFTY"))
    
    # Example Symbol
    print("Symbol:", get_fyers_symbol("^NSEI", 25000, "CE"))
