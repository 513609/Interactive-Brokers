import pandas as pd
import requests

def get_sp500_tickers():
    """
    Scrapes the list of S&P 500 tickers from Wikipedia in a more robust way.
    """
    print("Fetching S&P 500 ticker list from Wikipedia...")
    url = 'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies'
    
    # Add a User-Agent header to mimic a web browser
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
    }

    try:
        # Use requests to get the page content with headers
        response = requests.get(url, headers=headers)
        response.raise_for_status()  # This will raise an HTTPError for bad responses (4xx or 5xx)

        # Use pandas to read the HTML table from the response text
        tables = pd.read_html(response.text)
        sp500_df = tables[0]
        tickers = sp500_df['Symbol'].tolist()
        print(f"Found {len(tickers)} tickers.")

        # Handle symbol differences (e.g., BRK.B on Wiki vs. BRK B on IBKR)
        if 'BRK.B' in tickers:
            tickers[tickers.index('BRK.B')] = 'BRK B'
        if 'BF.B' in tickers:
            tickers[tickers.index('BF.B')] = 'BF B'
            
        return tickers

    except requests.exceptions.HTTPError as err:
        print(f"\n--- HTTP Error ---")
        print(f"Failed to fetch the Wikipedia page. Status code: {err.response.status_code}")
        print("Please check your internet connection or try again later.")
        return []  # Return an empty list to prevent crashing the next script
    except Exception as e:
        print(f"\nAn unexpected error occurred: {e}")
        return []

if __name__ == '__main__':
    tickers = get_sp500_tickers()
    if tickers:
        print("Successfully fetched tickers. First 20:")
        print(tickers[:20])