import streamlit as st
import json
import pandas as pd

# Load stock data
def load_stock_data():
    try:
        with open('XNAS_stock.json', 'r', encoding='utf-8') as f:
            nas_stocks = json.load(f)
        with open('XHKG_stock.json', 'r', encoding='utf-8') as f:
            hk_stocks = json.load(f)

        # Convert to DataFrames
        nas_df = pd.DataFrame(nas_stocks.get('data', []))
        hk_df = pd.DataFrame(hk_stocks.get('data', []))

        #取data
        
        # Combine all stocks
        all_stocks = pd.concat([nas_df, hk_df], ignore_index=True)
        return all_stocks
    except Exception as e:
        st.error(f"Error loading stock data: {str(e)}")
        return pd.DataFrame()

def main():
    st.title("Stock Search Interface")
    st.write("Search for stocks in NASDAQ and Hong Kong exchanges")
    
    # Load data
    df = load_stock_data()
    filtered_df = pd.DataFrame()
    if not df.empty:
        # Search box
        search_query = st.text_input("Enter stock code or name to search:", "")
        
        if search_query:
            #搜索name
            filtered_df = df[df['name'].str.contains(search_query, case=False) | df['ticker'].str.contains(search_query, case=False)]
            # Display results
            if not filtered_df.empty:
                st.write(f"Found {len(filtered_df)} matching stocks:")
                st.dataframe(filtered_df, use_container_width=True)
            else:
                st.info("No matching stocks found.")

if __name__ == "__main__":
    main()
