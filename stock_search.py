import streamlit as st
import json
import pandas as pd
# import agents.plan_agent as plan_agent
import agents.trade_analysis_agent as plan_agent

# Load stock data
def load_stock_data():
    try:
        with open('XNAS_stock.json', 'r', encoding='utf-8') as f:
            nas_stocks = json.load(f)
        with open('XHKG_stock.json', 'r', encoding='utf-8') as f:
            hk_stocks = json.load(f)
        with open('XSHG_stock.json', 'r', encoding='utf-8') as f:
            sh_stocks = json.load(f)
        with open('XSHE_stock.json', 'r', encoding='utf-8') as f:
            she_stocks = json.load(f)
        with open('XNYS_stock.json', 'r', encoding='utf-8') as f:
            nys_stocks = json.load(f)

        # Convert to DataFrames
        nas_df = pd.DataFrame(nas_stocks.get('data', []))
        hk_df = pd.DataFrame(hk_stocks.get('data', []))
        sh_df = pd.DataFrame(sh_stocks.get('data', []))
        she_df = pd.DataFrame(she_stocks.get('data', []))
        nys_df = pd.DataFrame(nys_stocks.get('data', []))
        
        # Combine all stocks
        all_stocks = pd.concat([nas_df, hk_df, sh_df, she_df, nys_df], ignore_index=True)
        return all_stocks
    except Exception as e:
        st.error(f"Error loading stock data: {str(e)}")
        return pd.DataFrame()

def analyze_stock(name, ticker, exchange_code):
    st.header(f"Analyzing {name} ({ticker})")
    with st.spinner('Analyzing stock data...'):
        analysis = plan_agent.run_agent(f"分析股票 {name}（交易所代码（exchange_code：{exchange_code}）：代码：{ticker}）的技术指标和走势")
        # for chunk in analysis:
        st.write_stream(analysis)
        # st.write(analysis)

def main():
    # 搜索区域
    search_container = st.container()
    # 分析结果区域
    analysis_container = st.container()
    
    with search_container:
        st.title("Stock Search And Analysis")
        st.write("Search for stocks in NASDAQ, NYS, Hong Kong, Shengzheng and Shanghai Exchanges")
        
        # 搜索框占据较小的宽度
        col1, col2 = st.columns([4, 4])
        with col1:
            search_query = st.text_input("Enter stock code or name to search:", "")
        
        if search_query:
            # Load data
            df = load_stock_data()
            if not df.empty:
                # 搜索name
                filtered_df = df[df['name'].str.contains(search_query, case=False) | df['ticker'].str.contains(search_query, case=False)]
                # Display results
                if not filtered_df.empty:
                    st.markdown("---")  # 添加分隔线                    
                    # 使用更紧凑的布局显示股票列表
                    for _, row in filtered_df.iterrows():
                        cols = st.columns([3, 2, 2, 2])
                        with cols[0]:
                            st.write(row['name'])
                        with cols[1]:
                            st.write(row['exchange_code'])
                        with cols[2]:
                            st.write(row['ticker'])
                        with cols[3]:
                            if st.button("Analyze", key=f"analyze_{row['ticker']}"):
                                with analysis_container:
                                    st.markdown("---")  # 添加分隔线
                                    analyze_stock(row['name'], row['ticker'], row['exchange_code'])
                else:
                    st.info("No matching stocks found.")

if __name__ == "__main__":
    main()
