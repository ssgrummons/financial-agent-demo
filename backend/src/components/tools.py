from abc import ABC, abstractmethod
from langchain_core.tools import BaseTool, tool
from langchain_experimental.tools import PythonREPLTool
from typing import List
import yfinance as yf
from .fraud.baseline import get_user_baseline
from .fraud.analyzer import StatisticalFraudAnalyzer
import json


class CustomToolSetBase(ABC):
    """
    Generic class for custom toolsets
    """
    @abstractmethod
    def load_tools(self) -> List[BaseTool]:
        """
        Abstract method for loading tools
        """
        pass

class FinanceTools(CustomToolSetBase):
    """
    Class containing the tools required for Finance
    """
    def __init__(self):
        """
        Initialize the FinanceTools
        """
        pass
    
    @tool
    def get_stock_data(symbol: str, period: str = "1mo", info_type: str = "all") -> str:
        """
        Fetch stock data using Yahoo Finance.
        
        Args:
            symbol: Stock ticker symbol (e.g., 'AAPL', 'GOOGL', 'TSLA')
            period: Time period for historical data. Options: 1d, 5d, 1mo, 3mo, 6mo, 1y, 2y, 5y, 10y, ytd, max
            info_type: Type of information to return. Options: 'price', 'info', 'history', 'all'
            
        Returns:
            String containing requested stock information
        """
        try:
            ticker = yf.Ticker(symbol.upper())
            result = []
            
            if info_type in ["price", "all"]:
                # Get current price info
                hist_1d = ticker.history(period="1d")
                if not hist_1d.empty:
                    current_price = hist_1d['Close'].iloc[-1]
                    result.append(f"Current Price ({symbol}): ${current_price:.2f}")
            
            if info_type in ["info", "all"]:
                # Get basic company information
                info = ticker.info
                company_name = info.get('longName', 'N/A')
                market_cap = info.get('marketCap', 'N/A')
                pe_ratio = info.get('trailingPE', 'N/A')
                
                result.append(f"Company: {company_name}")
                if market_cap != 'N/A':
                    result.append(f"Market Cap: ${market_cap:,}")
                if pe_ratio != 'N/A':
                    result.append(f"P/E Ratio: {pe_ratio:.2f}")
            
            if info_type in ["history", "all"]:
                # Get historical data
                hist = ticker.history(period=period)
                if not hist.empty:
                    result.append(f"\nHistorical Data ({period}):")
                    result.append(f"Period Start: {hist.index[0].strftime('%Y-%m-%d')}")
                    result.append(f"Period End: {hist.index[-1].strftime('%Y-%m-%d')}")
                    result.append(f"Highest Price: ${hist['High'].max():.2f}")
                    result.append(f"Lowest Price: ${hist['Low'].min():.2f}")
                    result.append(f"Average Volume: {hist['Volume'].mean():,.0f}")
                    
                    # Calculate basic metrics
                    returns = hist['Close'].pct_change().dropna()
                    volatility = returns.std() * (252**0.5)  # Annualized volatility
                    result.append(f"Annualized Volatility: {volatility:.2%}")
            
            return "\n".join(result)
            
        except Exception as e:
            return f"Error fetching data for {symbol}: {str(e)}"
    
    @tool 
    def compare_stocks(symbols: str, period: str = "3mo") -> str:
        """
        Compare performance of multiple stocks.
        
        Args:
            symbols: Comma-separated stock symbols (e.g., 'AAPL,GOOGL,MSFT')
            period: Time period for comparison (1mo, 3mo, 6mo, 1y, 2y)
            
        Returns:
            String with comparative analysis of the stocks
        """
        try:
            symbol_list = [s.strip().upper() for s in symbols.split(',')]
            results = []
            
            for symbol in symbol_list:
                ticker = yf.Ticker(symbol)
                hist = ticker.history(period=period)
                
                if not hist.empty:
                    start_price = hist['Close'].iloc[0]
                    end_price = hist['Close'].iloc[-1]
                    total_return = ((end_price - start_price) / start_price) * 100
                    
                    results.append({
                        'symbol': symbol,
                        'start_price': start_price,
                        'end_price': end_price,
                        'total_return': total_return
                    })
            
            # Sort by performance
            results.sort(key=lambda x: x['total_return'], reverse=True)
            
            output = [f"Stock Performance Comparison ({period}):"]
            for i, stock in enumerate(results, 1):
                output.append(
                    f"{i}. {stock['symbol']}: {stock['total_return']:.2f}% "
                    f"(${stock['start_price']:.2f} → ${stock['end_price']:.2f})"
                )
            
            return "\n".join(output)
            
        except Exception as e:
            return f"Error comparing stocks: {str(e)}"
        
    @tool
    def detect_fraud_statistical(transaction_description: str, user_id: str = "default") -> str:
        """
        Analyze a transaction for fraud using statistical anomaly detection.
        
        Compares transaction patterns against user's historical spending baseline
        to identify suspicious activity using statistical measures.
        
        Args:
            transaction_description: Natural language description of the transaction
                                (e.g., "Transfer of $5,000 to unknown account at midnight")
            user_id: User identifier for personalized baseline (defaults to "default")
        
        Returns:
            JSON string containing risk analysis with score, flags, and recommendation
        """
        try:
            # Initialize analyzer and get user baseline
            analyzer = StatisticalFraudAnalyzer()
            user_baseline = get_user_baseline(user_id)
            
            # Parse the transaction description
            parsed_transaction = analyzer.parse_transaction_description(transaction_description)
            
            # Perform statistical analysis
            analysis_result = analyzer.analyze_transaction(parsed_transaction, user_baseline)
            
            # Format the response for the AI agent
            response = {
                "transaction": transaction_description,
                "risk_assessment": {
                    "risk_score": analysis_result["risk_score"],
                    "risk_level": analysis_result["risk_level"], 
                    "recommendation": analysis_result["recommendation"]
                },
                "anomaly_detection": {
                    "flags": analysis_result["anomaly_flags"],
                    "statistical_measures": analysis_result["statistical_measures"]
                },
                "parsed_details": {
                    "amount": parsed_transaction["amount"],
                    "time": f"{parsed_transaction['hour']}:00 ({parsed_transaction['time_description']})",
                    "recipient": parsed_transaction["recipient"],
                    "transaction_type": parsed_transaction["transaction_type"]
                },
                "user_baseline_summary": {
                    "typical_amount_range": f"${user_baseline.amount_mean:.0f} ± ${user_baseline.amount_std:.0f}",
                    "max_typical_amount": f"${user_baseline.max_typical_amount}",
                    "common_hours": f"{min(user_baseline.typical_hours)}-{max(user_baseline.typical_hours)}"
                }
            }
            
            return json.dumps(response, indent=2)
            
        except Exception as e:
            return json.dumps({
                "error": f"Failed to analyze transaction: {str(e)}",
                "transaction": transaction_description,
                "risk_assessment": {
                    "risk_score": 0,
                    "risk_level": "ERROR",
                    "recommendation": "Unable to process - manual review required"
                }
            }, indent=2)

        
    def load_tools(self) -> List[BaseTool]:
        """
        Load and return finance-related tools
        """
        description = """
        A Python shell for executing Python commands and calculations.

        AVAILABLE MODULES: Only use these modules - others will cause import errors:
        - Standard library: os, sys, json, csv, datetime, time, math, random, re, collections, itertools, functools, pathlib, urllib, http, sqlite3, logging
        - Data science: pandas, numpy, matplotlib (if available)
        - Other: requests (if available)

        INPUT FORMAT: Valid Python code. Use print() to display results.

        RESTRICTIONS: 
        - Do NOT import or attempt to use modules not listed above
        - Do NOT create fictional modules like 'gagent' 
        - Stick to the available modules only

        Example usage:
        - import pandas as pd
        - import numpy as np  
        - import math
        - result = math.sqrt(16); print(result)
        """
        self.repltool = PythonREPLTool(description=description)
        return [
            self.get_stock_data,
            self.compare_stocks,
            PythonREPLTool(),
            self.detect_fraud_statistical
        ]