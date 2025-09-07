import re
from datetime import datetime
from typing import Dict, List, Tuple, Optional
import json

class StatisticalFraudAnalyzer:
    """
    Analyzes transactions for statistical anomalies against user baseline.
    """
    
    def __init__(self):
        self.risk_thresholds = {
            "low": 30,
            "medium": 60, 
            "high": 85
        }
    
    def parse_transaction_description(self, description: str) -> Dict:
        """
        Parse natural language transaction description into structured data.
        
        Examples:
        - "Transfer of $5,000 to an unknown account at midnight"
        - "Payment of $25 to Starbucks at 2:30 PM"
        - "$1,200 wire transfer to offshore account at 3 AM"
        """
        
        # Extract amount
        amount_match = re.search(r'\$?([\d,]+(?:\.\d{2})?)', description)
        amount = float(amount_match.group(1).replace(',', '')) if amount_match else 0
        
        # Extract time
        time_info = self._extract_time(description)
        
        # Extract recipient/merchant info
        recipient_info = self._extract_recipient(description)
        
        # Extract transaction type
        transaction_type = self._extract_transaction_type(description)
        
        return {
            "amount": amount,
            "hour": time_info["hour"],
            "time_description": time_info["description"],
            "recipient": recipient_info["name"],
            "recipient_type": recipient_info["type"],
            "transaction_type": transaction_type,
            "original_description": description
        }
    
    def _extract_time(self, description: str) -> Dict:
        """Extract time information from description"""
        desc_lower = description.lower()
        
        # Specific times
        time_patterns = [
            (r'(\d{1,2}):(\d{2})\s*(am|pm)', lambda h, m, period: 
             int(h) + (12 if period.lower() == 'pm' and int(h) != 12 else 0) - (12 if period.lower() == 'am' and int(h) == 12 else 0)),
            (r'(\d{1,2})\s*(am|pm)', lambda h, period: 
             int(h) + (12 if period.lower() == 'pm' and int(h) != 12 else 0) - (12 if period.lower() == 'am' and int(h) == 12 else 0))
        ]
        
        for pattern, converter in time_patterns:
            match = re.search(pattern, desc_lower)
            if match:
                if len(match.groups()) == 3:  # hour:minute am/pm
                    hour = converter(match.group(1), match.group(2), match.group(3))
                else:  # hour am/pm
                    hour = converter(match.group(1), match.group(2))
                return {"hour": hour, "description": match.group(0)}
        
        # Named times
        time_keywords = {
            "midnight": 0, "noon": 12, "morning": 9, "afternoon": 15,
            "evening": 19, "night": 22, "late": 23, "early": 6
        }
        
        for keyword, hour in time_keywords.items():
            if keyword in desc_lower:
                return {"hour": hour, "description": keyword}
        
        return {"hour": 12, "description": "unknown"}  # Default to noon
    
    def _extract_recipient(self, description: str) -> Dict:
        """Extract recipient information"""
        desc_lower = description.lower()
        
        # Check for suspicious recipient types
        suspicious_keywords = ["unknown", "offshore", "foreign", "suspicious", "anonymous"]
        known_merchants = ["starbucks", "walmart", "amazon", "target", "bank", "atm"]
        
        recipient_type = "unknown"
        recipient_name = "unknown"
        
        for keyword in suspicious_keywords:
            if keyword in desc_lower:
                recipient_type = "suspicious"
                recipient_name = keyword
                break
        
        for merchant in known_merchants:
            if merchant in desc_lower:
                recipient_type = "known_merchant"
                recipient_name = merchant
                break
        
        # Look for "to [recipient]" pattern
        to_match = re.search(r'to\s+([^at\s]+)', desc_lower)
        if to_match:
            recipient_name = to_match.group(1).strip()
        
        return {"name": recipient_name, "type": recipient_type}
    
    def _extract_transaction_type(self, description: str) -> str:
        """Extract transaction type"""
        desc_lower = description.lower()
        
        type_keywords = {
            "transfer": ["transfer", "wire", "send"],
            "payment": ["payment", "pay", "purchase"],
            "withdrawal": ["withdrawal", "withdraw", "atm"],
            "deposit": ["deposit"]
        }
        
        for trans_type, keywords in type_keywords.items():
            if any(keyword in desc_lower for keyword in keywords):
                return trans_type
        
        return "unknown"
    
    def analyze_transaction(self, parsed_transaction: Dict, user_baseline) -> Dict:
        """
        Perform statistical analysis of transaction against baseline.
        """
        risk_score = 0
        anomaly_flags = []
        statistical_measures = {}
        
        # Amount analysis
        amount_zscore = user_baseline.calculate_amount_zscore(parsed_transaction["amount"])
        statistical_measures["amount_zscore"] = round(amount_zscore, 2)
        
        if amount_zscore > 3:  # 3 standard deviations
            risk_score += 40
            anomaly_flags.append(f"Amount ${parsed_transaction['amount']} is {amount_zscore:.1f} std devs above normal")
        elif amount_zscore > 2:
            risk_score += 25
            anomaly_flags.append(f"Amount ${parsed_transaction['amount']} is unusually high ({amount_zscore:.1f} std devs)")
        
        # Time analysis
        if not user_baseline.is_typical_hour(parsed_transaction["hour"]):
            risk_score += 30
            anomaly_flags.append(f"Transaction time ({parsed_transaction['hour']}:00) is outside normal hours")
        
        # Recipient analysis
        if not user_baseline.is_typical_recipient(parsed_transaction["recipient"]):
            risk_score += 25
            anomaly_flags.append(f"Recipient '{parsed_transaction['recipient']}' is not in typical merchant list")
        
        if parsed_transaction["recipient_type"] == "suspicious":
            risk_score += 35
            anomaly_flags.append(f"Recipient type flagged as suspicious")
        
        # Transaction type risk
        if parsed_transaction["transaction_type"] == "transfer":
            risk_score += 15
            anomaly_flags.append("Wire transfers have elevated risk")
        
        # Determine risk level
        if risk_score >= self.risk_thresholds["high"]:
            risk_level = "HIGH"
        elif risk_score >= self.risk_thresholds["medium"]:
            risk_level = "MEDIUM" 
        elif risk_score >= self.risk_thresholds["low"]:
            risk_level = "LOW"
        else:
            risk_level = "MINIMAL"
        
        return {
            "risk_score": risk_score,
            "risk_level": risk_level,
            "anomaly_flags": anomaly_flags,
            "statistical_measures": statistical_measures,
            "recommendation": self._get_recommendation(risk_level),
            "parsed_transaction": parsed_transaction
        }
    
    def _get_recommendation(self, risk_level: str) -> str:
        """Get recommendation based on risk level"""
        recommendations = {
            "MINIMAL": "APPROVE - Transaction appears normal",
            "LOW": "APPROVE - Minor anomalies detected, monitor account",
            "MEDIUM": "REVIEW - Multiple anomalies detected, manual review recommended", 
            "HIGH": "BLOCK - High risk transaction, requires immediate investigation"
        }
        return recommendations.get(risk_level, "REVIEW - Unknown risk level")