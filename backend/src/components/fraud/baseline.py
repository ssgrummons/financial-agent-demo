import random
from datetime import datetime, time
from typing import Dict, List, Tuple
import statistics

class UserBaseline:
    """
    Synthetic user spending patterns for fraud detection.
    Generates realistic baseline behavior for statistical anomaly detection.
    """
    
    def __init__(self, user_id: str = "default"):
        self.user_id = user_id
        self._generate_baseline()
    
    def _generate_baseline(self):
        """Generate synthetic but realistic user spending patterns"""
        
        # Transaction amounts (normal distribution around typical spending)
        self.typical_amounts = [
            25, 45, 12, 89, 156, 67, 34, 78, 23, 190, 
            45, 67, 123, 89, 34, 56, 78, 45, 123, 67,
            234, 89, 45, 156, 78, 23, 345, 67, 89, 123
        ]
        
        # Common transaction times (business hours + evening)
        self.typical_hours = [
            9, 10, 11, 12, 13, 14, 15, 16, 17,  # Business hours
            18, 19, 20, 21, 22,  # Evening
            10, 11, 14, 18, 19   # Weekend patterns
        ]
        
        # Common recipients/merchants
        self.typical_recipients = [
            "walmart", "target", "amazon", "grocery_store", "gas_station",
            "restaurant", "coffee_shop", "pharmacy", "bank_transfer",
            "utility_company", "netflix", "spotify"
        ]
        
        # Calculate statistics
        self.amount_mean = statistics.mean(self.typical_amounts)
        self.amount_std = statistics.stdev(self.typical_amounts)
        self.max_typical_amount = max(self.typical_amounts)
        
        # Time patterns
        self.common_hour_range = (min(self.typical_hours), max(self.typical_hours))
        
        # Monthly spending
        self.monthly_total = sum(self.typical_amounts) * 4  # Approximate monthly
        
    def get_stats(self) -> Dict:
        """Return user's baseline statistics"""
        return {
            "amount_mean": round(self.amount_mean, 2),
            "amount_std": round(self.amount_std, 2),
            "max_typical": self.max_typical_amount,
            "common_hours": self.typical_hours,
            "typical_recipients": self.typical_recipients,
            "monthly_total": self.monthly_total,
            "user_id": self.user_id
        }
    
    def is_typical_hour(self, hour: int) -> bool:
        """Check if transaction hour is typical for this user"""
        return hour in self.typical_hours
    
    def is_typical_recipient(self, recipient: str) -> bool:
        """Check if recipient is typical for this user"""
        recipient_lower = recipient.lower()
        return any(typical in recipient_lower for typical in self.typical_recipients)
    
    def calculate_amount_zscore(self, amount: float) -> float:
        """Calculate z-score for transaction amount"""
        if self.amount_std == 0:
            return 0
        return abs(amount - self.amount_mean) / self.amount_std


# Singleton pattern for quick access
_default_baseline = None

def get_user_baseline(user_id: str = "default") -> UserBaseline:
    """Get user baseline data (creates default if needed)"""
    global _default_baseline
    
    if user_id == "default":
        if _default_baseline is None:
            _default_baseline = UserBaseline("default")
        return _default_baseline
    else:
        # In a real system, you'd load from database
        # For demo, just create new baseline
        return UserBaseline(user_id)