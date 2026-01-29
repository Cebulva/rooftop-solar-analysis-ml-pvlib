"""
Dealer Grading Engine
Scores and ranks solar dealers based on:
- Service Quality (45%)
- Price Competitiveness (35%)
- Delivery Time (20%)
"""

from dataclasses import dataclass
from typing import List, Optional
from src.dealer_finder import Dealer


# Grading weights
WEIGHTS = {
    "quality": 0.45,
    "price": 0.35,
    "delivery": 0.20
}

# Regional average price per kWp in EUR (Germany 2024)
REGIONAL_PRICE_AVG = 1400

# Grade thresholds
GRADE_THRESHOLDS = [
    (95, "A+"),
    (90, "A"),
    (85, "A-"),
    (80, "B+"),
    (75, "B"),
    (70, "B-"),
    (65, "C+"),
    (60, "C"),
    (55, "C-"),
    (50, "D"),
    (0, "F"),
]


@dataclass
class GradedDealer:
    """A dealer with calculated grades and scores."""
    dealer: Dealer
    quality_score: float
    price_score: float
    delivery_score: float
    total_score: float
    grade: str

    @property
    def name(self) -> str:
        return self.dealer.name

    @property
    def distance_km(self) -> float:
        return self.dealer.distance_km

    @property
    def address(self) -> str:
        return self.dealer.address

    @property
    def phone(self) -> Optional[str]:
        return self.dealer.phone

    @property
    def email(self) -> Optional[str]:
        return self.dealer.email

    @property
    def website(self) -> Optional[str]:
        return self.dealer.website

    @property
    def rating(self) -> Optional[float]:
        return self.dealer.rating

    @property
    def review_count(self) -> int:
        return self.dealer.review_count


def calculate_quality_score(dealer: Dealer) -> float:
    """
    Calculate service quality score (0-100).

    Components:
    - Rating score (60%): Based on review rating
    - Verification bonus (25%): For verified/partner dealers
    - Review confidence (15%): More reviews = more reliable
    """
    score = 50.0  # Base score for unknown quality

    # Rating component (0-60 points)
    if dealer.rating is not None:
        # Normalize rating (assume 5-point scale)
        rating_normalized = (dealer.rating / 5.0) * 100
        rating_score = rating_normalized * 0.6
        score = rating_score
    else:
        # No rating - give neutral score
        score = 50 * 0.6

    # Verification bonus (0-25 points)
    if dealer.is_partner:
        score += 25
    elif dealer.is_verified:
        score += 15

    # Review confidence bonus (0-15 points)
    if dealer.review_count > 0:
        # Logarithmic scale: more reviews = higher confidence
        if dealer.review_count >= 100:
            score += 15
        elif dealer.review_count >= 50:
            score += 12
        elif dealer.review_count >= 20:
            score += 8
        elif dealer.review_count >= 5:
            score += 5
        else:
            score += 2

    return min(100, max(0, score))


def calculate_price_score(dealer: Dealer, system_kwp: float = 5.0) -> float:
    """
    Calculate price competitiveness score (0-100).

    Methodology:
    - Compare to regional average (1400 EUR/kWp)
    - Below average = higher score
    - Unknown price = neutral score
    """
    if dealer.price_per_kwp is None:
        # Unknown price - give neutral score
        return 50.0

    price = dealer.price_per_kwp

    # Calculate deviation from average
    deviation_percent = ((price - REGIONAL_PRICE_AVG) / REGIONAL_PRICE_AVG) * 100

    # Score calculation:
    # At average: 70 points
    # 10% below average: 85 points
    # 20% below average: 100 points
    # 10% above average: 55 points
    # 20% above average: 40 points
    # 30%+ above average: 25 points

    if deviation_percent <= -20:
        score = 100
    elif deviation_percent <= -10:
        score = 85 + ((-10 - deviation_percent) / 10) * 15
    elif deviation_percent <= 0:
        score = 70 + ((-deviation_percent) / 10) * 15
    elif deviation_percent <= 10:
        score = 70 - (deviation_percent / 10) * 15
    elif deviation_percent <= 20:
        score = 55 - ((deviation_percent - 10) / 10) * 15
    elif deviation_percent <= 30:
        score = 40 - ((deviation_percent - 20) / 10) * 15
    else:
        score = 25

    return min(100, max(0, score))


def calculate_delivery_score(dealer: Dealer) -> float:
    """
    Calculate delivery time score (0-100).

    Scoring based on average delivery days:
    - Under 21 days (3 weeks): 100
    - 21-28 days (4 weeks): 90
    - 28-42 days (6 weeks): 75
    - 42-56 days (8 weeks): 60
    - 56-84 days (12 weeks): 45
    - Over 84 days: 30
    - Unknown: 50 (neutral)
    """
    if dealer.avg_delivery_days is None:
        return 50.0

    days = dealer.avg_delivery_days

    if days <= 21:
        return 100
    elif days <= 28:
        return 90 + ((28 - days) / 7) * 10
    elif days <= 42:
        return 75 + ((42 - days) / 14) * 15
    elif days <= 56:
        return 60 + ((56 - days) / 14) * 15
    elif days <= 84:
        return 45 + ((84 - days) / 28) * 15
    else:
        return 30


def score_to_grade(score: float) -> str:
    """Convert a numeric score to a letter grade."""
    for threshold, grade in GRADE_THRESHOLDS:
        if score >= threshold:
            return grade
    return "F"


def grade_dealer(dealer: Dealer, system_kwp: float = 5.0) -> GradedDealer:
    """
    Grade a single dealer based on all criteria.

    Args:
        dealer: The dealer to grade
        system_kwp: System size for price calculation context

    Returns:
        GradedDealer with all scores and final grade
    """
    quality_score = calculate_quality_score(dealer)
    price_score = calculate_price_score(dealer, system_kwp)
    delivery_score = calculate_delivery_score(dealer)

    # Calculate weighted total
    total_score = (
        quality_score * WEIGHTS["quality"] +
        price_score * WEIGHTS["price"] +
        delivery_score * WEIGHTS["delivery"]
    )

    grade = score_to_grade(total_score)

    return GradedDealer(
        dealer=dealer,
        quality_score=round(quality_score, 1),
        price_score=round(price_score, 1),
        delivery_score=round(delivery_score, 1),
        total_score=round(total_score, 1),
        grade=grade
    )


def grade_dealers(
    dealers: List[Dealer],
    system_kwp: float = 5.0,
    sort_by: str = "score"
) -> List[GradedDealer]:
    """
    Grade multiple dealers and return sorted list.

    Args:
        dealers: List of dealers to grade
        system_kwp: System size for price context
        sort_by: Sort criteria - "score", "distance", or "price"

    Returns:
        List of GradedDealer objects, sorted as specified
    """
    graded = [grade_dealer(d, system_kwp) for d in dealers]

    if sort_by == "score":
        graded.sort(key=lambda x: x.total_score, reverse=True)
    elif sort_by == "distance":
        graded.sort(key=lambda x: x.distance_km)
    elif sort_by == "price":
        graded.sort(key=lambda x: x.price_score, reverse=True)

    return graded


def get_top_dealers(
    dealers: List[Dealer],
    system_kwp: float = 5.0,
    limit: int = 3
) -> List[GradedDealer]:
    """
    Get the top N dealers by combined score.

    Args:
        dealers: List of dealers to evaluate
        system_kwp: System size for context
        limit: Number of top dealers to return

    Returns:
        Top dealers sorted by total score
    """
    graded = grade_dealers(dealers, system_kwp, sort_by="score")
    return graded[:limit]


def get_grade_color(grade: str) -> str:
    """Get a color code for displaying the grade."""
    colors = {
        "A+": "#00C853",  # Green
        "A": "#00E676",
        "A-": "#69F0AE",
        "B+": "#FFEB3B",  # Yellow
        "B": "#FFC107",
        "B-": "#FFB300",
        "C+": "#FF9800",  # Orange
        "C": "#FF7043",
        "C-": "#FF5722",
        "D": "#F44336",   # Red
        "F": "#D32F2F",
    }
    return colors.get(grade, "#9E9E9E")


if __name__ == "__main__":
    # Test grading with sample dealers
    from src.dealer_finder import find_nearby_dealers, add_sample_dealers, init_database

    init_database()
    add_sample_dealers()

    dealers = find_nearby_dealers(52.52, 13.405, radius_km=500)
    print(f"Found {len(dealers)} dealers")

    graded = get_top_dealers(dealers, system_kwp=5.0, limit=3)

    print("\nTop 3 Dealers:")
    for i, gd in enumerate(graded, 1):
        print(f"\n{i}. {gd.name} [{gd.grade}]")
        print(f"   Distance: {gd.distance_km:.1f} km")
        print(f"   Quality:  {gd.quality_score}/100")
        print(f"   Price:    {gd.price_score}/100")
        print(f"   Delivery: {gd.delivery_score}/100")
        print(f"   Total:    {gd.total_score}/100")
