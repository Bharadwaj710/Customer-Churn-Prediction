"""
src/dashboard/recommendations.py
--------------------------------
Business Retention Intelligence Engine.
Maps customer profiles, risk tiers, and predictive drivers to actionable,
domain-justified retention interventions.
Distinguishes empirical statistical associations from causal explanations.
"""


def generate_retention_playbook(customer_data: dict, prob: float, risk_tier: str) -> dict:
    """
    Generates tailored retention strategies based on actual customer attributes
    and calculated churn probability.
    
    Returns:
        dict containing:
            - summary: high-level executive assessment
            - actions: list of specific actionable intervention strategies
            - disclaimer: epistemological notice on statistical associations vs causality
    """
    tenure = float(customer_data.get("tenure", 1))
    contract = str(customer_data.get("Contract", "Month-to-month"))
    monthly = float(customer_data.get("MonthlyCharges", 50.0))
    internet = str(customer_data.get("InternetService", "DSL"))
    tech_support = str(customer_data.get("TechSupport", "No"))
    online_sec = str(customer_data.get("OnlineSecurity", "No"))
    payment = str(customer_data.get("PaymentMethod", "Electronic check"))
    
    actions = []
    
    # 1. Contract Commitment Strategy
    if contract == "Month-to-month":
        actions.append({
            "category": "Contract Structure",
            "title": "Annual Contract Migration Incentive",
            "priority": "Critical" if prob >= 0.50 else "High",
            "observation": f"Customer is on a {contract} contract. EDA confirms month-to-month subscribers exhibit a 42.7% churn rate (15x higher than 2-year subscribers).",
            "intervention": "Offer a guaranteed 15% rate lock or a free streaming service upgrade conditional on transitioning to a 1-year or 2-year service agreement.",
            "impact": "Reduces immediate monthly cancellation vulnerability and establishes relationship stability."
        })
    elif contract == "One year" and tenure >= 10:
        actions.append({
            "category": "Contract Structure",
            "title": "Early Renewal Loyalty Extension",
            "priority": "Medium",
            "observation": "Customer is approaching the conclusion of their 1-year agreement where churn hazard temporarily spikes.",
            "intervention": "Present an early renewal incentive package 60 days before contract expiration with grandfathered monthly pricing.",
            "impact": "Locks in retention before open-market shopping behavior initiates."
        })

    # 2. Onboarding & Early Tenure Vulnerability
    if tenure <= 12:
        actions.append({
            "category": "Customer Success",
            "title": "First-Year Concierge Onboarding Touchpoint",
            "priority": "Critical" if prob >= 0.60 else "High",
            "observation": f"Customer tenure is {int(tenure)} months. Historical data reveals that nearly 50% of all customer churn occurs within the first 12 months.",
            "intervention": "Assign an onboarding specialist for a proactive 30-day satisfaction check-in to resolve initial configuration or billing friction.",
            "impact": "Addresses customer dissatisfaction before it solidifies into an early cancellation."
        })

    # 3. High-Value Fiber Optic & Value Friction
    if internet == "Fiber optic" and monthly >= 70.0 and (tech_support != "Yes" or online_sec != "Yes"):
        missing_services = []
        if tech_support != "Yes":
            missing_services.append("Tech Support")
        if online_sec != "Yes":
            missing_services.append("Online Security")
        missing_str = " and ".join(missing_services)
        
        actions.append({
            "category": "Service Value & Support",
            "title": "Proactive Premium Support & Security Bundle",
            "priority": "High",
            "observation": f"Customer pays ${monthly:.2f}/mo for Fiber optic without {missing_str}. High expenditure combined with unsupported technical friction strongly correlates with churn.",
            "intervention": f"Provide a complimentary 6-month trial of {missing_str} paired with a network speed optimization review.",
            "impact": "Increases perceived service value and eliminates unassisted technical frustration."
        })

    # 4. Payment Method Friction
    if payment == "Electronic check":
        actions.append({
            "category": "Billing Operations",
            "title": "Automated Payment Modernization Incentive",
            "priority": "Medium",
            "observation": "Customer pays via Electronic Check (historical cohort churn rate: 45.3%, highest among payment methods).",
            "intervention": "Offer a one-time $10 bill credit or $2/month recurring discount for enrolling in automated bank debit or credit card billing.",
            "impact": "Reduces monthly active payment friction and associated bill-shock churn triggers."
        })

    # 5. Low-Risk Engagement for Loyal Customers
    if prob < 0.35 and len(actions) == 0:
        actions.append({
            "category": "Loyalty & Expansion",
            "title": "VIP Loyalty Appreciation & Account Review",
            "priority": "Standard",
            "observation": f"Customer is well-retained ({prob*100:.1f}% churn risk) with {int(tenure)} months of tenure.",
            "intervention": "Deliver an anniversary loyalty perk (e.g., speed upgrade credit or device protection bonus) to maintain high brand affinity.",
            "impact": "Fortifies brand advocacy and preempts competitive solicitation."
        })

    # Executive Summary Assessment
    if prob >= 0.60:
        exec_summary = (
            f"HIGH RISK ALERT ({prob*100:.1f}% Churn Probability): This account requires immediate, "
            f"proactive retention intervention across contract terms, billing friction, and customer support channels."
        )
    elif prob >= 0.35:
        exec_summary = (
            f"MODERATE RISK ({prob*100:.1f}% Churn Probability): Account shows emerging vulnerability factors. "
            f"Targeted value-add bundling and proactive communication are recommended to prevent escalation."
        )
    else:
        exec_summary = (
            f"STABLE ACCOUNT ({prob*100:.1f}% Churn Probability): Customer displays strong retention characteristics. "
            f"Maintain standard engagement and pursue organic value-expansion opportunities."
        )

    disclaimer = (
        "Methodological Note: Recommended retention actions are operational heuristics derived from "
        "observed empirical associations in historical telecom data. Correlation does not imply direct causation, "
        "and individual subscriber responsiveness should be validated through structured A/B experimentation."
    )

    return {
        "summary": exec_summary,
        "actions": actions,
        "disclaimer": disclaimer,
        "risk_tier": risk_tier,
        "probability": prob,
    }

