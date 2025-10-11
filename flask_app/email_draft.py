# email_draft.py

def generate_churn_email(customer_name: str, discount_percent: int) -> str:
    """
    Generates a professional draft email to a customer who is leaving Chubb Insurance.
    
    Parameters:
        customer_name (str): The customer's full name.
        discount_percent (int): The discount percentage to offer for rejoining.
    
    Returns:
        str: The email draft text.
    """
    
    email_body = f"""
Dear {customer_name},

We noticed that you've recently chosen to discontinue your policy with Chubb, and we’re truly sorry to see you go. 
Your trust and partnership mean a great deal to us, and we’d greatly value any feedback you can share about your experience 
or the reasons behind your decision.

We’re constantly working to improve our products and customer experience, and your insights would be incredibly helpful in guiding us.

As a gesture of appreciation, we’d like to offer you an exclusive **{discount_percent}% discount** should you decide to return 
to Chubb within the next 30 days. We’d be delighted to have you back and ensure that your insurance needs are met with 
even greater care and value.

If you’d like to discuss your policy options or this offer further, please don’t hesitate to reach out to our customer 
service team at **support@chubb.com** or call us at **1-800-CHUBB-CARE**.

Thank you again for being a valued part of the Chubb family.

Warm regards,  
**The Chubb Customer Experience Team**  
Chubb Insurance  
https://www.chubb.com
    """
    return email_body.strip()


# Example usage (for testing):
if __name__ == "__main__":
    print(generate_churn_email("Alex Johnson", 15))
