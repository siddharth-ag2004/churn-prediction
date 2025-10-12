def generate_retention_email(customer_name: str, discount_percent: int) -> str:
    """
    Generates a proactive retention email offering a discount to valued customers.

    Parameters:
        customer_name (str): Customer's full name.
        discount_percent (int): The discount percentage to offer.

    Returns:
        str: Email draft text suitable for mailto links.
    """

    email_body = f"""
Dear {customer_name},

We greatly value you as a customer at Chubb Insurance and want to ensure you continue to enjoy the best coverage possible.

As a token of our appreciation, we’re excited to offer you an exclusive {discount_percent}% discount on your policy. This is our way of saying thank you for trusting us with your insurance needs.

If you’d like to take advantage of this offer or discuss your policy options, please reach out to our customer service team at support@chubb.com or call us at 1-800-CHUBB-CARE. We’d be delighted to assist you.

Thank you for being a valued part of the Chubb family.

Warm regards,
The Chubb Customer Experience Team
Chubb Insurance
https://www.chubb.com
"""
    return email_body.strip()


# Example usage (for testing):
if __name__ == "__main__":
    print(generate_churn_email("Alex Johnson", 15))