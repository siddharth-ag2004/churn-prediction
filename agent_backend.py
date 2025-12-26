import os
import json
import re
from datetime import datetime
from dotenv import load_dotenv
from groq import Groq
from flask import session

# Load env
load_dotenv()
client = Groq(api_key=os.environ.get("GROQ_API_KEY"))

# --- Global State for Demo ---
DISCOUNT_LOGS = []

# Mock customer data for the demo (Alex Johnson) - Global so it can be modified
MOCK_CUSTOMER_DATA = {
    'curr_ann_amt': 1200.50,
    'days_tenure': 500,
    'age_in_years': 45,
    'income': 65000,
    'has_children': 1,
    'marital_status': 1, # Married (encoded)
    'home_owner': 0,
    'good_credit': 1,
}

def apply_discount_tool(discount_percent: int):
    """
    Applies a discount to the customer's policy.
    """
    print(f"\n[SYSTEM] ⚡️ TOOL CALLED: apply_discount({discount_percent})")
    
    # Calculate new premium
    current_premium = MOCK_CUSTOMER_DATA['curr_ann_amt']
    new_premium = current_premium * (1 - (discount_percent / 100.0))
    
    # Update global state
    MOCK_CUSTOMER_DATA['curr_ann_amt'] = round(new_premium, 2)
    
    # Log the action
    log_entry = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "customer": "Alex Johnson", # Mock customer for demo
        "discount": discount_percent
    }
    DISCOUNT_LOGS.insert(0, log_entry) # Add to top of list
    
    return f"SUCCESS: Applied {discount_percent}% discount. New annual premium is ${MOCK_CUSTOMER_DATA['curr_ann_amt']}."

# Tool Definition for Groq (OpenAI format)
tools = [
    {
        "type": "function",
        "function": {
            "name": "apply_discount_tool",
            "description": "Applies a discount of the specified percentage to the customer's policy. Use this ONLY when the customer explicitly agrees.",
            "parameters": {
                "type": "object",
                "properties": {
                    "discount_percent": {
                        "type": "integer",
                        "description": "The percentage discount to apply (e.g., 15).",
                    }
                },
                "required": ["discount_percent"],
            },
        },
    }
]

def get_system_instruction(target_discount, discount_applied):
    base_instruction = f"""You are a helpful and empathetic customer service agent for Retainly, an auto insurance company.
Your goal is to retain the customer who is thinking about cancelling their policy.

Current Customer: Alex Johnson
Current Premium: ${MOCK_CUSTOMER_DATA['curr_ann_amt']}/year
Tenure: 500 days

**CRITICAL RULES:**
1.  **Tone**: Be professional, understanding, and polite. Acknowledge their concerns.
2.  **Knowledge**: You ALREADY KNOW the customer's current premium is ${MOCK_CUSTOMER_DATA['curr_ann_amt']}. **DO NOT** ask the user for this information.
"""

    if discount_applied:
        system_instruction = base_instruction + f"""
3.  **NO MORE DISCOUNTS**: You have ALREADY applied a discount. The customer's final premium is ${MOCK_CUSTOMER_DATA['curr_ann_amt']}.
    - **DO NOT** offer any further discounts.
    - If the customer asks for more, politely explain that this is the best offer available and you cannot lower it further.
    - Do NOT call the `apply_discount_tool` again.
"""
    else:
        system_instruction = base_instruction + f"""
3.  **Target Discount**: Our predictive model indicates that a **{target_discount}% discount** is optimal to retain this customer.
    - Start by offering **10%**.
    - You are authorized to go up to **{target_discount}%** to save the customer.
    - **ONLY** if the customer is extremely persistent and about to leave, you may go up to a hard cap of **20%**.
    - Try your best to settle at or below {target_discount}%.
    - **CRITICAL**: Use it **ONLY** when the customer explicitly agrees to a specific discount (e.g., says "yes", "deal", "ok", "apply it").
    - **ACTION**: When the customer agrees, you MUST generate a **Tool Call** to `apply_discount_tool`.
    - **Do NOT** write the function name or JSON in your message. Just make the tool call.
    - **Do NOT** call the tool to "check" prices. Use mental math for estimates.
    - **NEGOTIATION LOGIC**: 
        - If the user agrees to a discount (e.g., 10%), **APPLY IT IMMEDIATELY**. Do NOT offer a higher discount (e.g., 12%) after they have already agreed.
        - If you reject a requested discount (e.g., customer asks for 17%), do **NOT** counter with a *higher* discount (e.g., 18%). Counter with a lower one (e.g., 16%) or the same.
"""
    return system_instruction

def execute_discount_logic(discount_percent, session):
    """Helper to execute discount and update session"""
    if session.get('discount_applied', False):
         return "Error: A discount has already been applied. You cannot apply another one.", None
    
    result = apply_discount_tool(discount_percent)
    new_premium = MOCK_CUSTOMER_DATA['curr_ann_amt']
    session['discount_applied'] = True
    return result, new_premium

def chat_with_agent(user_message, history, session):
    target_discount = session.get('target_discount', 15)
    discount_applied = session.get('discount_applied', False)
    
    system_instruction = get_system_instruction(target_discount, discount_applied)
    print(f"DEBUG: Agent System Instruction:\n{system_instruction}")

    try:
        messages = [{"role": "system", "content": system_instruction}]
        for msg in history:
            role = "user" if msg['role'] == "user" else "assistant"
            messages.append({"role": role, "content": msg['text']})
        messages.append({"role": "user", "content": user_message})
        
        # First API call
        response = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=messages,
            tools=tools,
            tool_choice="auto",
            max_tokens=1024
        )
        
        response_message = response.choices[0].message
        tool_calls = response_message.tool_calls
        
        print(f"DEBUG: Raw Response Content: {response_message.content}")
        print(f"DEBUG: Tool Calls: {tool_calls}")
        
        agent_text = ""
        new_premium = None
        
        if tool_calls:
            messages.append(response_message)
            for tool_call in tool_calls:
                function_name = tool_call.function.name
                function_args = json.loads(tool_call.function.arguments)
                
                if function_name == "apply_discount_tool":
                    discount_percent = function_args.get("discount_percent")
                    function_response, premium_update = execute_discount_logic(discount_percent, session)
                    if premium_update:
                        new_premium = premium_update
                    
                    messages.append({
                        "tool_call_id": tool_call.id,
                        "role": "tool",
                        "name": function_name,
                        "content": function_response,
                    })
            
            second_response = client.chat.completions.create(
                model="llama-3.1-8b-instant",
                messages=messages
            )
            agent_text = second_response.choices[0].message.content
            
        else:
            # FALLBACK: Aggressive parser for ANY discount_percent JSON
            match = re.search(r'"discount_percent"\s*:\s*(\d+)', response_message.content)
            
            if match:
                try:
                    discount_percent = int(match.group(1))
                    print(f"DEBUG: Found hallucinated tool call via AGGRESSIVE fallback: {discount_percent}%")
                    
                    _, premium_update = execute_discount_logic(discount_percent, session)
                    if premium_update:
                        new_premium = premium_update

                    # Clean up the response text
                    cleaned_content = re.sub(r'\{.*?"discount_percent".*?\}', '', response_message.content, flags=re.DOTALL)
                    cleaned_content = re.sub(r'<function.*?>.*?</function>', '', cleaned_content, flags=re.DOTALL)
                    cleaned_content = re.sub(r'Action:.*', '', cleaned_content)
                    cleaned_content = re.sub(r'Thought:.*', '', cleaned_content)
                    
                    if not cleaned_content.strip():
                        cleaned_content = f"I have applied the {discount_percent}% discount for you."
                        
                    agent_text = cleaned_content.strip()
                        
                except Exception as e:
                    print(f"Error parsing aggressive tool: {e}")
                    agent_text = response_message.content
            else:
                agent_text = response_message.content

        return agent_text, new_premium

    except Exception as e:
        # ERROR RESCUE: Check for tool_use_failed
        error_str = str(e)
        if "tool_use_failed" in error_str and "failed_generation" in error_str:
            print("DEBUG: Caught tool_use_failed error. Attempting rescue...")
            # Extract the failed generation text
            # It usually looks like: ... 'failed_generation': '...text...' ...
            # We can use regex to find the content of failed_generation
            rescue_match = re.search(r"'failed_generation':\s*'(.*?)'", error_str, re.DOTALL)
            if not rescue_match:
                 rescue_match = re.search(r'"failed_generation":\s*"(.*?)"', error_str, re.DOTALL)
            
            if rescue_match:
                failed_text = rescue_match.group(1)
                # Unescape escaped quotes if necessary
                failed_text = failed_text.replace("\\'", "'").replace('\\"', '"').replace('\\n', '\n')
                
                print(f"DEBUG: Rescued text: {failed_text}")
                
                # Run aggressive parser on the rescued text
                match = re.search(r'"discount_percent"\s*:\s*(\d+)', failed_text)
                if match:
                    discount_percent = int(match.group(1))
                    print(f"DEBUG: Rescued discount: {discount_percent}%")
                    
                    _, premium_update = execute_discount_logic(discount_percent, session)
                    
                    # Return a polite message since we can't continue the chat chain easily
                    return f"I have successfully applied the {discount_percent}% discount for you.", premium_update
        
        print(f"Agent Error: {e}")
        raise e
