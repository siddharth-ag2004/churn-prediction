# [Chubb PS2 - Retainly] - Local Setup Guide

## Overview

This document provides step-by-step instructions to set up and run this project on your local machine for development and testing purposes.

---

## Prerequisites

Before you begin, ensure you have the following software installed on your system.

* **Python**: Version 3.13 or higher  
    ```bash
    python --version 
    # Or on some systems:
    python3 --version
    ```
* **Git**  
    ```bash
    git --version
    ```
* **pip**: Python's package installer

---

## Setup Instructions

### 1. Clone the Repository
```bash
git clone <YOUR_REPOSITORY_URL>
cd <PROJECT_FOLDER_NAME>
```

### 2. Create a Virtual Environment
python -m venv venv
source venv/bin/activate 
#### On Windows use `venv\Scripts\activate`

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Set Up Environment Variables 

**GEMINI_API_KEY**="YOUR_GEMINI_API_KEY"
**MEDIASTACK_API_KEY**="YOUR_MEDIASTACK_API_KEY" <br>
in **.env** file

### 5. Run the Application
```bash
python app.py
```

## Features:
⁠ Project Features

- AI-Powered Churn Prediction: Utilizes a trained Logistic Regression model to accurately predict the probability of a customer churning based on their demographic and policy data.

- Explainable AI (XAI) Insights: Integrates SHAP (SHapley Additive exPlanations) to generate a clear, intuitive waterfall chart. This explains why a prediction was made by showing the positive and negative impact of each customer feature.

- Interactive "What-If" Analysis: Includes a dynamic Discount Impact Analysis chart. For high-risk customers, it simulates the effect of various premium discounts, showing how each reduction lowers the churn probability.

- Prescriptive Retention Strategy: Automatically identifies and highlights the optimal discount percentage required to bring a high-risk customer's churn probability below the 50% threshold, providing a clear, actionable retention step.

- Automated Communication: Features a "Draft Email" button that generates a personalized retention email with the suggested discount, ready to be sent to the at-risk customer.

- Comprehensive Dashboard: Presents a high-level overview of churn analytics, including breakdowns by tenure, premium, income, and demographics, as well as a geospatial heatmap of churn hotspots.

- External Event Correlation: An "Insights" page connects real-world news events (fetched via a news API and analyzed by Gemini) to spikes in the historical churn rate, providing context for market-wide trends.

- **AI Negotiation Agent (New)**: A fully autonomous, text-based negotiation AI agent.
    - **Autonomous Tool Use**: The agent can autonomously execute a "tool call" to apply a discount to the customer's policy in real-time.
    - **Robust Negotiation Logic**: Adheres to strict business rules (e.g., never offer more than the target, require explicit confirmation, single-use discounts).

## Video Link:
https://drive.google.com/file/d/1EyoNU0Z9q8Gltf6HUcFt5sdvV8bmM2TI/view?usp=sharing

Agentic AI demo:
https://drive.google.com/file/d/1rjUnluVjEeMSqGZZ5ayj2Jmygsu-91Sd/view?usp=sharing

## Dataset Link:
https://drive.google.com/drive/folders/1tuefd734bJxHMcDXZneeJClwciLiFh9f?usp=sharing

### Note:
The dataset should be present in dataset/archive/...csv in root directory of project.
