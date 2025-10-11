import pandas as pd
import numpy as np

def calculate_churn_probability(customer_data, min_max_vals):
    """
    Calculates churn probability using a weighted linear combination of
    min-max normalized features.

    Args:
        customer_data (pd.Series): A customer's feature values.
        min_max_vals (dict): A dictionary of min and max values for normalization.

    Returns:
        float: A churn probability between 0 and 1.
    """
    # --- Step 1: Re-balanced Weights for Normalized Features ---
    # These weights now have a more direct and interpretable impact.
    weights = {
        'base_score': -1.0,                 # A slightly loyal starting point
        'curr_ann_amt': 3.5,                # High premiums are a strong churn driver
        'days_tenure': -5.0,                # Tenure is the strongest loyalty factor
        'age_in_years': -2.0,               # Age is a significant loyalty factor
        'income': -1.5,                     # Income provides stability
        'good_credit': -3.0,                # Good credit is a very strong loyalty factor
        'home_owner': -1.5,                 # Home ownership indicates stability
        'has_children': 1.0                 # Children add a moderate amount of risk
    }

    # --- Step 2: Min-Max Normalize Each Feature to a 0-1 Scale ---
    # Formula: (value - min) / (max - min)
    # This ensures a stable and predictable scale for every feature.
    
    # Use a small epsilon to prevent division by zero if max == min
    epsilon = 1e-6
    
    def normalize(key, value):
        min_val = min_max_vals[key]['min']
        max_val = min_max_vals[key]['max']
        return (value - min_val) / (max_val - min_val + epsilon)

    norm_premium = normalize('curr_ann_amt', customer_data.get('curr_ann_amt', 0))
    norm_tenure = normalize('days_tenure', customer_data.get('days_tenure', 0))
    norm_age = normalize('age_in_years', customer_data.get('age_in_years', 0))
    norm_income = normalize('income', customer_data.get('income', 0))

    # --- Step 3: Calculate the Raw Churn Score (Weighted Sum) ---
    churn_score = weights['base_score']
    
    churn_score += weights['curr_ann_amt'] * norm_premium
    churn_score += weights['days_tenure'] * norm_tenure
    churn_score += weights['age_in_years'] * norm_age
    churn_score += weights['income'] * norm_income
    
    # Binary features don't need normalization (they are already 0 or 1)
    churn_score += weights['good_credit'] * customer_data.get('good_credit', 0)
    churn_score += weights['home_owner'] * customer_data.get('home_owner', 0)
    churn_score += weights['has_children'] * customer_data.get('has_children', 0)

    # --- Step 4: Normalize the Score to a Probability (0 to 1) ---
    probability = 1 / (1 + np.exp(-churn_score))

    return probability

def main():
    ORIGINAL_FILENAME = './dataset/archive/autoinsurance_churn.csv'
    OUTPUT_FILENAME = 'formula_labeled_churn_data_final.csv'
    PROBABILITY_THRESHOLD = 0.02

    print("="*70)
    print("Dataset Re-labeling Pipeline (v3 - Linear Normalized)")
    print("="*70)

    # --- Step 1: Load and Prepare Data ---
    print(f"Loading original dataset from '{ORIGINAL_FILENAME}'...")
    try:
        df = pd.read_csv(ORIGINAL_FILENAME)
        df['curr_ann_amt'] = df['curr_ann_amt'].abs()
        original_churn_counts = df['Churn'].value_counts()
    except FileNotFoundError:
        print(f"ERROR: File not found at '{ORIGINAL_FILENAME}'.")
        return

    # --- Step 2: Calculate Min-Max Values for Normalization ---
    print("Calculating Min-Max values for normalization...")
    min_max_values = {
        'curr_ann_amt': {'min': df['curr_ann_amt'].min(), 'max': df['curr_ann_amt'].max()},
        'days_tenure': {'min': df['days_tenure'].min(), 'max': df['days_tenure'].max()},
        'age_in_years': {'min': df['age_in_years'].min(), 'max': df['age_in_years'].max()},
        'income': {'min': df['income'].min(), 'max': df['income'].max()}
    }

    # --- Step 3: Apply the Final Formula ---
    print("Applying final churn probability formula to each customer...")
    df['churn_probability'] = df.apply(
        lambda row: calculate_churn_probability(row, min_max_values),
        axis=1
    )

    # --- Step 4: Generate New Labels and Clean Data ---
    print(f"Generating new 'Churn' labels using a {PROBABILITY_THRESHOLD:.0%} threshold...")
    df['Churn'] = (df['churn_probability'] >= PROBABILITY_THRESHOLD).astype(int)
    print("Cleaning 'acct_suspd_date' to match new churn labels...")
    df.loc[df['Churn'] == 0, 'acct_suspd_date'] = pd.NaT

    # --- Step 5: Save the New Dataset ---
    print(f"Saving the re-labeled dataset to '{OUTPUT_FILENAME}'...")
    cols = list(df.columns)
    cols.insert(len(cols), cols.pop(cols.index('churn_probability'))) 
    df = df.loc[:, cols]
    df.to_csv(OUTPUT_FILENAME, index=False)
    print("File saved successfully.")

    # --- Step 6: Display a summary of the changes ---
    print("\n--- Summary of Changes ---")
    print("\nOriginal Churn Distribution:")
    print(original_churn_counts)
    
    print("\nNew Churn Distribution (based on final formula):")
    print(df['Churn'].value_counts())
    
    print("\nPreview of the new dataset (Probabilities should have a wide range):")
    print(df[['churn_probability', 'Churn']].head(10))
    print("\nStatistics for the new churn probabilities:")
    print(df['churn_probability'].describe())
    print("="*70)

if __name__ == "__main__":
    main()