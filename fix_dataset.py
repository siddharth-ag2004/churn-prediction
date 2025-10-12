import pandas as pd
import numpy as np

def add_noise(value, noise_level=0.05):
    """
    Adds a small amount of random noise to a feature value, proportional to its magnitude.
    A noise_level of 0.05 means the noise will have a standard deviation of 5% of the value.
    """
    if value == 0:
        return 0
    # Noise is drawn from a normal distribution with mean=0
    noise = np.random.normal(0, value * noise_level)
    # Ensure the noisy value doesn't become negative
    return max(0, value + noise)

def calculate_churn_probability(customer_data, min_max_vals):
    """
    Calculates a noisy churn probability using a weighted linear combination of
    min-max normalized features.

    Args:
        customer_data (pd.Series): A customer's feature values.
        min_max_vals (dict): A dictionary of min and max values for normalization.

    Returns:
        float: A churn probability between 0 and 1.
    """
    # --- Step 1: Add Feature Noise to a copy of the data ---
    noisy_data = {
        'curr_ann_amt': add_noise(customer_data.get('curr_ann_amt', 0)),
        'days_tenure': add_noise(customer_data.get('days_tenure', 0)),
        'age_in_years': add_noise(customer_data.get('age_in_years', 0)),
        'income': add_noise(customer_data.get('income', 0)),
        # Binary features are not made noisy
        'good_credit': customer_data.get('good_credit', 0),
        'home_owner': customer_data.get('home_owner', 0),
        'has_children': customer_data.get('has_children', 0)
    }

    # --- Step 2: Weights (same as before) ---
    weights = {
        'base_score': -1.0, 'curr_ann_amt': 3.5, 'days_tenure': -5.0,
        'age_in_years': -2.0, 'income': -1.5, 'good_credit': -3.0,
        'home_owner': -1.5, 'has_children': 0.05
    }

    # --- Step 3: Normalize the NOW NOISY Features ---
    epsilon = 1e-6
    def normalize(key, value):
        min_val = min_max_vals[key]['min']
        max_val = min_max_vals[key]['max']
        return (value - min_val) / (max_val - min_val + epsilon)

    norm_premium = normalize('curr_ann_amt', noisy_data['curr_ann_amt'])
    norm_tenure = normalize('days_tenure', noisy_data['days_tenure'])
    norm_age = normalize('age_in_years', noisy_data['age_in_years'])
    norm_income = normalize('income', noisy_data['income'])

    # --- Step 4: Calculate the Raw Churn Score (using noisy data) ---
    churn_score = weights['base_score']
    churn_score += weights['curr_ann_amt'] * norm_premium
    churn_score += weights['days_tenure'] * norm_tenure
    churn_score += weights['age_in_years'] * norm_age
    churn_score += weights['income'] * norm_income
    churn_score += weights['good_credit'] * noisy_data['good_credit']
    churn_score += weights['home_owner'] * noisy_data['home_owner']
    churn_score += weights['has_children'] * noisy_data['has_children']

    # --- Step 5: Normalize to a "perfect" probability and then add Label Noise ---
    perfect_probability = 1 / (1 + np.exp(-churn_score))
    
    # Add a small random value from a normal distribution (mean 0, std dev 2%)
    label_noise = np.random.normal(0, 0.02)
    noisy_probability = perfect_probability + label_noise
    
    # --- Step 6: Clamp the final result to ensure it's a valid probability ---
    return np.clip(noisy_probability, 0, 1)

def main():
    ORIGINAL_FILENAME = './dataset/archive/autoinsurance_churn.csv'
    OUTPUT_FILENAME = 'formula_labeled_churn_data_noisy.csv'
    PROBABILITY_THRESHOLD = 0.03 # With noise, a 50% threshold is more appropriate

    print("="*70)
    print("Dataset Re-labeling Pipeline (v4 - Linear Normalized with Noise)")
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

    # --- Step 3: Apply the Final Formula with Noise ---
    print("Applying final churn probability formula with noise to each customer...")
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
    
    print("\nNew Churn Distribution (based on noisy formula):")
    print(df['Churn'].value_counts())
    
    print("\nPreview of the new dataset (Probabilities should have a wide range):")
    print(df[['churn_probability', 'Churn']].head(10))
    print("\nStatistics for the new churn probabilities:")
    print(df['churn_probability'].describe())
    print("="*70)

if __name__ == "__main__":
    main()