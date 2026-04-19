import pandas as pd         # For creating data tables (DataFrames)
import numpy as np          # For math operations (random numbers, etc)
from datetime import datetime, timedelta    # For dates
import random       # For random choices

# Set random seeds so we get same data everytime
np.random.seed(42)
random.seed(42)
print("Random seeds set! Same data every run!")

def generate_fraud_data(n_samples=1000, fraud_rate=0.1):
    """
    Generate synthetic fraud data for training
    :param n_samples: Total transactions (default 1000)
    :param fraud_rate: Percentage fraud (default 0.1 = 10%)
    """

    # Calculate counts
    n_fraud = int(n_samples * fraud_rate)
    n_normal = n_samples - n_fraud

    print(f"\n Generating {n_samples} transactions:")
    print(f"\n Normal {n_normal} ({n_normal/n_samples*100:.1f}%)")
    print(f"\n Fraud: {n_fraud} ({n_fraud/n_samples*100:.1f}%)")

    # Part 1: Generate NORMAL transaction
    print("\n Creating normal transactions ...")

    # Start with amounts
    # Log-normal: Most small (~€8K), few large, realistic for money!
    normal_amounts = np.random.lognormal(mean=9, sigma=1, size=n_normal)

    print(f" Sample amounts: €{normal_amounts[0]:.0f}, €{normal_amounts[1]:.0f}, €{normal_amounts[2]:.0f}")

    # Hours : Business hours only (8am - 7pm)
    normal_hours = np.random.choice(range(8, 20), size=n_normal)

    # Days : Weekdays only (Monday=0 to Friday=4)
    normal_days = np.random.choice(range(0, 5), size=n_normal)

    print(f" Sample hours: {normal_hours[0]}, {normal_hours[1]}, {normal_hours[2]}")
    print(f" Sample days: {normal_days[0]}, {normal_days[1]}, {normal_days[2]}")

    # Location risk: Beta distributed skewed LOW
    # Most normal people use safe location
    normal_location_risk = np.random.beta(2, 5, size=n_normal) * 100

    print(f" Sample location risks: {normal_location_risk[0]:.1f}, {normal_location_risk[1]:.1f}, {normal_location_risk[2]:.1f}")

    #  Customer age: Normal distribution around 45 years
    normal_age = np.random.normal(45, 15, size=n_normal)

    # Transaction velocity: Poisson, average 2 recent transactions
    normal_velocity = np.random.poisson(2, size=n_normal)

    print(f" Sample ages: {normal_age[0]:.0f}, {normal_age[1]:.0f}, {normal_age[2]:.0f}")
    print(f" Sample velocity: {normal_velocity[0]}, {normal_velocity[1]}, {normal_velocity[2]}")

    # Combine all normal features into dictionary
    normal_data = {
        'transaction_amount': normal_amounts,
        'hour_of_day': normal_hours,
        'day_of_week': normal_days,
        'location_risk_score': normal_location_risk,
        'customer_age': normal_age,
        'transaction_velocity': normal_velocity,
        'is_fraud': np.zeros(n_normal) # All 0 (not fraud)
    }

    # Convert to DataFrame
    df_normal = pd.DataFrame(normal_data)
    print(f" Created DataFrame with {len(df_normal)} normal transactions")
    print("\n First 3 normal transactions:")
    print(df_normal.head(3))

    # PART 2: Generate FRAUD transactions
    print("\n🔴 Creating fraud transactions...")
    # Amounts: HIGHER! Median ~€60K
    fraud_amounts = np.random.lognormal(mean=11, sigma=1.5, size=n_fraud)

    # Hours: UNUSUAL hours ! (midnight - 5am, late night)
    fraud_hours = np.random.choice([0, 1, 2, 3, 4, 5, 22, 23], size=n_fraud)

    # Days: ANY day (including weekends)
    fraud_days = np.random.choice(range(0, 7), size=n_fraud)

    # Location risk: HIGH! Beta skewed right
    fraud_location_risk = np.random.beta(5, 2, size=n_fraud) * 100

    # Age: YOUNGER! Average 35
    fraud_age = np.random.normal(35, 12, size=n_fraud)

    # Velocity: MANY transactions! Average 8
    fraud_velocity = np.random.poisson(8, size=n_fraud)

    print(f"   Sample amounts: €{fraud_amounts[0]:.0f}, €{fraud_amounts[1]:.0f}, €{fraud_amounts[2]:.0f}")
    print(f"   Sample hours: {fraud_hours[0]}, {fraud_hours[1]}, {fraud_hours[2]}")
    print(f"   Sample location risks: {fraud_location_risk[0]:.1f}, {fraud_location_risk[1]:.1f}, {fraud_location_risk[2]:.1f}")
    print(f"   Sample velocity: {fraud_velocity[0]}, {fraud_velocity[1]}, {fraud_velocity[2]}")

    # Combine fraud features
    fraud_data = {
        'transaction_amount': fraud_amounts,
        'hour_of_day': fraud_hours,
        'day_of_week': fraud_days,
        'location_risk_score': fraud_location_risk,
        'customer_age': fraud_age,
        'transaction_velocity': fraud_velocity,
        'is_fraud': np.ones(n_fraud)  # All 1 (IS fraud)
    }

    df_fraud = pd.DataFrame(fraud_data)

    print(f"\n✅ Created DataFrame with {len(df_fraud)} fraud transactions")
    print("\n📋 First 3 fraud transactions:")
    print(df_fraud.head(3))

    # PART 3: Combine and shuffle
    print("\n🔀 Combining and shuffling data...")

    # Stack DataFrames vertically
    df = pd.concat([df_normal, df_fraud], ignore_index=True)
    print(f"   Combined: {len(df)} transactions")

    # Shuffle randomly (very important!)
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    print(f"   Shuffled randomly!")

    #PART 4: Clean up outliers
    print("\n Clipping values to realistic group")

    # Clip amount to €100 - €200,000
    df['transaction_amount'] = df['transaction_amount'].clip(100, 200000)

    # Clip age to 18-80 years
    df['customer_age'] = df['customer_age'].clip(18, 80).astype(int)

    # Clip location risk to 0-100
    df['location_risk_score'] = df['location_risk_score'].clip(0, 100)

    # Clip velocity to 0-20 transactions
    df['transaction_velocity'] = df['transaction_velocity'].clip(0, 20).astype(int)

    print("All values within realistic ranges!")

    # Save and display stats
    print("\n Final dataset statistics: ")
    print(f"    Total: {len(df)} transactions")
    print(f"    Normal: {len(df[df['is_fraud']==0])} ({len(df[df['is_fraud']==0])/len(df)*100:.1f}%)")
    print(f"    Fraud: {len(df[df['is_fraud']==1])} ({len(df[df['is_fraud']==1])/len(df)*100:.1f}%)")

    print("\n Feature ranges:")
    print(df.describe())

    # Save to CSV
    output_path = 'data/fraud_training_data.csv'
    df.to_csv(output_path, index=False)

    print(f"\n Saved to: {output_path}")
    print("Data generation complete!")

    return df

# Run the function
if __name__ == "__main__":
    generate_fraud_data()

