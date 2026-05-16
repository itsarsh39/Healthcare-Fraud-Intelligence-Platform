import pandas as pd
import numpy as np
from faker import Faker
import random
import os

# Set seed for reproducibility
np.random.seed(42)
random.seed(42)
fake = Faker()
Faker.seed(42)

NUM_PATIENTS = 5000
NUM_PROVIDERS = 200
NUM_CLAIMS = 20000

# Constants
SPECIALTIES = ['Cardiology', 'Orthopedics', 'General Practice', 'Neurology', 'Pediatrics']
REGIONS = ['North', 'South', 'East', 'West', 'Central']

def generate_patients():
    patients = []
    for i in range(NUM_PATIENTS):
        patients.append({
            'patient_id': f'P_{i:05d}',
            'age': np.random.randint(18, 90),
            'gender': random.choice(['M', 'F']),
            'region': random.choice(REGIONS)
        })
    return pd.DataFrame(patients)

def generate_providers():
    providers = []
    for i in range(NUM_PROVIDERS):
        providers.append({
            'provider_id': f'PR_{i:03d}',
            'specialty': random.choice(SPECIALTIES),
            'region': random.choice(REGIONS),
            'is_fraudulent': np.random.rand() < 0.05 # 5% of providers are fraud rings
        })
    return pd.DataFrame(providers)

def generate_claims(patients, providers):
    claims = []
    
    # Identify fraud ring providers
    fraud_providers = providers[providers['is_fraudulent'] == True]['provider_id'].tolist()
    normal_providers = providers[providers['is_fraudulent'] == False]['provider_id'].tolist()
    
    # Create a colluding patient ring
    fraud_patients = random.sample(patients['patient_id'].tolist(), int(NUM_PATIENTS * 0.02))
    
    for i in range(NUM_CLAIMS):
        claim_id = f'C_{i:06d}'
        
        # Decide if this claim is fraudulent
        # Fraud types: 1=Upcoding, 2=Collusion Ring, 0=Normal
        is_fraud = np.random.rand() < 0.1 # 10% of total claims are fraud
        
        if is_fraud:
            fraud_type = random.choice([1, 2])
            if fraud_type == 1:
                # Upcoding: Normal provider, normal patient, but extremely high cost
                provider = providers[providers['provider_id'] == random.choice(normal_providers)].iloc[0]
                patient = patients[patients['patient_id'] == random.choice(patients['patient_id'].tolist())].iloc[0]
                cost = np.random.normal(5000, 1000) # Much higher cost
                disease_code = fake.numerify(text="D-####")
            else:
                # Collusion Ring: Fraud provider and fraud patient, multiple repeated claims
                provider = providers[providers['provider_id'] == random.choice(fraud_providers)].iloc[0]
                patient = patients[patients['patient_id'] == random.choice(fraud_patients)].iloc[0]
                cost = np.random.normal(1500, 500)
                disease_code = fake.numerify(text="D-####")
        else:
            # Normal claim
            provider = providers[providers['provider_id'] == random.choice(normal_providers)].iloc[0]
            # Preference for patients in the same region as the provider
            if np.random.rand() < 0.8:
                regional_patients = patients[patients['region'] == provider['region']]
                if not regional_patients.empty:
                    patient = regional_patients.sample(1).iloc[0]
                else:
                    patient = patients.sample(1).iloc[0]
            else:
                patient = patients.sample(1).iloc[0]
            
            cost = np.random.lognormal(mean=5, sigma=1) # Log-normal distribution for costs
            disease_code = fake.numerify(text="D-####")
            
        claims.append({
            'claim_id': claim_id,
            'patient_id': patient['patient_id'],
            'provider_id': provider['provider_id'],
            'disease_code': disease_code,
            'claim_amount': round(cost, 2),
            'claim_date': fake.date_between(start_date='-1y', end_date='today'),
            'is_fraud': is_fraud
        })
        
    return pd.DataFrame(claims)

def main():
    print("Generating synthetic healthcare data...")
    patients = generate_patients()
    providers = generate_providers()
    claims = generate_claims(patients, providers)
    
    # Merge datasets to create a unified view for modeling
    df = claims.merge(patients, on='patient_id', how='left')
    df = df.merge(providers, on='provider_id', how='left', suffixes=('_patient', '_provider'))
    
    # Save to disk
    os.makedirs('data', exist_ok=True)
    df.to_csv('data/healthcare_claims.csv', index=False)
    print(f"Data generation complete. Saved {len(df)} claims to data/healthcare_claims.csv")
    print(f"Fraud prevalence: {df['is_fraud'].mean() * 100:.2f}%")

if __name__ == '__main__':
    main()
