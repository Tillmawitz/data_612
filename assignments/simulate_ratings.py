import pandas as pd
import numpy as np
import random

# Set seed for reproducibility
np.random.seed(42)
random.seed(42)

# Define products and people
products = ['Laptop Pro', 'Wireless Headphones', 'Smart Watch', 'Tablet Mini', 'Gaming Mouse']
people = ['Alice', 'Bob', 'Carol', 'David', 'Emma', 'Frank', 'Grace', 'Henry', 'Iris', 'Jack']

# Create base ratings for products (some products are generally better)
product_base_scores = {
    'Laptop Pro': 4.2,        # High-quality, expensive product
    'Wireless Headphones': 3.8, # Good product, popular
    'Smart Watch': 3.5,       # Mixed reviews, some love it
    'Tablet Mini': 3.0,       # Average product
    'Gaming Mouse': 4.0       # Enthusiasts love it, others neutral
}

# Create personality traits for people (some are more generous/harsh raters)
person_bias = {
    'Alice': 0.8,    # Very generous rater
    'Bob': 0.3,      # Slightly generous
    'Carol': -0.2,   # Slightly harsh
    'David': 0.0,    # Neutral
    'Emma': 0.5,     # Generous
    'Frank': -0.5,   # Harsh rater
    'Grace': 0.1,    # Slightly generous
    'Henry': -0.3,   # Slightly harsh
    'Iris': 0.4,     # Generous
    'Jack': -0.1     # Slightly harsh
}

# Generate ratings
ratings_data = []

for person in people:
    for product in products:
        # Base score + person bias + some random noise
        base_score = product_base_scores[product]
        bias = person_bias[person]
        noise = np.random.normal(0, 0.3)  # Small random variation
        
        # Calculate final rating
        rating = base_score + bias + noise
        
        # Ensure rating is between 1 and 5
        rating = max(1, min(5, rating))
        
        # Round to nearest 0.5 (common rating scale)
        rating = round(rating * 2) / 2
        
        ratings_data.append({
            'Person': person,
            'Product': product,
            'Rating': rating
        })

# Create DataFrame
df = pd.DataFrame(ratings_data)

# Display the dataset
print("Product Rating Dataset")
print("=" * 50)
print(df.to_string(index=False))

print("\n\nDataset Analysis")
print("=" * 50)

# Average rating by person (showing rating tendencies)
print("\nAverage Rating by Person:")
person_avg = df.groupby('Person')['Rating'].mean().sort_values(ascending=False)
for person, avg_rating in person_avg.items():
    tendency = "generous" if avg_rating > 3.5 else "harsh" if avg_rating < 3.0 else "neutral"
    print(f"{person}: {avg_rating:.2f} ({tendency})")

# Average rating by product (showing product quality)
print("\nAverage Rating by Product:")
product_avg = df.groupby('Product')['Rating'].mean().sort_values(ascending=False)
for product, avg_rating in product_avg.items():
    quality = "excellent" if avg_rating > 4.0 else "good" if avg_rating > 3.5 else "average" if avg_rating > 3.0 else "poor"
    print(f"{product}: {avg_rating:.2f} ({quality})")

# Create pivot table for easy viewing
print("\nRating Matrix (Person vs Product):")
pivot_table = df.pivot(index='Person', columns='Product', values='Rating')
print(pivot_table)

# Save to CSV
df.to_csv('product_ratings.csv', index=False)
print(f"\nDataset saved as 'product_ratings.csv' with {len(df)} ratings total.")