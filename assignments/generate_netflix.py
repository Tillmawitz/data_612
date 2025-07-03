# Generated using Claude.ai

import pandas as pd
import numpy as np
import os
from pathlib import Path
import time
from multiprocessing import Pool, cpu_count

def load_netflix_file(file_path, chunk_size=100000):
    """
    Load a single Netflix data file efficiently.
    
    Args:
        file_path (str): Path to the Netflix data file
        chunk_size (int): Number of rows to read at once
    
    Returns:
        pd.DataFrame: DataFrame with properly parsed Netflix data
    """
    print(f"Loading {file_path}...")
    start_time = time.time()
    
    # Netflix files have mixed format:
    # - Movie ID lines: "1234:"
    # - Rating lines: "customer_id,rating,date"
    # We need to read as text first, then parse manually
    
    all_rows = []
    current_movie_id = None
    
    with open(file_path, 'r') as f:
        for line_num, line in enumerate(f):
            line = line.strip()
            
            if not line:  # Skip empty lines
                continue
                
            if line.endswith(':'):
                # This is a movie ID line
                current_movie_id = int(line.rstrip(':'))
            else:
                # This is a rating line: customer_id,rating,date
                if current_movie_id is None:
                    print(f"Warning: Found rating line before movie ID at line {line_num}")
                    continue
                    
                try:
                    parts = line.split(',')
                    if len(parts) == 3:
                        customer_id = int(parts[0])
                        rating = int(parts[1])
                        date = parts[2]
                        
                        all_rows.append({
                            'movie_id': current_movie_id,
                            'user_id': customer_id,
                            'rating': rating,
                            'date': date
                        })
                except (ValueError, IndexError) as e:
                    print(f"Warning: Could not parse line {line_num}: '{line}' - {e}")
                    continue
            
            # Progress indicator
            if line_num % 1000000 == 0 and line_num > 0:
                print(f"  Processed {line_num:,} lines, {len(all_rows):,} ratings so far...")
    
    # Create DataFrame
    df = pd.DataFrame(all_rows)
    
    if len(df) > 0:
        # Convert date column
        df['date'] = pd.to_datetime(df['date'])
    
    end_time = time.time()
    print(f"Loaded {len(df):,} ratings from {file_path} in {end_time - start_time:.1f} seconds")
    
    return df

def load_netflix_data_kaggle(data_dir, files_to_load=None, sample_fraction=None):
    """
    Load Netflix Prize dataset from Kaggle format (4 files).
    
    Args:
        data_dir (str): Path to directory containing combined_data_*.txt files
        files_to_load (list): Specific files to load (e.g., [1, 2] for first two files)
        sample_fraction (float): Fraction of ratings to sample after loading
    
    Returns:
        pd.DataFrame: Combined dataset with columns [movie_id, user_id, rating, date]
    """
    print("Loading Netflix Prize dataset (Kaggle format)...")
    start_time = time.time()
    
    data_path = Path(data_dir)
    
    # Define the 4 Netflix data files
    netflix_files = [
        "combined_data_1.txt",
        "combined_data_2.txt", 
        "combined_data_3.txt",
        "combined_data_4.txt"
    ]
    
    # Filter files if specified
    if files_to_load:
        netflix_files = [f"combined_data_{i}.txt" for i in files_to_load]
    
    print(f"Loading files: {netflix_files}")
    
    all_data = []
    
    for file_name in netflix_files:
        file_path = data_path / file_name
        
        if not file_path.exists():
            print(f"Warning: {file_path} not found, skipping...")
            continue
            
        # Load the file (now returns properly formatted DataFrame)
        df = load_netflix_file(file_path)
        
        if len(df) > 0:
            all_data.append(df)
    
    if not all_data:
        raise ValueError("No data was loaded from any files")
    
    # Combine all data
    print("Combining data from all files...")
    combined_df = pd.concat(all_data, ignore_index=True)
    
    # Sample if requested
    if sample_fraction and sample_fraction < 1.0:
        original_size = len(combined_df)
        combined_df = combined_df.sample(frac=sample_fraction, random_state=42)
        print(f"Sampled {len(combined_df):,} rows ({sample_fraction*100:.1f}%) from {original_size:,}")
    
    end_time = time.time()
    print(f"Total loading time: {end_time - start_time:.1f} seconds")
    print(f"Final dataset: {len(combined_df):,} ratings")
    
    return combined_df

def process_netflix_format(df):
    """
    Process the Netflix format where movie IDs are in rows ending with ':'.
    
    Args:
        df (pd.DataFrame): Raw data with customer/rating/date columns
    
    Returns:
        pd.DataFrame: Processed data with movie_id, user_id, rating, date columns
    """
    print("Processing Netflix format...")
    print(f"Input DataFrame shape: {df.shape}")
    print(f"Input DataFrame columns: {list(df.columns)}")
    print(f"First few rows:\n{df.head()}")
    
    # Detect the actual column names
    cols = list(df.columns)
    
    # The Netflix format typically has no headers, so columns might be numbered
    if len(cols) >= 1:
        cust_col = cols[0]  # Customer ID or movie ID column
    else:
        raise ValueError("DataFrame has no columns")
    
    # For debugging, let's see what the data looks like
    print(f"Sample values from {cust_col} column:")
    print(df[cust_col].head(10).tolist())
    
    # Convert to string to check for movie ID pattern (ending with ':')
    df_copy = df.copy()
    df_copy[cust_col] = df_copy[cust_col].astype(str)
    
    # Find rows that contain movie IDs (end with ':')
    movie_id_mask = df_copy[cust_col].str.endswith(':')
    
    print(f"Found {movie_id_mask.sum()} movie ID rows out of {len(df)} total rows")
    
    if movie_id_mask.sum() == 0:
        raise ValueError("No movie ID rows found (rows ending with ':')")
    
    # Extract movie IDs
    movie_rows = df_copy[movie_id_mask].copy()
    movie_rows['movie_id'] = movie_rows[cust_col].str.rstrip(':').astype(int)
    
    # Get rating rows (not movie ID rows)
    rating_rows = df_copy[~movie_id_mask].copy()
    
    print(f"Movie rows: {len(movie_rows)}, Rating rows: {len(rating_rows)}")
    
    # Add movie_id to all rows by forward-filling
    all_rows = df_copy.copy()
    all_rows.loc[movie_id_mask, 'movie_id'] = all_rows.loc[movie_id_mask, cust_col].str.rstrip(':').astype(int)
    all_rows['movie_id'] = all_rows['movie_id'].fillna(method='ffill')
    
    # Keep only rating rows and clean up
    final_data = all_rows[~movie_id_mask].copy()
    final_data = final_data.dropna(subset=['movie_id'])
    
    print(f"Final data before cleanup: {len(final_data)} rows")
    
    # Rename columns based on what we have
    if len(cols) >= 3:
        # We have 3 columns: customer_id, rating, date
        rating_col = cols[1]
        date_col = cols[2]
        
        final_data = final_data.rename(columns={
            cust_col: 'user_id',
            rating_col: 'rating',
            date_col: 'date'
        })
    elif len(cols) == 1:
        # Only one column, need to parse it
        # This might happen if the CSV parsing didn't work correctly
        print("Only one column detected, attempting to parse manually...")
        
        # Split the single column by commas
        split_data = final_data[cust_col].str.split(',', expand=True)
        
        if len(split_data.columns) >= 3:
            final_data['user_id'] = split_data[0].astype(int)
            final_data['rating'] = split_data[1].astype(int)
            final_data['date'] = split_data[2]
            final_data = final_data.drop(columns=[cust_col])
        else:
            raise ValueError(f"Cannot parse single column data: {split_data.head()}")
    else:
        raise ValueError(f"Unexpected number of columns: {len(cols)}")
    
    # Convert types
    final_data['movie_id'] = final_data['movie_id'].astype(int)
    final_data['user_id'] = final_data['user_id'].astype(int)
    final_data['rating'] = final_data['rating'].astype(int)
    
    # Convert date
    final_data['date'] = pd.to_datetime(final_data['date'])
    
    # Reorder columns
    final_data = final_data[['movie_id', 'user_id', 'rating', 'date']]
    
    print(f"Final processed data: {len(final_data)} rows")
    print(f"Sample of final data:\n{final_data.head()}")
    
    return final_data.reset_index(drop=True)

def create_balanced_subset(df, target_size=100000, min_user_ratings=10, min_movie_ratings=10):
    """
    Create a balanced subset maintaining user and movie diversity.
    
    Args:
        df (pd.DataFrame): Full Netflix dataset
        target_size (int): Target number of ratings in subset
        min_user_ratings (int): Minimum ratings per user to include
        min_movie_ratings (int): Minimum ratings per movie to include
    
    Returns:
        pd.DataFrame: Subset of the dataset
    """
    print(f"\nCreating balanced subset of ~{target_size:,} ratings...")
    
    # Start with users and movies that have sufficient ratings
    user_counts = df['user_id'].value_counts()
    movie_counts = df['movie_id'].value_counts()
    
    active_users = user_counts[user_counts >= min_user_ratings].index
    popular_movies = movie_counts[movie_counts >= min_movie_ratings].index
    
    print(f"Active users (≥{min_user_ratings} ratings): {len(active_users):,}")
    print(f"Popular movies (≥{min_movie_ratings} ratings): {len(popular_movies):,}")
    
    # Filter to active users and popular movies
    filtered_df = df[
        (df['user_id'].isin(active_users)) & 
        (df['movie_id'].isin(popular_movies))
    ].copy()
    
    print(f"Filtered dataset size: {len(filtered_df):,} ratings")
    
    if len(filtered_df) <= target_size:
        print("Filtered dataset is already smaller than target size")
        return filtered_df
    
    # Strategy: Sample users, then sample their ratings
    np.random.seed(42)  # For reproducibility
    
    # Calculate how many users we need to approximately reach target
    avg_ratings_per_user = len(filtered_df) / len(active_users)
    target_users = int(target_size / avg_ratings_per_user * 1.2)  # 20% buffer
    
    # Sample users
    sampled_users = np.random.choice(
        active_users, 
        size=min(target_users, len(active_users)), 
        replace=False
    )
    
    # Get all ratings for sampled users
    user_subset = filtered_df[filtered_df['user_id'].isin(sampled_users)].copy()
    
    # If still too large, randomly sample ratings
    if len(user_subset) > target_size:
        user_subset = user_subset.sample(n=target_size, random_state=42)
    
    # Sort by user_id and movie_id for consistency
    user_subset = user_subset.sort_values(['user_id', 'movie_id']).reset_index(drop=True)
    
    return user_subset

def analyze_subset(df, subset_df):
    """Print analysis of the created subset."""
    print(f"\n{'='*50}")
    print("SUBSET ANALYSIS")
    print(f"{'='*50}")
    
    print(f"Original dataset: {len(df):,} ratings")
    print(f"Subset size: {len(subset_df):,} ratings ({len(subset_df)/len(df)*100:.2f}%)")
    print()
    
    print(f"Users in subset: {subset_df['user_id'].nunique():,}")
    print(f"Movies in subset: {subset_df['movie_id'].nunique():,}")
    print()
    
    print("Rating distribution:")
    rating_dist = subset_df['rating'].value_counts().sort_index()
    for rating, count in rating_dist.items():
        print(f"  {rating} stars: {count:,} ({count/len(subset_df)*100:.1f}%)")
    print()
    
    print(f"Average ratings per user: {len(subset_df)/subset_df['user_id'].nunique():.1f}")
    print(f"Average ratings per movie: {len(subset_df)/subset_df['movie_id'].nunique():.1f}")
    print()
    
    print("Date range:")
    print(f"  From: {subset_df['date'].min()}")
    print(f"  To: {subset_df['date'].max()}")

def save_subset(subset_df, output_dir="netflix_subset"):
    """Save the subset in multiple formats."""
    os.makedirs(output_dir, exist_ok=True)
    
    # Save as CSV
    csv_path = f"{output_dir}/netflix_subset_{len(subset_df)}.csv"
    subset_df.to_csv(csv_path, index=False)
    print(f"Saved CSV: {csv_path}")
    
    # Save as pickle for faster loading
    pickle_path = f"{output_dir}/netflix_subset_{len(subset_df)}.pkl"
    subset_df.to_pickle(pickle_path)
    print(f"Saved pickle: {pickle_path}")
    
    # Save in recommender-friendly format (user_id, movie_id, rating)
    recommender_df = subset_df[['user_id', 'movie_id', 'rating']].copy()
    recommender_path = f"{output_dir}/netflix_subset_recommender_{len(subset_df)}.csv"
    recommender_df.to_csv(recommender_path, index=False)
    print(f"Saved recommender format: {recommender_path}")
    
    # Save user and movie mappings for later use (convert numpy types to native Python)
    user_mapping = {int(old_id): new_id for new_id, old_id in enumerate(sorted(subset_df['user_id'].unique()))}
    movie_mapping = {int(old_id): new_id for new_id, old_id in enumerate(sorted(subset_df['movie_id'].unique()))}
    
    import json
    with open(f"{output_dir}/user_mapping.json", 'w') as f:
        json.dump(user_mapping, f)
    with open(f"{output_dir}/movie_mapping.json", 'w') as f:
        json.dump(movie_mapping, f)
    
    print(f"Saved ID mappings to {output_dir}/")
    
    # Also save a summary file
    summary = {
        "total_ratings": int(len(subset_df)),
        "unique_users": int(subset_df['user_id'].nunique()),
        "unique_movies": int(subset_df['movie_id'].nunique()),
        "rating_distribution": {int(k): int(v) for k, v in subset_df['rating'].value_counts().sort_index().items()},
        "date_range": {
            "min_date": subset_df['date'].min().strftime('%Y-%m-%d'),
            "max_date": subset_df['date'].max().strftime('%Y-%m-%d')
        },
        "avg_ratings_per_user": float(len(subset_df) / subset_df['user_id'].nunique()),
        "avg_ratings_per_movie": float(len(subset_df) / subset_df['movie_id'].nunique())
    }
    
    with open(f"{output_dir}/summary.json", 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"Saved summary: {output_dir}/summary.json")

# Main execution
if __name__ == "__main__":
    # Configuration for Kaggle dataset
    NETFLIX_DATA_DIR = "netflix_data"  # Current directory or path to folder with combined_data_*.txt
    TARGET_SIZE = 100000
    OUTPUT_DIR = "netflix_subset"
    
    # Loading options:
    # Option 1: Load all 4 files (full dataset ~100M ratings)
    # Option 2: Load specific files (faster)
    # Option 3: Sample from files (even faster)
    
    try:
        # Choose your loading strategy:
        
        # STRATEGY 1: Load first 2 files only (fastest, ~50M ratings)
        print("Loading first 2 Netflix files for faster processing...")
        full_df = load_netflix_data_kaggle(
            NETFLIX_DATA_DIR, 
            files_to_load=[1, 2]  # Load only combined_data_1.txt and combined_data_2.txt
        )
        
        # STRATEGY 2: Load all files but sample 20% (medium speed, ~20M ratings)
        # full_df = load_netflix_data_kaggle(
        #     NETFLIX_DATA_DIR,
        #     sample_fraction=0.2  # Sample 20% from each file
        # )
        
        # STRATEGY 3: Load all 4 files (slowest, full ~100M ratings)
        # full_df = load_netflix_data_kaggle(NETFLIX_DATA_DIR)
        
        # Create subset
        subset_df = create_balanced_subset(
            full_df, 
            target_size=TARGET_SIZE,
            min_user_ratings=10,
            min_movie_ratings=10
        )
        
        # Analyze the subset
        analyze_subset(full_df, subset_df)
        
        # Save the subset
        save_subset(subset_df, OUTPUT_DIR)
        
        print(f"\n✅ Successfully created Netflix subset with {len(subset_df):,} ratings!")
        print(f"Files saved in '{OUTPUT_DIR}/' directory")
        
    except FileNotFoundError as e:
        print(f"❌ Error: {e}")
        print("\nMake sure you have downloaded the Netflix Prize dataset from Kaggle and")
        print("extracted the combined_data_*.txt files to the specified directory")
    except Exception as e:
        print(f"❌ Unexpected error: {e}")

# Loading strategy recommendations:
"""
KAGGLE DATASET LOADING STRATEGIES:

1. FASTEST (recommended for quick experimentation):
   files_to_load=[1]  # Load only first file (~25M ratings)
   
2. FAST (good balance):
   files_to_load=[1, 2]  # Load first two files (~50M ratings)
   
3. MEDIUM (representative sample):
   sample_fraction=0.1  # Load 10% from each file (~10M ratings)
   
4. COMPLETE (if you need full dataset):
   # No parameters - loads all 4 files (~100M ratings)

The first strategy is usually best for developing recommender systems!
"""

# Example usage for loading the subset later:
"""
# Load the subset for experimentation
import pandas as pd

# Load from CSV
df = pd.read_csv('netflix_subset/netflix_subset_100000.csv')

# Or load from pickle (faster)
df = pd.read_pickle('netflix_subset/netflix_subset_100000.pkl')

# For recommender systems, use the simplified format
recommender_df = pd.read_csv('netflix_subset/netflix_subset_recommender_100000.csv')
"""