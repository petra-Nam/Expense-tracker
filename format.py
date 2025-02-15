import os
import pandas as pd

# Correct file path for the original dataset
original_file_path = "/Users/I750363/Downloads/personal_transactions.csv"

# Check if the file exists
if not os.path.exists(original_file_path):
    print(f"File not found: {original_file_path}")
else:
    # Load the original dataset
    data = pd.read_csv(original_file_path)

    # Step 1: Convert the 'Date' column to YYYY-MM-DD format
    data['Date'] = pd.to_datetime(data['Date'], format='%m/%d/%Y').dt.strftime('%Y-%m-%d')

    # Step 2: Keep only the relevant columns
    filtered_data = data[['Amount', 'Date', 'Category']]

    # Step 3: Adjust categories to the predefined list
    allowed_categories = [
        'Groceries', 'Transport', 'Entertainment', 'Shopping', 
        'Savings', 'Investment', 'Rent', 'Utilities', 'Other'
    ]

    # Map existing categories to the allowed ones or assign "Other"
    def map_category(category):
        if category in ['Gas & Fuel']:
            return 'Transport'
        elif category in ['Television', 'Music', 'Movies', 'DVDs']:
            return 'Entertainment'
        elif category in ['Mortgage & Rent', 'Home Improvement']:
            return 'Rent'
        elif category in ['Credit Card Payment']:
            return 'Shopping'
        elif category in allowed_categories:
            return category
        else:
            return 'Other'

    filtered_data['Category'] = filtered_data['Category'].apply(map_category)

    # Save the cleaned and mapped data to a new CSV file
    new_cleaned_file_path = "/Users/I750363/Desktop/new_cleaned_personal_transactions.csv"
    filtered_data.to_csv(new_cleaned_file_path, index=False)

    # Print the cleaned and mapped data to verify
    print(filtered_data.head())

    # Load the user profiles CSV file
    user_profiles_path = "/Users/I750363/Desktop/Python project/user_profiles.csv"
    user_profiles = pd.read_csv(user_profiles_path)

    # Create a new DataFrame with the necessary columns for the new user "ced"
    new_users = pd.DataFrame({
        'Name': ['ced'],
        'Email': ['ced@gmail.com'],
        'Age': [20],
        'Income': [100000],
        'Budget': [80000]
    })

    # Concatenate the new users to the user profiles DataFrame
    updated_user_profiles = pd.concat([user_profiles, new_users], ignore_index=True)

    # Save the updated user profiles to the CSV file
    updated_user_profiles.to_csv(user_profiles_path, index=False)

    # Print the updated user profiles to verify
    print(updated_user_profiles.head())

    # Load the expenses file for "ced"
    expenses_file = "/Users/I750363/Desktop/Python project/expenses_ced@gmail.com.csv"
    if not os.path.exists(expenses_file):
        expenses_df = pd.DataFrame(columns=['Date', 'Category', 'Amount', 'Description'])
    else:
        expenses_df = pd.read_csv(expenses_file)

    # Add the filtered data to the expenses file
    filtered_data['Description'] = ''  # Add an empty description column
    expenses_df = pd.concat([expenses_df, filtered_data], ignore_index=True)

    # Save the updated expenses file
    expenses_df.to_csv(expenses_file, index=False)

    # Print the updated expenses to verify
    print(expenses_df.head())
