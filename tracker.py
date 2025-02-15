 
import os
import re
from datetime import datetime
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from dateutil.relativedelta import relativedelta
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
import seaborn as sns

profiles_file = 'user_profiles.csv'

# Define categories as a global constant
CATEGORIES = [
    'Groceries', 'Transport', 'Entertainment', 'Shopping',
    'Savings', 'Investment', 'Rent', 'Utilities', 'Other'
]

# Initialize profiles file
def initialize_profiles_file():
    if not os.path.exists(profiles_file):
        df = pd.DataFrame(columns=['Name', 'Email', 'Age', 'Income', 'Budget'])
        df.to_csv(profiles_file, index=False)

# Load profiles
def load_profiles():
    return pd.read_csv(profiles_file)

# Save profiles
def save_profiles(profiles_df):
    profiles_df.to_csv(profiles_file, index=False)

# Check if user exists
def user_exists(email):
    profiles_df = load_profiles()
    return not profiles_df[profiles_df['Email'] == email].empty

# Validate email format
def validate_email(email):
    email_regex = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    return re.match(email_regex, email) is not None

# Create a new profile
def create_profile():
    email = input("Enter your email: ")
    if not validate_email(email):
        print("Invalid email address.")
        return

    name = input("Enter your name: ")
    if not name:
        print("Invalid name.")
        return

    try:
        age = int(input("Enter your age: "))
        income = float(input("Enter your monthly income (€): "))
        budget = float(input("Enter your total monthly budget (€): "))
    except ValueError:
        print("Invalid input. Please enter valid numbers for age, income, and budget.")
        return

    new_profile = pd.DataFrame({
        'Name': [name],
        'Email': [email],
        'Age': [age],
        'Income': [income],
        'Budget': [budget]
    })
    profiles_df = load_profiles()
    profiles_df = pd.concat([profiles_df, new_profile], ignore_index=True)
    save_profiles(profiles_df)

    print(f"Profile created successfully for {name}.")

# Handle expenses file
def handle_expenses_file(email):
    expenses_file = f'expenses_{email}.csv'
    if not os.path.exists(expenses_file):
        df = pd.DataFrame(columns=['Date', 'Category', 'Amount', 'Description'])
        df.to_csv(expenses_file, index=False)
    else:
        df = pd.read_csv(expenses_file)
    return expenses_file, df

# View expenses
def view_expenses(df):
    if df.empty:
        print("No expenses recorded yet.")
    else:
        print(df)

# Calculate total expenses for a category
def get_category_total(df, category):
    return df[df['Category'] == category]['Amount'].sum()

# Add expense
def add_expense(expenses_file, df, budget, expenses_by_month):
    date = input("Enter the date (YYYY-MM-DD) or press Enter for today: ") or datetime.now().strftime('%Y-%m-%d')
    try:
        datetime.strptime(date, '%Y-%m-%d')
    except ValueError:
        print("Invalid date format. Please enter the date in YYYY-MM-DD format.")
        return df, expenses_by_month

    category = input(f"Enter the category ({', '.join(CATEGORIES)}): ").capitalize()
    if category not in CATEGORIES:
        print("Invalid category. Please choose from the predefined categories.")
        return df, expenses_by_month

    try:
        amount = float(input("Enter the amount (€): "))
    except ValueError:
        print("Invalid amount. Please enter a valid number.")
        return df, expenses_by_month

    description = input("Enter a description (optional): ") or ""

    new_expense = {'Date': date, 'Category': category, 'Amount': amount, 'Description': description}
    df = pd.concat([df, pd.DataFrame([new_expense])], ignore_index=True)
    df.to_csv(expenses_file, index=False)
    print("Expense added successfully!")

    # Update the expenses by month for the remaining budget calculation
    month = date[:7]  # Get the YYYY-MM format
    if month not in expenses_by_month:
        expenses_by_month[month] = 0
    expenses_by_month[month] += amount

    # Calculate remaining budget
    total_spent = expenses_by_month.get(month, 0)
    remaining_budget = budget - total_spent
    print(f"Remaining Budget for {month}: €{remaining_budget:.2f}")

    return df, expenses_by_month

# Edit/Delete expense
def edit_delete_expense(expenses_file, df):
    if df.empty:
        print("No expenses to edit or delete.")
        return df

    print(df)
    index = int(input("Enter the index of the expense to edit/delete: "))
    if index < 0 or index >= len(df):
        print("Invalid index. Please try again.")
        return df

    action = input("Type 'edit' to modify or 'delete' to remove this expense: ").strip().lower()
    if action == 'edit':
        date = input(f"Enter new date (current: {df.at[index, 'Date']}): ") or df.at[index, 'Date']
        category = input(f"Enter new category (current: {df.at[index, 'Category']}): ") or df.at[index, 'Category']
        amount = input(f"Enter new amount (current: {df.at[index, 'Amount']}): ") or df.at[index, 'Amount']
        description = input(f"Enter new description (current: {df.at[index, 'Description']}): ") or df.at[index, 'Description']
        df.loc[index] = [date, category, float(amount), description]
        print("Expense updated successfully!")
    elif action == 'delete':
        df = df.drop(index).reset_index(drop=True)
        print("Expense deleted successfully!")
    else:
        print("Invalid action.")

    df.to_csv(expenses_file, index=False)
    return df

# Monthly expenses
def get_monthly_expenses(df, year, month):
    monthly_expenses = df[(pd.to_datetime(df['Date']).dt.year == year) & (pd.to_datetime(df['Date']).dt.month == month)]['Amount'].sum()
    return monthly_expenses

# Category totals
def get_category_totals(df, year, month):
    return df[(pd.to_datetime(df['Date']).dt.year == year) & (pd.to_datetime(df['Date']).dt.month == month)].groupby('Category')['Amount'].sum()

# Plot category expenses
def plot_expenses_by_category(df, year, month, budget):
    monthly_expenses = df[(pd.to_datetime(df['Date']).dt.year == year) & (pd.to_datetime(df['Date']).dt.month == month)]
    if monthly_expenses.empty:
        print("No expenses for the selected month.")
        return
    category_totals = monthly_expenses.groupby('Category')['Amount'].sum()

    fig, ax = plt.subplots(figsize=(10, 6))
    category_totals.plot(kind='bar', color='dodgerblue', edgecolor='black', ax=ax)
    ax.axhline(y=budget, color='r', linestyle='--', label='Budget')
    ax.set_title(f"Expenses by Category - {datetime(year, month, 1).strftime('%B %Y')}")
    ax.set_xlabel("Category")
    ax.set_ylabel("Total Amount (€)")
    plt.xticks(rotation=45, ha='right')
    plt.legend()
    plt.tight_layout()
    plt.show()

# Prepare budget data for trend analysis
def prepare_budget_data_for_trend(data):
    if data.empty:
        return None

    data['Date'] = pd.to_datetime(data['Date'], format='%Y-%m-%d')
    data.set_index('Date', inplace=True)
    data = data.sort_index()
    monthly_expenses = data.resample('MS').sum()  # Resample to start of the month
    monthly_expenses['MonthIndex'] = range(len(monthly_expenses))
    return monthly_expenses

# Train Random Forest model and make predictions
def train_and_predict_model(data, future_months=6):
    monthly_expenses = prepare_budget_data_for_trend(data)

    if monthly_expenses is None or len(monthly_expenses) <= 1:
        print("Insufficient data to train the model.")
        return None, None

    # Filter data for the range 2018 to 2019
    monthly_expenses = monthly_expenses[(monthly_expenses.index >= '2018-01-01') & (monthly_expenses.index <= '2019-09-30')]

    if monthly_expenses.empty:
        print("Insufficient data to train the model within the specified range.")
        return None, None

    X = monthly_expenses[['MonthIndex']]
    y = monthly_expenses['Amount']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # Predict future expenses
    last_index = monthly_expenses['MonthIndex'].max()
    future_indices = np.arange(last_index + 1, last_index + 1 + future_months).reshape(-1, 1)
    future_dates = [monthly_expenses.index.max() + relativedelta(months=i) for i in range(1, future_months + 1)]
    future_expenses = model.predict(future_indices)

    forecast_df = pd.DataFrame({'Date': future_dates, 'Predicted_Expense': future_expenses})
    forecast_df.set_index('Date', inplace=True)

    return monthly_expenses, forecast_df

# Plot budget trend
def plot_budget_trend(monthly_expenses, forecast_df, budget):
    plt.figure(figsize=(10, 6))
    plt.plot(monthly_expenses.index, monthly_expenses['Amount'], label='Historical Data', marker='o')
    plt.plot(forecast_df.index, forecast_df['Predicted_Expense'], label='Predicted Future', linestyle='--', marker='x')
    plt.axhline(y=budget, color='r', linestyle='--', label='Budget')
    plt.title('Expense Forecast')
    plt.xlabel('Date')
    plt.ylabel('Amount')
    plt.legend()
    plt.grid(True)
    plt.show()

# Plot heatmap for spending patterns
def plot_heatmap(df):
    df['Date'] = pd.to_datetime(df['Date'])
    df['Day'] = df['Date'].dt.day
    df['Month'] = df['Date'].dt.month
    df['Year'] = df['Date'].dt.year

    pivot_table = df.pivot_table(values='Amount', index=['Year', 'Month'], columns='Day', aggfunc='sum', fill_value=0)

    plt.figure(figsize=(12, 8))
    sns.heatmap(pivot_table, cmap='YlGnBu', linewidths=.5)
    plt.title('Spending Patterns Heatmap')
    plt.xlabel('Day of the Month')
    plt.ylabel('Month')
    plt.show()

# Show expense management tips
def show_expense_management_tips():
    tips = """
    Expense Management Tips:
    1. Track your expenses regularly.
    2. Create a detailed budget and stick to it.
    3. Prioritize your spending. Focus on needs before wants.
    4. Use cash for discretionary spending to avoid overspending.
    5. Save money by preparing meals at home instead of dining out.
    6. Negotiate better rates for recurring expenses (e.g., insurance, subscriptions).
    7. Automate savings to ensure your financial goals are met.
    8. Review your budget periodically and make adjustments as necessary.
    """
    print(tips)

def main():
    initialize_profiles_file()

    account_exists = input("Do you already have an account? (yes/no): ").strip().lower()

    if account_exists == 'yes':
        email = input("Enter your email: ").strip()
        if not user_exists(email):
            print("No profile found for this email.")
            return
    else:
        create_profile()
        email = input("Enter your email: ").strip()

    expenses_file, df = handle_expenses_file(email)
    profiles_df = load_profiles()
    user_profile = profiles_df[profiles_df['Email'] == email].iloc[0]
    budget = user_profile['Budget']
    expenses_by_month = {}

    while True:
        print("\nMenu:")
        print("1. View Expenses")
        print("2. Add Expense")
        print("3. Edit/Delete Expense")
        print("4. View Monthly Summary")
        print("5. Plot Expenses by Category")
        print("6. View Budget Trend")
        print("7. Plot Heatmap")
        print("8. Expense Management Tips")
        print("9. Exit")
        choice = input("Enter your choice: ").strip()

        if choice == '1':
            view_expenses(df)
        elif choice == '2':
            df, expenses_by_month = add_expense(expenses_file, df, budget, expenses_by_month)
        elif choice == '3':
            df = edit_delete_expense(expenses_file, df)
        elif choice == '4':
            year = int(input("Enter the year (e.g., 2023): "))
            month = int(input("Enter the month (1-12): "))
            if month < 1 or month > 12:
                print("Invalid month. Please try again.")
                continue
            total_expenses = get_monthly_expenses(df, year, month)
            category_totals = get_category_totals(df, year, month)
            print(f"Total expenses for {datetime(year, month, 1).strftime('%B %Y')}: €{total_expenses:.2f}")
            print("Category breakdown:")
            for category, amount in category_totals.items():
                print(f"{category}: €{amount:.2f}")
        elif choice == '5':
            year = int(input("Enter the year (e.g., 2023): "))
            month = int(input("Enter the month (1-12): "))
            if month < 1 or month > 12:
                print("Invalid month. Please try again.")
                continue
            plot_expenses_by_category(df, year, month, budget)
        elif choice == '6':
            monthly_expenses, forecast_df = train_and_predict_model(df)
            if monthly_expenses is not None and forecast_df is not None:
                plot_budget_trend(monthly_expenses, forecast_df, budget)
        elif choice == '7':
            plot_heatmap(df)
        elif choice == '8':
            show_expense_management_tips()
        elif choice == '9':
            print("Exiting...")
            break
        else:
            print("Invalid choice. Please try again.")

        continue_choice = input("Do you want to continue? (yes/no): ").strip().lower()
        if continue_choice != 'yes':
            print("Exiting...")
            break

if __name__ == "__main__":
    main()
