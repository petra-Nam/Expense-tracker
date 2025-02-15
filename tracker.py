
import os
import re
from datetime import datetime
import pandas as pd
import matplotlib.pyplot as plt
from dateutil.relativedelta import relativedelta
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
import numpy as np

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
def create_profile(email):
    print("\nCreating a new profile:")
    name = input("Enter your name: ").strip()
    while True:
        try:
            age = int(input("Enter your age: "))
            income = float(input("Enter your monthly income (€): "))
            budget = float(input("Enter your total monthly budget (€): "))
            break
        except ValueError:
            print("Invalid input. Please enter a valid number.")

    while True:
        email = input("Enter your email: ").strip()
        if validate_email(email):
            break
        else:
            print("Invalid email format. Please enter a valid email address.")

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

    print(f"\nProfile created successfully for {name}.")

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
        print("\nNo expenses recorded yet.")
    else:
        print("\nYour Expenses:")
        print(df.to_string(index=False))

# Calculate total expenses for a category
def get_category_total(df, category):
    return df[df['Category'] == category]['Amount'].sum()

# Add expense
def add_expense(expenses_file, df, budget, expenses_by_month):
    print("\nAdding a new expense:")
    while True:
        date = input("Enter the date (YYYY-MM-DD) or press Enter for today: ").strip()
        if not date:
            date = datetime.now().strftime('%Y-%m-%d')
            break
        try:
            datetime.strptime(date, '%Y-%m-%d')
            break
        except ValueError:
            print("Invalid date format. Please enter the date in YYYY-MM-DD format.")

    category = input(f"Enter the category ({', '.join(CATEGORIES)}): ").capitalize()
    if category not in CATEGORIES:
        print("Invalid category. Please choose from the predefined categories.")
        return df, expenses_by_month
    try:
        amount = float(input("Enter the amount (€): "))
    except ValueError:
        print("Invalid amount. Please enter a valid number.")
        return df, expenses_by_month
    description = input("Enter a description (optional): ")

    new_expense = {'Date': date, 'Category': category, 'Amount': amount, 'Description': description}
    df = pd.concat([df, pd.DataFrame([new_expense])], ignore_index=True)
    df.to_csv(expenses_file, index=False)
    print("\nExpense added successfully!")

    # Update the expenses by month for the remaining budget calculation
    month = date[:7]  # Get the YYYY-MM format
    if month not in expenses_by_month:
        expenses_by_month[month] = 0
    expenses_by_month[month] += amount

    # Calculate remaining budget
    total_spent = expenses_by_month.get(month, 0)
    remaining_budget = budget - total_spent
    print(f"Remaining Budget for {month}: €{remaining_budget:.2f}")

    # Inform the user of the remaining budget
    print(f"Total spent this month: €{total_spent:.2f}")
    print(f"Remaining budget for this month: €{remaining_budget:.2f}")

    return df, expenses_by_month

# Edit/Delete expense
def edit_delete_expense(expenses_file, df):
    if df.empty:
        print("\nNo expenses to edit or delete.")
        return df
    print("\nYour Expenses:")
    print(df)
    try:
        index = int(input("Enter the index of the expense to edit/delete: "))
        if index < 0 or index >= len(df):
            raise IndexError
    except (ValueError, IndexError):
        print("Invalid index. Please try again.")
        return df

    action = input("Type 'edit' to modify or 'delete' to remove this expense: ").strip().lower()
    if action == 'edit':
        print(f"Editing expense at index {index}. Leave fields blank to keep existing values.")
        date = input(f"Enter new date (current: {df.at[index, 'Date']}): ") or df.at[index, 'Date']
        category = input(f"Enter new category (current: {df.at[index, 'Category']}): ") or df.at[index, 'Category']
        amount = input(f"Enter new amount (current: {df.at[index, 'Amount']}): ")
        amount = float(amount) if amount else df.at[index, 'Amount']
        description = input(f"Enter new description (current: {df.at[index, 'Description']}): ") or df.at[index, 'Description']
        df.loc[index] = [date, category, amount, description]
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
    monthly_expenses = df[(pd.to_datetime(df['Date']).dt.year == year) &
                          (pd.to_datetime(df['Date']).dt.month == month)]['Amount'].sum()
    return monthly_expenses

# Category totals
def get_category_totals(df, year, month):
    return df[(pd.to_datetime(df['Date']).dt.year == year) &
              (pd.to_datetime(df['Date']).dt.month == month)].groupby('Category')['Amount'].sum()

# Plot category expenses
def plot_expenses_by_category(df, year, month, budget):
    monthly_expenses = df[(pd.to_datetime(df['Date']).dt.year == year) &
                          (pd.to_datetime(df['Date']).dt.month == month)]
    if monthly_expenses.empty:
        print("\nNo expenses for the selected month.")
        return
    category_totals = monthly_expenses.groupby('Category')['Amount'].sum()
    category_totals.plot(kind='bar', color='dodgerblue', edgecolor='black', figsize=(10, 6))
    plt.axhline(y=budget, color='r', linestyle='--', label='Budget')
    plt.title(f"Expenses by Category - {datetime(year, month, 1).strftime('%B %Y')}")
    plt.xlabel("Category")
    plt.ylabel("Total Amount (€)")
    plt.xticks(rotation=45, ha='right')
    plt.legend()
    plt.tight_layout()
    plt.show()

# Prepare budget data for trend
def prepare_budget_data_for_trend():
    """Generates budget data for the last 6 months based on profile budgets."""
    profiles_df = load_profiles()
    last_6_months_data = []

    # Generate last 6 months budget data
    for i in range(6):
        month = (datetime.now() - relativedelta(months=i)).strftime('%Y-%m')
        for _, profile in profiles_df.iterrows():
            last_6_months_data.append({'YearMonth': month, 'Budget': profile['Budget']})

    return pd.DataFrame(last_6_months_data)

def process_expense_data(expenses_df):
    """Processes expense data for the desired timeframe (example: 2018-2019) and aggregates it by Year-Month."""
    expenses_df['Date'] = pd.to_datetime(expenses_df['Date'])
    expenses_df = expenses_df[expenses_df['Date'].dt.year.isin([2018, 2019])]
    expenses_df['YearMonth'] = expenses_df['Date'].dt.to_period('M').astype(str)

    # Aggregate expenses
    return expenses_df.groupby('YearMonth', as_index=False)['Amount'].sum().rename(columns={'Amount': 'Expense'})

# Forecast expenses
def forecast_expenses(expense_data):
    """Applies linear regression to predict future expenses."""
    expense_data['MonthIndex'] = np.arange(len(expense_data))

    # Train Linear Regression Model
    X = expense_data[['MonthIndex']]
    y = expense_data['Expense']
    model = LinearRegression()
    model.fit(X, y)

    # Predict for the next 6 months
    future_months = 6
    last_month_index = expense_data['MonthIndex'].max()
    future_indices = np.arange(last_month_index + 1, last_month_index + future_months + 1).reshape(-1, 1)
    future_forecast = model.predict(future_indices)

    # Generate future dates
    last_date = pd.to_datetime(expense_data['YearMonth'].max())
    future_dates = [last_date + relativedelta(months=i) for i in range(1, future_months + 1)]

    return pd.DataFrame({'YearMonth': [d.strftime('%Y-%m') for d in future_dates], 'Expense': future_forecast})

def plot_budget_trend(expense_data, forecast_data):
    """Plots actual and predicted expenses."""
    plt.figure(figsize=(10, 6))
    plt.plot(expense_data['YearMonth'], expense_data['Expense'], marker='o', linestyle='-', color='blue', label='Actual Expense')
    plt.plot(forecast_data['YearMonth'], forecast_data['Expense'], marker='o', linestyle='--', color='red', label='Predicted Expense')

    plt.title('Expense Trend & Forecast')
    plt.xlabel('Year-Month')
    plt.ylabel('Total Expense (€)')
    plt.xticks(rotation=45)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

# === Execution ===
# Load example expense data (replace with actual CSV loading)
expense_data = pd.read_csv('expenses_ced@gmail.com.csv')

# Assuming 'Date' is a column in your dataset and you want to ensure it's in datetime format
expense_data['Date'] = pd.to_datetime(expense_data['Date'])

# If you need expenses from a specific period, you can filter like this:
filtered_expenses = expense_data[(expense_data['Date'] >= '2019-03-01') & (expense_data['Date'] < '2019-09-30')]

# Process Data
expense_data_processed = process_expense_data(expense_data)

# Forecast Expenses
forecast_data = forecast_expenses(expense_data_processed)

class User:
    def __init__(self, email, total_budget):
        self.email = email
        self.total_budget = total_budget
        self.expenses_by_month = {}  # Store expenses by month (e.g., '2025-01': 500)

    def get_month_from_date(self, date):
        # Extract the year and month from the date (YYYY-MM format)
        return date.strftime('%Y-%m')

    def add_expense(self, expense_amount, expense_date):
        # Get the month for the expense
        month = self.get_month_from_date(expense_date)

        # If no expenses are recorded for the month, initialize it
        if month not in self.expenses_by_month:
            self.expenses_by_month[month] = 0

        # Add the expense to the current month's total
        self.expenses_by_month[month] += expense_amount
        self.show_remaining_budget(month)

    def show_remaining_budget(self, month):
        # Calculate the remaining budget for the month
        total_spent = self.expenses_by_month.get(month, 0)
        remaining_budget = self.total_budget - total_spent
        print(f"Remaining Budget for {month}: €{remaining_budget:.2f}")

# Select month and year
def select_month_year():
    while True:
        try:
            year = int(input("Enter the year (e.g., 2023): "))
            month = int(input("Enter the month (1-12): "))
            if 1 <= month <= 12:
                return year, month
            else:
                print("Invalid month. Please enter a value between 1 and 12.")
        except ValueError:
            print("Invalid input. Please enter numeric values for year and month.")

# Main menu
def main_menu(email):
    expenses_file, df = handle_expenses_file(email)
    profiles_df = load_profiles()
    user_profile = profiles_df[profiles_df['Email'] == email].iloc[0]
    budget = user_profile['Budget']
    expenses_by_month = {}

    while True:
        print("\nExpense Tracker Menu:")
        print("1. View Expenses")
        print("2. Add Expense")
        print("3. Edit/Delete Expense")
        print("4. View Monthly Summary")
        print("5. Plot Expenses by Category")
        print("6. View Budget Trend")
        print("7. Select Month and Year")
        print("8. Exit")
        choice = input("Enter your choice: ").strip()
        if choice == '1':
            view_expenses(df)
        elif choice == '2':
            df, expenses_by_month = add_expense(expenses_file, df, budget, expenses_by_month)
        elif choice == '3':
            df = edit_delete_expense(expenses_file, df)
        elif choice == '4':
            year, month = select_month_year()
            total_expenses = get_monthly_expenses(df, year, month)
            category_totals = get_category_totals(df, year, month)
            print(f"\nTotal expenses for {datetime(year, month, 1).strftime('%B %Y')}: €{total_expenses:.2f}")
            print("Category breakdown:")
            for category, amount in category_totals.items():
                print(f"{category}: €{amount:.2f}")
        elif choice == '5':
            year, month = select_month_year()
            plot_expenses_by_category(df, year, month, budget)
        elif choice == '6':
            # Ensure the correct budget data is used
            expense_data_processed = process_expense_data(expense_data)
            plot_budget_trend(expense_data_processed, forecast_data)
        elif choice == '7':
            year, month = select_month_year()
            print(f"Selected Month and Year: {datetime(year, month, 1).strftime('%B %Y')}")
        elif choice == '8':
            break
        else:
            print("Invalid choice. Please try again.")

        # Ask the user if they want to continue
        continue_choice = input("\nDo you want to continue? (yes/no): ").strip().lower()
        if continue_choice != 'yes':
            break

if __name__ == "__main__":
    initialize_profiles_file()
    email = input("Enter your email: ").strip()
    if user_exists(email):
        print("\nWelcome back!")
        main_menu(email)
    else:
        print("\nNo profile found for this email.")
        create_profile(email)
