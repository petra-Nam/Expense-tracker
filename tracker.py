
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
def create_profile(email, name, age, income, budget):
    if not validate_email(email):
        print("Invalid email address.")
        return

    if not name:
        print("Invalid name.")
        return

    try:
        age = int(age)
        income = float(income)
        budget = float(budget)
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
def add_expense(expenses_file, df, budget, expense):
    date = expense['date'] or datetime.now().strftime('%Y-%m-%d')
    try:
        datetime.strptime(date, '%Y-%m-%d')
    except ValueError:
        print("Invalid date format. Please enter the date in YYYY-MM-DD format.")
        return df

    category = expense['category'].capitalize()
    if category not in CATEGORIES:
        print("Invalid category. Please choose from the predefined categories.")
        return df

    try:
        amount = float(expense['amount'])
    except ValueError:
        print("Invalid amount. Please enter a valid number.")
        return df

    description = expense['description'] or ""

    new_expense = {'Date': date, 'Category': category, 'Amount': amount, 'Description': description}
    df = pd.concat([df, pd.DataFrame([new_expense])], ignore_index=True)
    df.to_csv(expenses_file, index=False)
    print("Expense added successfully!")

    # Update the expenses by month for the remaining budget calculation
    month = date[:7]  # Get the YYYY-MM format
    if month not in expense['expenses_by_month']:
        expense['expenses_by_month'][month] = 0
    expense['expenses_by_month'][month] += amount

    # Calculate remaining budget
    total_spent = expense['expenses_by_month'].get(month, 0)
    remaining_budget = budget - total_spent
    print(f"Remaining Budget for {month}: €{remaining_budget:.2f}")

    return df

# Edit/Delete expense
def edit_delete_expense(expenses_file, df, index, action, expense_details):
    if df.empty:
        print("No expenses to edit or delete.")
        return df

    if index < 0 or index >= len(df):
        print("Invalid index. Please try again.")
        return df

    if action == 'edit':
        df.loc[index] = expense_details
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

if __name__ == "__main__":
    initialize_profiles_file()

    # Example profile creation
    create_profile("test@example.com", "John Doe", 30, 3000, 2000)

    # Example expense management
    email = "test@example.com"
    expenses_file, df = handle_expenses_file(email)
    profiles_df = load_profiles()
    user_profile = profiles_df[profiles_df['Email'] == email].iloc[0]
    budget = user_profile['Budget']
    expenses_by_month = {}

    # Example add expense
    expense = {'date': '2023-10-01', 'category': 'Groceries', 'amount': 50, 'description': 'Food shopping', 'expenses_by_month': expenses_by_month}
    df = add_expense(expenses_file, df, budget, expense)

    # View expenses
    view_expenses(df)

    # Example edit/delete expense
    df = edit_delete_expense(expenses_file, df, 0, 'edit', ['2023-10-01', 'Groceries', 55, 'Updated food shopping'])

    # Plot expenses by category
    plot_expenses_by_category(df, 2023, 10, budget)

    # Train and predict model
    monthly_expenses, forecast_df = train_and_predict_model(df)
    if monthly_expenses is not None and forecast_df is not None:
        plot_budget_trend(monthly_expenses, forecast_df, budget)

    # Plot heatmap
    plot_heatmap(df)
