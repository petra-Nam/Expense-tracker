
import os
import re
from datetime import datetime
import pandas as pd
import matplotlib.pyplot as plt
from dateutil.relativedelta import relativedelta
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
import numpy as np
import tkinter as tk
from tkinter import messagebox, simpledialog
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

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
    email = simpledialog.askstring("Create Profile", "Enter your email:")
    if not validate_email(email):
        messagebox.showerror("Invalid Email", "Please enter a valid email address.")
        return

    name = simpledialog.askstring("Create Profile", "Enter your name:")
    if not name:
        messagebox.showerror("Invalid Name", "Please enter a valid name.")
        return

    try:
        age = int(simpledialog.askstring("Create Profile", "Enter your age:"))
        income = float(simpledialog.askstring("Create Profile", "Enter your monthly income (€):"))
        budget = float(simpledialog.askstring("Create Profile", "Enter your total monthly budget (€):"))
    except ValueError:
        messagebox.showerror("Invalid Input", "Please enter valid numbers for age, income, and budget.")
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

    messagebox.showinfo("Profile Created", f"Profile created successfully for {name}.")

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
        messagebox.showinfo("No Expenses", "No expenses recorded yet.")
    else:
        view_window = tk.Toplevel(root)
        view_window.title("Your Expenses")
        text = tk.Text(view_window)
        text.insert(tk.END, df.to_string(index=False))
        text.pack()

# Calculate total expenses for a category
def get_category_total(df, category):
    return df[df['Category'] == category]['Amount'].sum()

# Add expense
def add_expense(expenses_file, df, budget, expenses_by_month):
    date = simpledialog.askstring("Add Expense", "Enter the date (YYYY-MM-DD) or press Enter for today:") or datetime.now().strftime('%Y-%m-%d')
    try:
        datetime.strptime(date, '%Y-%m-%d')
    except ValueError:
        messagebox.showerror("Invalid Date", "Invalid date format. Please enter the date in YYYY-MM-DD format.")
        return df, expenses_by_month

    category = simpledialog.askstring("Add Expense", f"Enter the category ({', '.join(CATEGORIES)}):").capitalize()
    if category not in CATEGORIES:
        messagebox.showerror("Invalid Category", "Invalid category. Please choose from the predefined categories.")
        return df, expenses_by_month

    try:
        amount = float(simpledialog.askstring("Add Expense", "Enter the amount (€):"))
    except ValueError:
        messagebox.showerror("Invalid Amount", "Invalid amount. Please enter a valid number.")
        return df, expenses_by_month

    description = simpledialog.askstring("Add Expense", "Enter a description (optional):") or ""

    new_expense = {'Date': date, 'Category': category, 'Amount': amount, 'Description': description}
    df = pd.concat([df, pd.DataFrame([new_expense])], ignore_index=True)
    df.to_csv(expenses_file, index=False)
    messagebox.showinfo("Expense Added", "Expense added successfully!")

    # Update the expenses by month for the remaining budget calculation
    month = date[:7]  # Get the YYYY-MM format
    if month not in expenses_by_month:
        expenses_by_month[month] = 0
    expenses_by_month[month] += amount

    # Calculate remaining budget
    total_spent = expenses_by_month.get(month, 0)
    remaining_budget = budget - total_spent
    messagebox.showinfo("Budget Update", f"Remaining Budget for {month}: €{remaining_budget:.2f}")

    return df, expenses_by_month

# Edit/Delete expense
def edit_delete_expense(expenses_file, df):
    if df.empty:
        messagebox.showinfo("No Expenses", "No expenses to edit or delete.")
        return df

    edit_view = tk.Toplevel(root)
    edit_view.title("Edit/Delete Expense")
    text = tk.Text(edit_view)
    text.insert(tk.END, df.to_string(index=False))
    text.pack()

    index = simpledialog.askinteger("Edit/Delete Expense", "Enter the index of the expense to edit/delete:")
    if index is None or index < 0 or index >= len(df):
        messagebox.showerror("Invalid Index", "Invalid index. Please try again.")
        return df

    action = simpledialog.askstring("Edit/Delete Expense", "Type 'edit' to modify or 'delete' to remove this expense:").strip().lower()
    if action == 'edit':
        date = simpledialog.askstring("Edit Expense", f"Enter new date (current: {df.at[index, 'Date']}):") or df.at[index, 'Date']
        category = simpledialog.askstring("Edit Expense", f"Enter new category (current: {df.at[index, 'Category']}):") or df.at[index, 'Category']
        amount = simpledialog.askstring("Edit Expense", f"Enter new amount (current: {df.at[index, 'Amount']}):")
        amount = float(amount) if amount else df.at[index, 'Amount']
        description = simpledialog.askstring("Edit Expense", f"Enter new description (current: {df.at[index, 'Description']}):") or df.at[index, 'Description']
        df.loc[index] = [date, category, amount, description]
        messagebox.showinfo("Expense Updated", "Expense updated successfully!")
    elif action == 'delete':
        df = df.drop(index).reset_index(drop=True)
        messagebox.showinfo("Expense Deleted", "Expense deleted successfully!")
    else:
        messagebox.showerror("Invalid Action", "Invalid action.")

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
        messagebox.showinfo("No Expenses", "No expenses for the selected month.")
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

    plot_window = tk.Toplevel(root)
    plot_window.title("Category Expenses")
    canvas = FigureCanvasTkAgg(fig, master=plot_window)
    canvas.draw()
    canvas.get_tk_widget().pack(fill=tk.BOTH, expand=1)

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
    """Processes expense data for 2018-2019 and aggregates it by Year-Month."""
    expenses_df['Date'] = pd.to_datetime(expenses_df['Date'])
    expenses_df = expenses_df[expenses_df['Date'].dt.year.isin([2018, 2019])]
    expenses_df['YearMonth'] = expenses_df['Date'].dt.to_period('M').astype(str)

    # Aggregate expenses
    return expenses_df.groupby('YearMonth', as_index=False)['Expense'].sum()

# Plot budget trend over time
def forecast_expenses(budget_df, expenses_df):
    expenses_df = process_expense_data(expenses_df)
    merged_df = budget_df.merge(expenses_df, how='inner', on='YearMonth', suffixes=('_budget', '_expense'))
    
 # Assign month index
    merged_df ['MonthIndex'] = np.arange(len(merged_df))   
    scaler = StandardScaler()
    merged_df[['MonthIndex']] = scaler.fit_transform(merged_df[['MonthIndex']])
    
    # Train Linear Regression Model
    X = merged_df[['MonthIndex']]
    y = merged_df['Amount_expense']
    model = LinearRegression()
    model.fit(X, y)
    
  


    
    # Predict for the next 6 months
    future_months = 6
    last_month_index = merged_df['MonthIndex'].max()
    future_indices = np.arange(last_month_index + 1, last_month_index + future_months + 1).reshape(-1, 1)
    future_forecast = model.predict(future_indices)

    # Generate future dates
    last_date = pd.to_datetime(budget_df['YearMonth'].max())
    future_dates = [last_date + relativedelta(months=i) for i in range(1, future_months + 1)]

    return pd.DataFrame({'YearMonth': [d.strftime('%Y-%m') for d in future_dates], 'Expense': future_forecast})

def plot_budget_trend(budget_df, forecast_df):
    """Plots actual and predicted expenses."""
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(budget_df['YearMonth'], budget_df['Amount'], marker='o', linestyle='-', color='blue', label='Actual Expense')
    ax.plot(forecast_df['YearMonth'], forecast_df['Expense'], marker='o', linestyle='--', color='red', label='Predicted Expense')

    ax.set_title('Expense Trend & Forecast (2018-2019)')
    ax.set_xlabel('Year-Month')
    ax.set_ylabel('Total Expense (€)')
    plt.xticks(rotation=45)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    plot_window = tk.Toplevel(root)
    plot_window.title("Expense Trend & Forecast")
    canvas = FigureCanvasTkAgg(fig, master=plot_window)
    canvas.draw()
    canvas.get_tk_widget().pack(fill=tk.BOTH, expand=1)
    
    def show_budget_trend(email):
        budget_df = prepare_budget_data_for_trend()
        expenses_file, df = handle_expenses_file(email) 
        forecast_df = forecast_expenses(budget_df, df) 
    plot_budget_trend(budget_df, forecast_df)

# Select month and year
def select_month_year():
    year = simpledialog.askinteger("Select Month and Year", "Enter the year (e.g., 2023):")
    month = simpledialog.askinteger("Select Month and Year", "Enter the month (1-12):")
    if not year or not month or month < 1 or month > 12:
        messagebox.showerror("Invalid Input", "Invalid year or month. Please try again.")
        return None, None
    return year, month

# Main menu
def main_menu(email):
    budget_df = prepare_budget_data_for_trend()
    expenses_file, df = handle_expenses_file(email)
    profiles_df = load_profiles()
    user_profile = profiles_df[profiles_df['Email'] == email].iloc[0]
    budget = user_profile['Budget']
    expenses_by_month = {}

    def show_view_expenses():
        view_expenses(df)

    def show_add_expense():
        nonlocal df, expenses_by_month
        df, expenses_by_month = add_expense(expenses_file, df, budget, expenses_by_month)

    def show_edit_delete_expense():
        nonlocal df
        df = edit_delete_expense(expenses_file, df)

    def show_monthly_summary():
        year, month = select_month_year()
        if year is None or month is None:
            return
        total_expenses = get_monthly_expenses(df, year, month)
        category_totals = get_category_totals(df, year, month)
        summary_window = tk.Toplevel(root)
        summary_window.title(f"Monthly Summary - {datetime(year, month, 1).strftime('%B %Y')}")
        summary_text = tk.Text(summary_window)
        summary_text.insert(tk.END, f"Total expenses for {datetime(year, month, 1).strftime('%B %Y')}: €{total_expenses:.2f}\n")
        summary_text.insert(tk.END, "Category breakdown:\n")
        for category, amount in category_totals.items():
            summary_text.insert(tk.END, f"{category}: €{amount:.2f}\n")
        summary_text.pack()

    def show_plot_expenses_by_category():
        year, month = select_month_year()
        if year is None or month is None:
            return
        plot_expenses_by_category(df, year, month, budget)

    def show_budget_trend():
        budget_df = prepare_budget_data_for_trend()
        expenses_file, df = handle_expenses_file(email)
        forecast_df = forecast_expenses(budget_df, df)
        plot_budget_trend(budget_df, forecast_df)

    def show_select_month_year():
        year, month = select_month_year()
        if year is None or month is None:
            return
        messagebox.showinfo("Selected Month and Year", f"Selected Month and Year: {datetime(year, month, 1).strftime('%B %Y')}")

    def show_budget_trend_wrapper():
     show_budget_trend() 

    menu_window = tk.Toplevel(root)
    menu_window.title("Expense Tracker Menu")
    button_frame = tk.Frame(menu_window)

    buttons = [
        ("View Expenses", show_view_expenses),
        ("Add Expense", show_add_expense),
        ("Edit/Delete Expense", show_edit_delete_expense),
        ("View Monthly Summary", show_monthly_summary),
        ("Plot Expenses by Category", show_plot_expenses_by_category),
        ("View Budget Trend", show_budget_trend_wrapper),
        ("Select Month and Year", show_select_month_year),
        ("Exit", menu_window.destroy)
    ]

    for text, command in buttons:
        button = tk.Button(button_frame, text=text, command=command)
        button.pack(fill=tk.X, padx=10, pady=5)

    button_frame.pack(padx=10, pady=10)

if __name__ == "__main__":
    initialize_profiles_file()

    root = tk.Tk()
    root.title("Expense Tracker")

    def on_login():
        email = email_entry.get().strip()
        if user_exists(email):
            messagebox.showinfo("Welcome Back", "Welcome back!")
            main_menu(email)
        else:
            messagebox.showinfo("No Profile", "No profile found for this email.")
            create_profile()

    email_label = tk.Label(root, text="Enter your email:")
    email_label.pack(padx=10, pady=5)
    email_entry = tk.Entry(root)
    email_entry.pack(padx=10, pady=5)
    login_button = tk.Button(root, text="Login", command=on_login)
    login_button.pack(padx=10, pady=5)

    root.mainloop()


