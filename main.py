import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from lifelines import KaplanMeierFitter
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder

# Set larger font sizes globally
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['xtick.labelsize'] = 10
plt.rcParams['ytick.labelsize'] = 10
plt.rcParams['legend.fontsize'] = 10

def load_data():
    data = pd.read_csv('WA_Fn-UseC_-Telco-Customer-Churn.csv')
    data['TotalCharges'] = pd.to_numeric(data['TotalCharges'], errors='coerce')
    data['TotalCharges'] = data['TotalCharges'].fillna(data['TotalCharges'].median())
    return data

def churn_rate_analysis(data):
    """Analysis 1: Churn Rate by Categories (3 separate plots)"""
    for col in ['Contract', 'PaymentMethod', 'InternetService']:
        plt.figure(figsize=(10, 6))
        churn_rate = data.groupby(col)['Churn'].value_counts(normalize=True).unstack()
        churn_rate.plot(kind='bar', stacked=True)
        plt.title(f'Churn Rate by {col}')
        plt.ylabel('Proportion')
        plt.xlabel(col)
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(f'churn_rate_{col.lower()}.png', dpi=100)
        plt.close()

def tenure_analysis(data):
    """Analysis 2: Tenure Impact"""
    data['TenureGroup'] = pd.cut(data['tenure'], bins=[0, 6, 12, 24, 60, np.inf],
                                labels=['0-6mo', '6-12mo', '1-2yr', '2-5yr', '5+yr'])
    
    plt.figure(figsize=(10, 6))
    sns.barplot(x='TenureGroup', y='Churn', 
                data=data.replace({'Churn': {'Yes': 1, 'No': 0}}),
                estimator=np.mean)
    plt.title('Churn Rate by Tenure Group')
    plt.xlabel('Tenure Group')
    plt.ylabel('Churn Rate')
    plt.tight_layout()
    plt.savefig('tenure_analysis.png', dpi=100)
    plt.close()

def financial_impact(data):
    """Analysis 3: Revenue Impact (2 separate plots)"""
    data['ChurnBinary'] = data['Churn'].map({'Yes': 1, 'No': 0})
    data['LTV'] = data['MonthlyCharges'] * data['tenure']
    
    # Monthly Charges plot
    plt.figure(figsize=(8, 6))
    sns.boxplot(x='Churn', y='MonthlyCharges', data=data)
    plt.title('Monthly Charges Distribution')
    plt.xlabel('Churn Status')
    plt.ylabel('Monthly Charges ($)')
    plt.tight_layout()
    plt.savefig('financial_monthly_charges.png', dpi=100)
    plt.close()
    
    # LTV plot
    plt.figure(figsize=(8, 6))
    sns.boxplot(x='Churn', y='LTV', data=data)
    plt.title('Lifetime Value Distribution')
    plt.xlabel('Churn Status')
    plt.ylabel('Lifetime Value ($)')
    plt.tight_layout()
    plt.savefig('financial_ltv.png', dpi=100)
    plt.close()
    
    churned_revenue = data[data['Churn'] == 'Yes']['MonthlyCharges'].sum()
    print(f"\nMonthly Revenue Lost to Churn: ${churned_revenue:,.2f}")

def survival_analysis(data):
    """Analysis 4: Contract Survival Rates"""
    plt.figure(figsize=(10, 6))
    
    for contract in ['Month-to-month', 'One year', 'Two year']:
        subset = data[data['Contract'] == contract]
        kmf = KaplanMeierFitter()
        kmf.fit(subset['tenure'], 
                subset['Churn'].map({'Yes': 1, 'No': 0}), 
                label=contract)
        kmf.plot_survival_function(linewidth=2)
    
    plt.title('Customer Retention by Contract Type')
    plt.xlabel('Tenure (months)')
    plt.ylabel('Retention Probability')
    plt.legend(title='Contract Type')
    plt.tight_layout()
    plt.savefig('survival_analysis.png', dpi=100)
    plt.close()

def feature_importance(data):
    """Feature Importance Analysis"""
    # Prepare data
    X = data.drop(['Churn', 'TotalCharges', 'customerID'], axis=1)
    y = data['Churn'].map({'Yes': 1, 'No': 0})
    
    # Encode categorical variables
    for col in X.select_dtypes(include=['object']).columns:
        X[col] = LabelEncoder().fit_transform(X[col])
    
    # Train Random Forest
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X, y)
    
    # Get feature importance
    importance = pd.DataFrame({
        'Feature': X.columns,
        'Importance': model.feature_importances_
    }).sort_values('Importance', ascending=False).head(15)
    
    # Plot
    plt.figure(figsize=(10, 6))
    sns.barplot(x='Importance', y='Feature', data=importance, palette='Blues_r')
    plt.title('Top 15 Important Features for Churn Prediction')
    plt.xlabel('Relative Importance')
    plt.ylabel('Feature')
    plt.tight_layout()
    plt.savefig('feature_importance.png', dpi=100)
    plt.close()

def main():
    data = load_data()
    
    print("Running Churn Rate Analysis...")
    churn_rate_analysis(data)
    
    print("Running Tenure Analysis...")
    tenure_analysis(data)
    
    print("Running Financial Impact Analysis...")
    financial_impact(data)
    
    print("Running Survival Analysis...")
    survival_analysis(data)
    
    print("Running Feature Importance Analysis...")
    feature_importance(data)

if __name__ == '__main__':
    main()
