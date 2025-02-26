import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from lifetimes import BetaGeoFitter, GammaGammaFitter
from lifetimes.utils import summary_data_from_transaction_data
import datetime as dt
import warnings
warnings.filterwarnings('ignore')

# Set random seed for reproducibility
np.random.seed(42)

# Simulate customer data
def simulate_customer_data(n_customers=5000):
    """
    Simulate e-commerce customer data with:
    - Customer demographics
    - Purchase history
    - Product interactions
    """
    # Customer ID
    customer_ids = np.arange(1, n_customers + 1)
    
    # Demographics
    ages = np.random.normal(35, 12, n_customers).astype(int)
    # Clip ages to realistic range
    ages = np.clip(ages, 18, 80)
    
    # Gender (0 for female, 1 for male, 2 for non-binary)
    genders = np.random.choice([0, 1, 2], size=n_customers, p=[0.48, 0.48, 0.04])
    
    # Income levels (in thousands)
    incomes = np.random.lognormal(mean=4.1, sigma=0.4, size=n_customers) * 10
    
    # Location - Random zip codes
    zip_codes = np.random.randint(10000, 99999, n_customers)
    
    # Signup date - Between 1-3 years ago
    signup_days_ago = np.random.randint(365, 365*3, n_customers)
    today = dt.datetime.now()
    signup_dates = [today - dt.timedelta(days=x) for x in signup_days_ago]
    
    # Purchase behavior
    # Average order value
    avg_order_value = np.random.lognormal(mean=3.5, sigma=0.7, size=n_customers) * 5
    
    # Purchase frequency (purchases per year)
    purchase_freq = np.random.lognormal(mean=1.5, sigma=1.0, size=n_customers)
    
    # Last purchase recency (days)
    recency_days = np.random.exponential(scale=100, size=n_customers).astype(int)
    # Ensure recency is not more than signup days
    recency_days = np.minimum(recency_days, signup_days_ago)
    
    # Total lifetime purchases
    tenure_years = signup_days_ago / 365
    total_purchases = np.random.poisson(purchase_freq * tenure_years)
    
    # Create the customer dataframe
    customers_df = pd.DataFrame({
        'customer_id': customer_ids,
        'age': ages,
        'gender': genders,
        'income': incomes,
        'zip_code': zip_codes,
        'signup_date': signup_dates,
        'avg_order_value': avg_order_value,
        'purchase_frequency': purchase_freq,
        'recency_days': recency_days,
        'total_purchases': total_purchases
    })
    
    # Add some derived features
    customers_df['tenure_days'] = signup_days_ago
    customers_df['tenure_years'] = customers_df['tenure_days'] / 365
    customers_df['total_spent'] = customers_df['avg_order_value'] * customers_df['total_purchases']
    
    # Convert gender codes to labels
    customers_df['gender'] = customers_df['gender'].map({0: 'Female', 1: 'Male', 2: 'Non-binary'})
    
    return customers_df

# Generate transaction data based on customer data
def simulate_transaction_data(customers_df):
    """
    Generate detailed transaction records based on customer profiles
    """
    transactions = []
    
    for _, customer in customers_df.iterrows():
        customer_id = customer['customer_id']
        n_transactions = customer['total_purchases']
        
        if n_transactions > 0:
            # Calculate the first purchase date (signup date)
            first_purchase = customer['signup_date']
            
            # Calculate average time between purchases
            if n_transactions > 1:
                avg_time_between_purchases = (customer['tenure_days'] - customer['recency_days']) / (n_transactions - 1)
            else:
                avg_time_between_purchases = 0
                
            # Generate transaction dates
            if n_transactions == 1:
                # If only one purchase, it happened at signup
                purchase_dates = [first_purchase]
            else:
                # Space out purchases with some randomness
                days_since_first = np.sort(np.random.choice(
                    range(1, int(customer['tenure_days'] - customer['recency_days'])), 
                    size=min(n_transactions-1, 1000),  # Cap to avoid excessive transactions
                    replace=True if n_transactions > 1000 else False
                ))
                
                purchase_dates = [first_purchase + dt.timedelta(days=int(days)) for days in days_since_first]
                # Add the most recent purchase
                purchase_dates.append(dt.datetime.now() - dt.timedelta(days=int(customer['recency_days'])))
                
            # Make sure we have the right number of transactions
            if len(purchase_dates) > n_transactions:
                purchase_dates = purchase_dates[:n_transactions]
            
            # Generate transaction amounts with some variability around avg_order_value
            amounts = np.random.lognormal(
                mean=np.log(customer['avg_order_value']),
                sigma=0.3,
                size=len(purchase_dates)
            )
            
            # Create transaction records
            for date, amount in zip(purchase_dates, amounts):
                transactions.append({
                    'customer_id': customer_id,
                    'date': date,
                    'amount': amount
                })
    
    # Convert to DataFrame and sort by date
    transactions_df = pd.DataFrame(transactions)
    transactions_df = transactions_df.sort_values('date')
    
    return transactions_df

# Main analysis function
def analyze_customer_segments(customers_df, transactions_df):
    """
    Perform RFM analysis, customer segmentation, and CLV prediction
    """
    print("Step 1: Exploring the dataset")
    print(f"Number of customers: {len(customers_df)}")
    print(f"Number of transactions: {len(transactions_df)}")
    print("\nCustomer data sample:")
    print(customers_df.head())
    print("\nTransaction data sample:")
    print(transactions_df.head())
    
    # Basic statistics
    print("\nCustomer statistics:")
    print(customers_df[['age', 'income', 'tenure_days', 'total_spent', 'total_purchases']].describe())
    
    # Gender distribution
    plt.figure(figsize=(10, 6))
    sns.countplot(x='gender', data=customers_df)
    plt.title('Gender Distribution')
    plt.savefig('gender_distribution.png')
    
    # Age distribution
    plt.figure(figsize=(10, 6))
    sns.histplot(customers_df['age'], bins=20, kde=True)
    plt.title('Age Distribution')
    plt.savefig('age_distribution.png')
    
    # Income vs. Total Spent
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x='income', y='total_spent', data=customers_df, alpha=0.6)
    plt.title('Income vs. Total Spent')
    plt.savefig('income_vs_spent.png')
    
    # RFM Analysis
    print("\nStep 2: RFM Analysis")
    
    # Prepare RFM metrics
    today = dt.datetime.now()
    
    rfm_data = transactions_df.groupby('customer_id').agg({
        'date': lambda x: (today - x.max()).days,  # Recency
        'customer_id': 'count',  # Frequency
        'amount': 'sum'  # Monetary
    }).rename(columns={
        'date': 'recency',
        'customer_id': 'frequency',
        'amount': 'monetary'
    }).reset_index()
    
    # Merge with original customer data
    rfm_df = pd.merge(rfm_data, customers_df[['customer_id', 'age', 'gender', 'income']], on='customer_id')
    
    # RFM Score calculation
    rfm_df['R_Score'] = pd.qcut(rfm_df['recency'], 5, labels=[5, 4, 3, 2, 1])
    rfm_df['F_Score'] = pd.qcut(rfm_df['frequency'], 5, labels=[1, 2, 3, 4, 5])
    rfm_df['M_Score'] = pd.qcut(rfm_df['monetary'], 5, labels=[1, 2, 3, 4, 5])
    
    # Overall RFM Score
    rfm_df['RFM_Score'] = rfm_df['R_Score'].astype(int) + rfm_df['F_Score'].astype(int) + rfm_df['M_Score'].astype(int)
    
    # Customer Segment based on RFM Score
    def segment_customer(score):
        if score >= 13:
            return 'Champions'
        elif score >= 10:
            return 'Loyal Customers'
        elif score >= 7:
            return 'Potential Loyalists'
        elif score >= 5:
            return 'At Risk'
        else:
            return 'Needs Attention'
    
    rfm_df['RFM_Segment'] = rfm_df['RFM_Score'].apply(segment_customer)
    
    print("RFM Segments:")
    print(rfm_df['RFM_Segment'].value_counts())
    
    # Visualize segments
    plt.figure(figsize=(10, 6))
    sns.countplot(x='RFM_Segment', data=rfm_df, order=rfm_df['RFM_Segment'].value_counts().index)
    plt.title('Customer Segments based on RFM')
    plt.xticks(rotation=45)
    plt.savefig('rfm_segments.png')
    
    # K-means Clustering
    print("\nStep 3: K-means Clustering")
    
    # Select features for clustering
    features = ['recency', 'frequency', 'monetary', 'age', 'income']
    X = rfm_df[features].copy()
    
    # Scale the data
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Determine optimal number of clusters
    silhouette_scores = []
    for k in range(2, 11):
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        clusters = kmeans.fit_predict(X_scaled)
        silhouette_scores.append(silhouette_score(X_scaled, clusters))
    
    # Plot silhouette scores
    plt.figure(figsize=(10, 6))
    plt.plot(range(2, 11), silhouette_scores, marker='o')
    plt.xlabel('Number of clusters')
    plt.ylabel('Silhouette Score')
    plt.title('Silhouette Score for different numbers of clusters')
    plt.savefig('silhouette_scores.png')
    
    # Get the best number of clusters
    best_k = np.argmax(silhouette_scores) + 2
    print(f"Optimal number of clusters: {best_k}")
    
    # Perform K-means with the optimal number of clusters
    kmeans = KMeans(n_clusters=best_k, random_state=42, n_init=10)
    rfm_df['Cluster'] = kmeans.fit_predict(X_scaled)
    
    # Analyze the clusters
    cluster_analysis = rfm_df.groupby('Cluster').agg({
        'recency': 'mean',
        'frequency': 'mean',
        'monetary': 'mean',
        'age': 'mean',
        'income': 'mean',
        'customer_id': 'count'
    }).rename(columns={'customer_id': 'count'})
    
    print("\nCluster Analysis:")
    print(cluster_analysis)
    
    # Visualize clusters
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x='recency', y='monetary', hue='Cluster', data=rfm_df, palette='viridis')
    plt.title('Clusters by Recency and Monetary Value')
    plt.savefig('clusters_recency_monetary.png')
    
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x='frequency', y='monetary', hue='Cluster', data=rfm_df, palette='viridis')
    plt.title('Clusters by Frequency and Monetary Value')
    plt.savefig('clusters_frequency_monetary.png')
    
    # Customer Lifetime Value Prediction
    print("\nStep 4: Customer Lifetime Value Prediction")
    
    # Format the data for the lifetimes package
    df_for_clv = transactions_df.copy()
    df_for_clv['date'] = pd.to_datetime(df_for_clv['date'])
    
    # Calculate summary data
    summary_data = summary_data_from_transaction_data(
        transactions_df,
        'customer_id',
        'date',
        'amount',
        observation_period_end=today
    )
    
    # Fit BG/NBD model
    print("Fitting BG/NBD model...")
    bgf = BetaGeoFitter(penalizer_coef=0.01)
    bgf.fit(summary_data['frequency'], summary_data['recency'], summary_data['T'])
    
    # Fit Gamma-Gamma model
    print("Fitting Gamma-Gamma model...")
    ggf = GammaGammaFitter(penalizer_coef=0.01)
    ggf.fit(
        summary_data[summary_data['frequency'] > 0]['frequency'],
        summary_data[summary_data['frequency'] > 0]['monetary_value']
    )
    
    # Predict future purchases (6 months)
    summary_data['predicted_purchases_6m'] = bgf.predict(180, summary_data['frequency'], summary_data['recency'], summary_data['T'])
    
    # Calculate expected CLV for 1 year
    clv_prediction = ggf.customer_lifetime_value(
        bgf,
        summary_data['frequency'],
        summary_data['recency'],
        summary_data['T'],
        summary_data['monetary_value'],
        time=12,  # months
        discount_rate=0.01  # monthly discount rate
    )
    
    # Add CLV prediction to summary data
    summary_data['CLV_1_year'] = clv_prediction
    
    # Merge CLV with RFM data
    final_df = pd.merge(rfm_df, summary_data[['CLV_1_year', 'predicted_purchases_6m']], left_on='customer_id', right_index=True, how='left')
    
    # Visualize CLV by RFM Segment
    plt.figure(figsize=(12, 8))
    sns.boxplot(x='RFM_Segment', y='CLV_1_year', data=final_df, order=final_df.groupby('RFM_Segment')['CLV_1_year'].mean().sort_values(ascending=False).index)
    plt.title('Customer Lifetime Value by RFM Segment')
    plt.xticks(rotation=45)
    plt.savefig('clv_by_segment.png')
    
    # Visualize CLV by Cluster
    plt.figure(figsize=(12, 8))
    sns.boxplot(x='Cluster', y='CLV_1_year', data=final_df)
    plt.title('Customer Lifetime Value by Cluster')
    plt.savefig('clv_by_cluster.png')
    
    # Top CLV customers
    top_customers = final_df.sort_values('CLV_1_year', ascending=False).head(10)[['customer_id', 'age', 'gender', 'income', 'RFM_Segment', 'Cluster', 'CLV_1_year', 'predicted_purchases_6m']]
    print("\nTop 10 Customers by Predicted CLV:")
    print(top_customers)
    
    # Save the final dataframe
    final_df.to_csv('customer_segmentation_analysis.csv', index=False)
    
    # Cluster naming based on characteristics
    cluster_names = {}
    for cluster in cluster_analysis.index:
        cluster_data = cluster_analysis.loc[cluster]
        
        # High-value active customers
        if cluster_data['monetary'] > cluster_analysis['monetary'].mean() and cluster_data['recency'] < cluster_analysis['recency'].mean():
            cluster_names[cluster] = "High-Value Active"
        # High-value inactive customers    
        elif cluster_data['monetary'] > cluster_analysis['monetary'].mean() and cluster_data['recency'] > cluster_analysis['recency'].mean():
            cluster_names[cluster] = "High-Value Inactive"
        # Low-value active customers
        elif cluster_data['monetary'] < cluster_analysis['monetary'].mean() and cluster_data['recency'] < cluster_analysis['recency'].mean():
            cluster_names[cluster] = "Low-Value Active"
        # Low-value inactive customers
        elif cluster_data['monetary'] < cluster_analysis['monetary'].mean() and cluster_data['recency'] > cluster_analysis['recency'].mean():
            cluster_names[cluster] = "Low-Value Inactive"
        # Middle-value customers
        else:
            cluster_names[cluster] = "Mid-Value"
    
    print("\nCluster Names:")
    for cluster, name in cluster_names.items():
        print(f"Cluster {cluster}: {name}")
    
    # Add cluster names to the final dataframe
    final_df['Cluster_Name'] = final_df['Cluster'].map(cluster_names)
    
    # Business Recommendations
    print("\nStep 5: Business Recommendations")
    
    # For each segment
    segments = final_df['RFM_Segment'].unique()
    for segment in segments:
        segment_data = final_df[final_df['RFM_Segment'] == segment]
        avg_clv = segment_data['CLV_1_year'].mean()
        count = len(segment_data)
        avg_recency = segment_data['recency'].mean()
        avg_frequency = segment_data['frequency'].mean()
        avg_monetary = segment_data['monetary'].mean()
        
        print(f"\nSegment: {segment}")
        print(f"Count: {count} customers")
        print(f"Average CLV: ${avg_clv:.2f}")
        print(f"Average Recency: {avg_recency:.0f} days")
        print(f"Average Frequency: {avg_frequency:.1f} transactions")
        print(f"Average Monetary Value: ${avg_monetary:.2f}")
        
        # Segment-specific recommendations
        print("Recommendations:")
        if segment == 'Champions':
            print("- Implement a loyalty rewards program to maintain engagement")
            print("- Use them as brand ambassadors for referral programs")
            print("- Seek product feedback and reviews")
        elif segment == 'Loyal Customers':
            print("- Offer exclusive access to new products")
            print("- Provide personalized recommendations based on purchase history")
            print("- Create upsell opportunities with complementary products")
        elif segment == 'Potential Loyalists':
            print("- Increase engagement with targeted email campaigns")
            print("- Offer limited-time discounts to encourage more frequent purchases")
            print("- Implement a customer loyalty program")
        elif segment == 'At Risk':
            print("- Reactivation campaign with personalized offers")
            print("- Ask for feedback to understand pain points")
            print("- Consider win-back incentives")
        elif segment == 'Needs Attention':
            print("- Send a 'We miss you' campaign with steep discounts")
            print("- Gather feedback on why they stopped purchasing")
            print("- Consider if this segment is worth targeting based on CLV")
    
    return final_df

# Example usage
if __name__ == "__main__":
    # Create simulated customer data
    print("Simulating customer data...")
    customers = simulate_customer_data(n_customers=5000)
    
    # Create simulated transaction data
    print("Simulating transaction data...")
    transactions = simulate_transaction_data(customers)
    
    # Perform analysis
    print("Performing customer segmentation analysis...")
    results = analyze_customer_segments(customers, transactions)
    
    print("\nAnalysis complete! Results saved to 'customer_segmentation_analysis.csv'")
    print("Visualizations saved as PNG files.")