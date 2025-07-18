import pandas as pd
import numpy as np
from mlxtend.frequent_patterns import apriori
from mlxtend.frequent_patterns import association_rules
import matplotlib.pyplot as plt
import seaborn as sns

# Read the cleaned data
df = pd.read_csv('data/online_retail_II_cleaned.csv')

# Create a basket matrix (one-hot encoded DataFrame)
basket = (df.groupby(['Invoice', 'Description'])['Quantity']
          .sum().unstack().reset_index().fillna(0)
          .set_index('Invoice'))

# Convert quantities to boolean (1 for purchased, 0 for not purchased)
basket_sets = (basket > 0).astype(bool)

# Generate frequent itemsets
min_support = 0.01
frequent_itemsets = apriori(basket_sets, 
                          min_support=min_support, 
                          use_colnames=True)

# Generate association rules
min_confidence = 0.5
rules = association_rules(frequent_itemsets, 
                        metric="confidence", 
                        min_threshold=min_confidence)

# Calculate additional metrics
rules["conviction"] = np.where(rules["confidence"] == 1,
                                float('inf'),
                                (1 - rules["antecedent support"]) / (1 - rules["confidence"]))

# Sort rules by lift
rules = rules.sort_values('lift', ascending=False)

# Save results to CSV
rules.to_csv('market_basket/association_rules.csv', index=False)

# Print top 10 rules with better formatting
print("\nTop 10 Association Rules by Lift:")
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.width', 1000)
pd.set_option('display.max_colwidth', None)

# Format the rules for better readability
for idx, rule in rules[['antecedents', 'consequents', 'support', 'confidence', 'lift']].head(10).iterrows():
    print(f"\nRule {idx + 1}:")
    print(f"If customer buys: {list(rule['antecedents'])}")
    print(f"They are likely to buy: {list(rule['consequents'])}")
    print(f"Support: {rule['support']:.3f}")
    print(f"Confidence: {rule['confidence']:.3f}")
    print(f"Lift: {rule['lift']:.3f}")

# Visualize the distribution of metrics
plt.figure(figsize=(12, 8))
plt.subplot(221)
sns.histplot(rules['support'])
plt.title('Support Distribution')

plt.subplot(222)
sns.histplot(rules['confidence'])
plt.title('Confidence Distribution')

plt.subplot(223)
sns.histplot(rules['lift'])
plt.title('Lift Distribution')

plt.subplot(224)
sns.scatterplot(data=rules, x='support', y='confidence', size='lift', sizes=(50, 400))
plt.title('Support vs Confidence')

plt.tight_layout()
plt.savefig('market_basket/rule_metrics.png')
plt.close()

# Visualize support vs confidence with lift as size
plt.figure(figsize=(12, 8)) # Adjust figure size as needed for a single plot
sns.scatterplot(data=rules, x='support', y='confidence', size='lift', sizes=(50, 400))
plt.title('Support vs Confidence')

plt.tight_layout()
plt.savefig('market_basket/support_confidence_scatter.png') # New filename for this plot
plt.close()

# Print summary statistics
print("\nAssociation Rules Summary:")
print(f"Total number of rules generated: {len(rules)}")
print(f"Average lift: {rules['lift'].mean():.2f}")
print(f"Average confidence: {rules['confidence'].mean():.2f}")
print(f"Average support: {rules['support'].mean():.2f}")

# Save itemsets to CSV
frequent_itemsets.to_csv('market_basket/frequent_itemsets.csv', index=False)
print(f"\nResults saved to 'market_basket/association_rules.csv' and 'market_basket/frequent_itemsets.csv'")
print(f"Visualizations saved to 'market_basket/rule_metrics.png'")
print()