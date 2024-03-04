import pandas as pd
import numpy as np

data=pd.read_csv("/Users/sowmyaeemani/Desktop/projects/sales prediction/data/olist_customers_dataset.csv")

df= pd.DataFrame(data)

df = df.drop(
                [
                    "order_approved_at",
                    "order_delivered_carrier_date",
                    "order_delivered_customer_date",
                    "order_estimated_delivery_date",
                    "order_purchase_timestamp",
                ],
                axis=1,
            )
df["product_weight_g"].fillna(df["product_weight_g"].median(), inplace=True)
df["product_length_cm"].fillna(df["product_length_cm"].median(), inplace=True)
df["product_height_cm"].fillna(df["product_height_cm"].median(), inplace=True)
df["product_width_cm"].fillna(df["product_width_cm"].median(), inplace=True)
# write "No review" in review_comment_message column
df["review_comment_message"].fillna("No review", inplace=True)

df = df.select_dtypes(include=[np.number])
cols_to_drop = ["customer_zip_code_prefix", "order_item_id"]
df = df.drop(cols_to_drop, axis=1)

# Check for NaN values in the whole DataFrame
nan_values = df.isna()

# Count NaN values in each column
nan_counts_per_column = df.isna().sum()

print("NaN values in the whole DataFrame:")
print(nan_values)

print("\nNaN counts per column:")
print(nan_counts_per_column)