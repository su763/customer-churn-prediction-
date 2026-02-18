import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
import joblib
import os

print("🚀 Starting the Model Training Pipeline...")

# 1. Load the data
df = pd.read_excel('../Telco_customer_churn.xlsx')

# 2. Basic cleaning (same as your notebook)
df['Total Charges'] = pd.to_numeric(df['Total Charges'], errors='coerce')
df = df.dropna(subset=['Total Charges'])

# 3. Separate features (X) and target (y)
y = df['Churn Value'] # This is our 1 or 0 target

# Drop columns that are useless or cause data leakage
drop_cols = ['CustomerID', 'Count', 'Country', 'State', 'City', 'Zip Code', 
             'Lat Long', 'Latitude', 'Longitude', 'Churn Label', 
             'Churn Value', 'Churn Score', 'Churn Reason', 'CLTV']
X = df.drop(columns=drop_cols)

# 4. Split into Training and Testing sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
print(f"Data split: {len(X_train)} training rows, {len(X_test)} testing rows.")

# 5. Define which columns are numbers and which are text
numeric_features = ['Tenure Months', 'Monthly Charges', 'Total Charges']
categorical_features = [col for col in X.columns if col not in numeric_features]

# 6. Create the Preprocessing Engine (The ColumnTransformer)
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numeric_features),
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
    ])

# 7. Build the full pipeline (Preprocess -> Train Model)
# Notice 'class_weight="balanced"'. This automatically fixes the 73/26 imbalance you found!
model_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', RandomForestClassifier(random_state=42, class_weight='balanced'))
])

# 8. Train the model!
print("🧠 Training the Random Forest model...")
model_pipeline.fit(X_train, y_train)

# 9. Evaluate the model
print("\n📊 Model Evaluation on Test Data:")
y_pred = model_pipeline.predict(X_test)
print(classification_report(y_test, y_pred))

# 10. Save the trained pipeline for the API later
os.makedirs('models', exist_ok=True)
joblib.dump(model_pipeline, 'models/churn_model.pkl')
print("\n✅ Model saved successfully to models/churn_model.pkl!")