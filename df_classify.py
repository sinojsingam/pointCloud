import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import time
import sys
import send_email

print("Reading...")
start_read = time.time()

classified_data = pd.read_csv('../working/classification/multiscale/multiscale_features.csv')
rest_data = pd.read_csv('../working/classification/multiscale/multi_features_nonCls.csv')


# # Preprocessing
cls_df = classified_data.dropna()
lln_df = rest_data.dropna()

# Split data into features and target
X = cls_df[['H', 'S', 'V', 'omnivariance', 'eigenentropy','anisotropy',
            'linearity','planarity', 'curvature', 'sphericity', 'verticality', 'height_range',
            'height_below', 'height_above', 'neighbor_H', 'neighbor_S','neighbor_V']]
y = cls_df['classification']

lln = lln_df[['H', 'S', 'V', 'omnivariance', 'eigenentropy','anisotropy',
            'linearity','planarity', 'curvature', 'sphericity', 'verticality', 'height_range',
            'height_below', 'height_above', 'neighbor_H', 'neighbor_S','neighbor_V']]
# Split data into training and testing sets

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

end_read = time.time()
print(f"Read in {round((end_read-start_read)/60,2)} mins. Now training...")
# # Create a model
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
gbt_model = GradientBoostingClassifier(n_estimators=300, learning_rate=0.2,
                                    max_depth=3, random_state=0)
# Train the model
rf_model.fit(X_train, y_train)
gbt_model.fit(X_train, y_train)

end_train = time.time()
print(f"Trained in {round((end_train-end_read)/60,2)} mins. Now predicting...")

# Make predictions
y_pred_rf = rf_model.predict(X_test)
y_pred_gbt = gbt_model.predict(X_test)

predictions_RF = rf_model.predict(lln)
predictions_GBT = gbt_model.predict(lln)

# Evaluate the model

end_pred = time.time()

print(f"Trained in {round((end_pred-end_train)/60,2)} mins. Now writing...")
print("="*10)
print("RF accuracy:", accuracy_score(y_test, y_pred_rf))
print("-"*10)
print("GBT accuracy:", accuracy_score(y_test, y_pred_gbt))
print("="*10)
# Assigning predictions back to full dataset points
lln_copy = lln_df[['X', 'Y', 'Z']]
lln_copy['predictions_RF'] = predictions_RF
lln_copy['predictions_GBT'] = predictions_GBT
lln_copy.to_csv('../working/to_ML/predictionsMulti.csv', index=False)

done_time = time.time()

message = f"""
    Writing took {round((done_time-end_pred)/60,2)} mins. 
    Classification without color data is done.
    \nThe whole process elapsed {round((done_time - start_read)/3600,2)} hours.
    \nClassified LAS file saved in ./results/lln_classified.las.\nGoodbye
    """

print(message)
# add mailme to CLI and get an email notification sent when scipt is done
try:
    if len(sys.argv) > 1 and sys.argv[1]=='mailme':
            send_email.sendNotification(message)
except:
    print("mail was not send, due to API key error")

