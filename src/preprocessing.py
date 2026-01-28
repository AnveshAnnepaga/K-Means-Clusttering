from sklearn.preprocessing import StandardScaler

def preprocess_data(df):
    features = df[
        ['Fresh', 'Milk', 'Grocery', 'Frozen', 'Detergents_Paper', 'Delicassen']
    ]
    scaler = StandardScaler()
    scaled = scaler.fit_transform(features)
    return scaled, features.columns