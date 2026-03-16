import os
import joblib

try:
    import tensorflow as tf
    from tensorflow.keras import layers, models
    from sklearn.decomposition import PCA
    from sklearn.linear_model import LogisticRegression
    # Set up parameters
    os.makedirs('models', exist_ok=True)
    epochs = 1
    batch_size = 64
except ImportError:
    print("TensorFlow not installed. Skipping training as models are pre-trained for Render.")
    import sys
    sys.exit(0)

# Set up parameters
os.makedirs('models', exist_ok=True)
epochs = 1
batch_size = 64

def load_and_preprocess_data():
    print("Loading MNIST dataset...")
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    
    # Reshape and normalize
    x_train = x_train.reshape(-1, 28, 28, 1).astype('float32') / 255.0
    x_test = x_test.reshape(-1, 28, 28, 1).astype('float32') / 255.0
    
    return (x_train, y_train), (x_test, y_test)

def build_cnn():
    print("Building CNN architecture...")
    model = models.Sequential([
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.Flatten(),
        # We'll use the output of the Flatten layer as our feature extractor
        # We add classification layers just to pre-train the feature extractor
        layers.Dense(64, activation='relu', name='feature_layer'),
        layers.Dense(10, activation='softmax')
    ])
    
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    return model

def train_pipeline():
    (x_train, y_train), (x_test, y_test) = load_and_preprocess_data()
    
    # 1. Train the Base CNN
    print("\n--- Training Base CNN ---")
    cnn_model = build_cnn()
    cnn_model.fit(x_train, y_train, epochs=epochs, batch_size=batch_size, validation_data=(x_test, y_test))
    
    # Create the Feature Extractor Model
    # We strip the final classification layer and output the learned dense features
    feature_extractor = models.Model(inputs=cnn_model.input, outputs=cnn_model.get_layer('feature_layer').output)
    feature_extractor.save('models/cnn_feature_extractor.h5')
    print("Saved CNN feature extractor to models/cnn_feature_extractor.h5")
    
    # 2. Extract Features using CNN
    print("\n--- Extracting CNN Features ---")
    # To save memory/time on training logic, we extract on a subset if needed, but MNIST is small enough
    cnn_features_train = feature_extractor.predict(x_train, batch_size=512)
    cnn_features_test = feature_extractor.predict(x_test, batch_size=512)
    
    print(f"Extracted features shape: {cnn_features_train.shape}")
    
    # 3. Fit PCA
    print("\n--- Fitting PCA ---")
    pca = PCA(n_components=0.95) # Reduce dimensions but keep 95% variance
    pca.fit(cnn_features_train)
    
    pca_features_train = pca.transform(cnn_features_train)
    pca_features_test = pca.transform(cnn_features_test)
    
    print(f"PCA reduced dimensions to: {pca.n_components_}")
    
    joblib.dump(pca, 'models/pca_transformer.pkl')
    print("Saved PCA model to models/pca_transformer.pkl")
    
    # 4. Train Logistic Regression
    print("\n--- Training Logistic Regression ---")
    lr_model = LogisticRegression(max_iter=1000)
    lr_model.fit(pca_features_train, y_train)
    
    accuracy = lr_model.score(pca_features_test, y_test)
    print(f"Logistic Regression final validation accuracy: {accuracy:.4f}")
    
    joblib.dump(lr_model, 'models/lr_model.pkl')
    print("Saved Logistic Regression model to models/lr_model.pkl")

if __name__ == '__main__':
    train_pipeline()
    print("\nPipeline training complete and ready for deployment!")
