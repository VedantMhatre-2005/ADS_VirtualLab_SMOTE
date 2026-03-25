import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from collections import Counter
import warnings

warnings.filterwarnings('ignore')

try:
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers, Model
    from tensorflow.keras.optimizers import Adam
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False


class SimpleGAN:
    """
    Simple GAN-based generative model for class balancing.
    Generates synthetic minority class samples.
    """
    
    def __init__(self, latent_dim=20, epochs=100, batch_size=32, random_state=42):
        """
        Initialize GAN.
        
        Parameters:
        -----------
        latent_dim : int
            Dimension of latent space
        epochs : int
            Number of training epochs
        batch_size : int
            Batch size for training
        random_state : int
            Random state for reproducibility
        """
        if not TF_AVAILABLE:
            raise ImportError("TensorFlow is required for GAN functionality")
        
        self.latent_dim = latent_dim
        self.epochs = epochs
        self.batch_size = batch_size
        self.random_state = random_state
        self.generator = None
        self.discriminator = None
        self.gan = None
        self.scaler = MinMaxScaler()
        
        np.random.seed(random_state)
        tf.random.set_seed(random_state)
    
    def _build_generator(self, input_dim):
        """Build generator network."""
        model = keras.Sequential([
            layers.Dense(64, activation='relu', input_dim=self.latent_dim),
            layers.BatchNormalization(),
            layers.Dense(128, activation='relu'),
            layers.BatchNormalization(),
            layers.Dense(input_dim, activation='sigmoid')
        ])
        return model
    
    def _build_discriminator(self, input_dim):
        """Build discriminator network."""
        model = keras.Sequential([
            layers.Dense(128, activation='relu', input_dim=input_dim),
            layers.Dropout(0.3),
            layers.Dense(64, activation='relu'),
            layers.Dropout(0.3),
            layers.Dense(1, activation='sigmoid')
        ])
        return model
    
    def apply_gan(self, X_train, y_train, verbose=False):
        """
        Train GAN and generate synthetic samples.
        
        Parameters:
        -----------
        X_train, y_train : arrays
            Training data
        verbose : bool
            Whether to print training progress
            
        Returns:
        --------
        X_train_gan, y_train_gan : arrays
            Balanced training data
        """
        # Ensure indices are aligned
        if isinstance(X_train, pd.DataFrame) and isinstance(y_train, pd.Series):
            if not X_train.index.equals(y_train.index):
                X_train = X_train.reset_index(drop=True)
                y_train = y_train.reset_index(drop=True)
        
        # Separate majority and minority classes
        X_majority = X_train[y_train == 0]
        X_minority = X_train[y_train == 1]
        y_majority = y_train[y_train == 0]
        y_minority = y_train[y_train == 1]
        
        # Normalize minority class data
        X_minority_scaled = self.scaler.fit_transform(X_minority)
        
        # Build and compile networks
        input_dim = X_train.shape[1]
        self.generator = self._build_generator(input_dim)
        self.discriminator = self._build_discriminator(input_dim)
        
        self.discriminator.compile(
            optimizer=Adam(learning_rate=0.0002),
            loss='binary_crossentropy'
        )
        
        self.generator.compile(
            optimizer=Adam(learning_rate=0.0002),
            loss='binary_crossentropy'
        )
        
        # Build GAN
        noise_input = layers.Input(shape=(self.latent_dim,))
        generated_data = self.generator(noise_input)
        validity = self.discriminator(generated_data)
        self.gan = Model(noise_input, validity)
        self.gan.compile(
            optimizer=Adam(learning_rate=0.0002),
            loss='binary_crossentropy'
        )
        
        # Train GAN
        n_samples_to_generate = len(X_majority) - len(X_minority)
        
        for epoch in range(self.epochs):
            # Train discriminator
            idx = np.random.randint(0, len(X_minority_scaled), self.batch_size)
            real_data = X_minority_scaled[idx]
            
            noise = np.random.normal(0, 1, (self.batch_size, self.latent_dim))
            fake_data = self.generator.predict(noise, verbose=0)
            
            # Combine real and fake data
            real_labels = np.ones((self.batch_size, 1))
            fake_labels = np.zeros((self.batch_size, 1))
            
            self.discriminator.train_on_batch(real_data, real_labels)
            self.discriminator.train_on_batch(fake_data, fake_labels)
            
            # Train generator
            noise = np.random.normal(0, 1, (self.batch_size, self.latent_dim))
            valid_labels = np.ones((self.batch_size, 1))
            self.gan.train_on_batch(noise, valid_labels)
            
            if verbose and (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch + 1}/{self.epochs}")
        
        # Generate synthetic samples
        noise = np.random.normal(0, 1, (n_samples_to_generate, self.latent_dim))
        synthetic_data = self.generator.predict(noise, verbose=0)
        synthetic_data = self.scaler.inverse_transform(synthetic_data)
        
        # Combine all data
        X_train_gan = np.vstack([X_train, synthetic_data])
        y_train_gan = np.hstack([y_train, np.ones(n_samples_to_generate, dtype=int)])
        
        # Convert back to DataFrame if input was DataFrame
        if isinstance(X_train, pd.DataFrame):
            X_train_gan = pd.DataFrame(X_train_gan, columns=X_train.columns)
        
        if isinstance(y_train, pd.Series):
            y_train_gan = pd.Series(y_train_gan, name=y_train.name)
        
        return X_train_gan, y_train_gan
    
    @staticmethod
    def get_distribution_info(y_train_original, y_train_gan):
        """
        Get class distribution before and after GAN.
        
        Returns:
        --------
        dict : Distribution information
        """
        original_dist = Counter(y_train_original)
        gan_dist = Counter(y_train_gan)
        
        return {
            "Original Distribution": dict(original_dist),
            "GAN Distribution": dict(gan_dist),
            "Original Ratio": f"{original_dist[0] / original_dist[1]:.2f}:1",
            "GAN Ratio": f"{gan_dist[0] / gan_dist[1]:.2f}:1",
            "Samples Generated": gan_dist[1] - original_dist[1]
        }


class GANHandler:
    """Wrapper for GAN operations."""
    
    def __init__(self, epochs=50, random_state=42):
        """Initialize GAN handler."""
        if TF_AVAILABLE:
            self.gan = SimpleGAN(epochs=epochs, random_state=random_state)
        else:
            self.gan = None
    
    def apply_gan(self, X_train, y_train, verbose=False):
        """Apply GAN balancing."""
        if not TF_AVAILABLE or self.gan is None:
            # Fallback: use SMOTE if GAN is not available
            from imblearn.over_sampling import SMOTE
            smote = SMOTE(random_state=42)
            X_balanced, y_balanced = smote.fit_resample(X_train, y_train)
            
            if isinstance(X_train, pd.DataFrame):
                X_balanced = pd.DataFrame(X_balanced, columns=X_train.columns)
            if isinstance(y_train, pd.Series):
                y_balanced = pd.Series(y_balanced, name=y_train.name)
            
            return X_balanced, y_balanced, None
        
        X_gan, y_gan = self.gan.apply_gan(X_train, y_train, verbose=verbose)
        return X_gan, y_gan, None
    
    @staticmethod
    def get_distribution_dataframe(y_train_original, y_train_gan):
        """Get distribution as dataframe."""
        original_counts = pd.Series(y_train_original).value_counts().sort_index()
        gan_counts = pd.Series(y_train_gan).value_counts().sort_index()
        
        df = pd.DataFrame({
            "Class": ["Majority (Class 0)", "Minority (Class 1)"],
            "Original Count": [original_counts[0], original_counts[1]],
            "After GAN": [gan_counts[0], gan_counts[1]]
        })
        
        return df
