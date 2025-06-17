import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.optimizers import Adam
import os
import numpy as np
from pathlib import Path
import json
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

class TrainingCheckpoint(tf.keras.callbacks.Callback):
    def __init__(self, checkpoint_dir):
        super().__init__()
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(exist_ok=True)
        self.epoch_file = self.checkpoint_dir / 'last_epoch.json'
        self.initial_epoch = 0
        
        # Load last epoch if exists
        if self.epoch_file.exists():
            with open(self.epoch_file, 'r') as f:
                self.initial_epoch = json.load(f)['last_epoch'] + 1
    
    def on_epoch_end(self, epoch, logs=None):
        # Save last completed epoch
        with open(self.epoch_file, 'w') as f:
            json.dump({'last_epoch': epoch}, f)

def create_model():
    """Create and return the model architecture"""
    # Load the pretrained model
    base_model = EfficientNetB0(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
    
    # Add custom layers
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(512, activation='relu')(x)
    x = Dropout(0.3)(x)
    predictions = Dense(6, activation='softmax')(x)  # 6 classes (Anthracnose, Cutting Weevil, Die Back, Gall Midge, Powdery Mildew, Sooty Mould)
    
    # Create the full model
    model = Model(inputs=base_model.input, outputs=predictions)
    
    # Freeze the base model layers
    for layer in base_model.layers:
        layer.trainable = False
        
    return model

def prepare_data():
    """Prepare data generators for training and validation"""
    # Data augmentation for training
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest'
    )
    
    # Only rescaling for validation
    valid_datagen = ImageDataGenerator(rescale=1./255)
    
    # Base directory for the dataset
    base_dir = Path('E:\hasan\final year project mango enhancement version\dataset')
    
    # Create generators
    train_generator = train_datagen.flow_from_directory(
        base_dir / 'train',
        target_size=(224, 224),
        batch_size=32,
        class_mode='categorical'
    )
    
    validation_generator = valid_datagen.flow_from_directory(
        base_dir / 'validation',
        target_size=(224, 224),
        batch_size=32,
        class_mode='categorical'
    )
    
    return train_generator, validation_generator

def train_model():
    """Train the model with support for resuming"""
    # Create model directory if it doesn't exist
    model_dir = Path('../models')
    model_dir.mkdir(exist_ok=True)
    
    checkpoint_dir = model_dir / 'checkpoints'
    checkpoint_dir.mkdir(exist_ok=True)
    
    model_path = model_dir / 'best_model.keras'
    
    # Create or load model
    try:
        model = load_model(model_path)
    except ValueError:
        print("Model loading failed. Creating a new model.")
        model = create_model()
        
    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    # Prepare data
    train_generator, validation_generator = prepare_data()
    
    # Calculate steps per epoch
    steps_per_epoch = train_generator.samples // train_generator.batch_size
    validation_steps = validation_generator.samples // validation_generator.batch_size
    
    # Create checkpoint callback
    training_checkpoint = TrainingCheckpoint(checkpoint_dir)
    
    # Adjust hyperparameters and callbacks
    model_checkpoint = ModelCheckpoint('models/best_model.keras', save_best_only=True, monitor='val_accuracy', mode='max')
    early_stopping = EarlyStopping(monitor='val_loss', patience=5)

    # Train the model with the updated settings
    model.fit(
        train_generator,
        epochs=50,
        validation_data=validation_generator,
        callbacks=[training_checkpoint, model_checkpoint, early_stopping]
    )
    
    # Evaluate the model after training
    val_loss, val_accuracy = model.evaluate(validation_generator)
    print(f"Validation Accuracy: {val_accuracy * 100:.2f}%")
    print(f"Validation Loss: {val_loss:.4f}")
    
    # Evaluate the model after training
    train_loss, train_accuracy = model.evaluate(train_generator)
    print(f"Training Accuracy: {train_accuracy * 100:.2f}%")
    print(f"Training Loss: {train_loss:.4f}")
    
    # Step 1: Evaluate the Model
    validation_data = validation_generator
    predictions = model.predict(validation_data)
    evaluation_metrics = model.evaluate(validation_data)
    
    # Step 2: Save the Model
    model.save('models/best_model.keras')
    
    # Step 3: Fine-tune the Model
    # Adjust hyperparameters or retrain if necessary
    # Example: model.fit(..., epochs=...)
    # Unfreeze the base model layers
    for layer in model.layers:
        layer.trainable = True
        
    model.compile(
        optimizer=Adam(learning_rate=0.0001),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    # Fine-tune the model
    model.fit(
        train_generator,
        epochs=10,
        validation_data=validation_generator,
        callbacks=[training_checkpoint, early_stopping, model_checkpoint]
    )
    
    return model

if __name__ == "__main__":
    print("Starting model training...")
    model = train_model()
