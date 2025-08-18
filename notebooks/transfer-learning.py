import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Define constants
train_dir = 'cats-v-dogs/train'
validation_dir = 'cats-v-dogs/validation'

# Image preprocessing and augmentation
train_datagen = ImageDataGenerator(
    rescale=1./255,         # Normalize the images
    rotation_range=40,      # Randomly rotate images
    width_shift_range=0.2,  # Randomly shift images horizontally
    height_shift_range=0.2, # Randomly shift images vertically
    shear_range=0.2,        # Randomly shear images
    zoom_range=0.2,         # Randomly zoom images
    horizontal_flip=True,   # Randomly flip images horizontally
    fill_mode='nearest'     # Fill empty pixels after transformations
)

validation_datagen = ImageDataGenerator(rescale=1./255)

# Load training and validation datasets
train_generator = train_datagen.flow_from_directory(
    train_dir,
    batch_size=32,
    class_mode='binary',
    target_size=(150, 150)  # Adjust the image size as required
)

validation_generator = validation_datagen.flow_from_directory(
    validation_dir,
    batch_size=32,
    class_mode='binary',
    target_size=(150, 150)
)

# Load pre-trained VGG16 model without the top classification layer
base_model = tf.keras.applications.VGG16(
    weights='imagenet',  # Load weights from ImageNet
    include_top=False,   # Exclude the fully connected layers
    input_shape=(150, 150, 3)  # Match image dimensions
)

# Freeze the base model layers
base_model.trainable = False

# Build the custom model
model = models.Sequential([
    base_model,
    layers.Flatten(),
    layers.Dense(256, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(1, activation='sigmoid')  # Binary classification (cats vs. dogs)
])

# Compile the model
model.compile(
    loss='binary_crossentropy',  # Binary classification
    optimizer=tf.keras.optimizers.Adam(),
    metrics=['accuracy']
)

# Train the model
history = model.fit(
    train_generator,
    epochs=10,
    validation_data=validation_generator
)

# Save the model (optional)
model.save('transfer_learning_model.h5')
