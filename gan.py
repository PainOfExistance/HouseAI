import tensorflow as tf
from tensorflow.keras import layers, models, optimizers
import os
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import save_img

IMG_HEIGHT = 64
IMG_WIDTH = 64
BATCH_SIZE = 32
NOISE_DIM = 100
EPOCHS = 50  # DCGAN training epochs
CLASSIFIER_EPOCHS = 20  # Classifier training epochs
AUGMENT_MULTIPLIER = 0.5  # Generate synthetic images to add ~50% extra samples

# Paths
data_dir = "Data/train"
save_dir = "generated_images"
os.makedirs(save_dir, exist_ok=True)

# -------------------------------
# 1. Prepare Data for DCGAN Training
# -------------------------------
train_ds = tf.keras.preprocessing.image_dataset_from_directory(
    data_dir,
    labels='inferred',
    label_mode='categorical',
    batch_size=BATCH_SIZE,
    image_size=(IMG_HEIGHT, IMG_WIDTH),
    shuffle=True
)

def normalize_img(image, label):
    # Normalize to [-1, 1] for DCGAN
    image = (tf.cast(image, tf.float32) - 127.5) / 127.5
    return image, label

train_ds = train_ds.map(normalize_img)

# For the GAN we only care about images:
def extract_images(image, label):
    return image

dcgan_dataset = train_ds.map(lambda img, label: extract_images(img, label))
dcgan_dataset = dcgan_dataset.prefetch(buffer_size=tf.data.AUTOTUNE)

# -------------------------------
# 2. Build and Train DCGAN
# -------------------------------
def make_generator_model():
    model = models.Sequential(name="generator")
    model.add(layers.Dense(8*8*256, use_bias=False, input_shape=(NOISE_DIM,)))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())
    model.add(layers.Reshape((8, 8, 256)))  # 8x8x256

    model.add(layers.Conv2DTranspose(128, (5, 5), strides=(2, 2), padding='same', use_bias=False))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())
    # Now: 16x16x128

    model.add(layers.Conv2DTranspose(64, (5, 5), strides=(2, 2), padding='same', use_bias=False))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())
    # Now: 32x32x64

    model.add(layers.Conv2DTranspose(3, (5, 5), strides=(2, 2), padding='same', use_bias=False, activation='tanh'))
    # Output: 64x64x3
    return model

def make_discriminator_model():
    model = models.Sequential(name="discriminator")
    model.add(layers.Conv2D(64, (5, 5), strides=(2, 2), padding='same', input_shape=[IMG_HEIGHT, IMG_WIDTH, 3]))
    model.add(layers.LeakyReLU())
    model.add(layers.Dropout(0.3))
    
    model.add(layers.Conv2D(128, (5, 5), strides=(2, 2), padding='same'))
    model.add(layers.LeakyReLU())
    model.add(layers.Dropout(0.3))
    
    model.add(layers.Flatten())
    model.add(layers.Dense(1))
    return model

generator = make_generator_model()
discriminator = make_discriminator_model()

cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)

def discriminator_loss(real_output, fake_output):
    real_loss = cross_entropy(tf.ones_like(real_output), real_output)
    fake_loss = cross_entropy(tf.zeros_like(fake_output), fake_output)
    return real_loss + fake_loss

def generator_loss(fake_output):
    return cross_entropy(tf.ones_like(fake_output), fake_output)

generator_optimizer = optimizers.Adam(1e-4)
discriminator_optimizer = optimizers.Adam(1e-4)

@tf.function
def train_step(images):
    noise = tf.random.normal([BATCH_SIZE, NOISE_DIM])
    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
        generated_images = generator(noise, training=True)
        
        real_output = discriminator(images, training=True)
        fake_output = discriminator(generated_images, training=True)
        
        gen_loss = generator_loss(fake_output)
        disc_loss = discriminator_loss(real_output, fake_output)
        
    gradients_of_generator = gen_tape.gradient(gen_loss, generator.trainable_variables)
    gradients_of_discriminator = disc_tape.gradient(disc_loss, discriminator.trainable_variables)
    
    generator_optimizer.apply_gradients(zip(gradients_of_generator, generator.trainable_variables))
    discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, discriminator.trainable_variables))
    
    return gen_loss, disc_loss

def train_gan(dataset, epochs):
    for epoch in range(epochs):
        print(f"Starting epoch {epoch+1}/{epochs}")
        for image_batch in dataset:
            g_loss, d_loss = train_step(image_batch)
        if (epoch + 1) % 10 == 0:
            noise = tf.random.normal([16, NOISE_DIM])
            generated_images = generator(noise, training=False)
            fig = plt.figure(figsize=(4, 4))
            for i in range(generated_images.shape[0]):
                plt.subplot(4, 4, i+1)
                plt.imshow((generated_images[i] * 127.5 + 127.5).numpy().astype("uint8"))
                plt.axis('off')
            plt.show()
    print("DCGAN training complete.")

train_gan(dcgan_dataset, EPOCHS)

# Function to generate synthetic images
def generate_synthetic_images(num_images):
    noise = tf.random.normal([num_images, NOISE_DIM])
    synthetic_images = generator(noise, training=False)
    # Rescale images to [0, 255]
    synthetic_images = (synthetic_images * 127.5 + 127.5)
    return synthetic_images

# Generate synthetic images
total_train_images = sum([len(tf.io.gfile.listdir(os.path.join("Data/train", folder))) 
                          for folder in os.listdir("Data/train")])
num_synthetic = int(total_train_images * AUGMENT_MULTIPLIER)
synthetic_images = generate_synthetic_images(num_synthetic)

# Save generated images to filesystem
def generate_and_save_synthetic_images(num_images, save_directory):
    noise = tf.random.normal([num_images, NOISE_DIM])
    synthetic_images = generator(noise, training=False)
    synthetic_images = (synthetic_images * 127.5 + 127.5)
    for i in range(num_images):
        img = tf.clip_by_value(synthetic_images[i], 0, 255)
        img = img.numpy().astype("uint8")
        file_path = os.path.join(save_directory, f"synthetic_{i}.png")
        save_img(file_path, img)
    print(f"Saved {num_images} synthetic images to '{save_directory}'")

# Example: Save 100 synthetic images
num_synthetic_images = 100
generate_and_save_synthetic_images(num_synthetic_images, save_dir)

# -------------------------------
# 3. Prepare Data for Classification
# -------------------------------
def normalize_classification(image, label):
    image = tf.cast(image, tf.float32) / 255.0  # Normalize to [0, 1]
    return image, label

# Original training, validation, and test datasets (using real images)
classification_train_ds = tf.keras.preprocessing.image_dataset_from_directory(
    "Data/train",
    labels='inferred',
    label_mode='categorical',
    batch_size=BATCH_SIZE,
    image_size=(IMG_HEIGHT, IMG_WIDTH),
    shuffle=True
).map(normalize_classification)

classification_valid_ds = tf.keras.preprocessing.image_dataset_from_directory(
    "Data/valid",
    labels='inferred',
    label_mode='categorical',
    batch_size=BATCH_SIZE,
    image_size=(IMG_HEIGHT, IMG_WIDTH),
    shuffle=True
).map(normalize_classification)

classification_test_ds = tf.keras.preprocessing.image_dataset_from_directory(
    "Data/test",
    labels='inferred',
    label_mode='categorical',
    batch_size=BATCH_SIZE,
    image_size=(IMG_HEIGHT, IMG_WIDTH),
    shuffle=False
).map(normalize_classification)

# -------------------------------
# 4. Build the Classification Model
# -------------------------------
def build_classifier_model():
    model = models.Sequential([
        layers.Input(shape=(IMG_HEIGHT, IMG_WIDTH, 3)),
        layers.Conv2D(32, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(128, (3, 3), activation='relu'),
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dense(4, activation='softmax')  # Four classes
    ])
    return model

# -------------------------------
# 5. Train and Evaluate Classifier WITHOUT Augmentation
# -------------------------------
classifier_no_aug = build_classifier_model()
classifier_no_aug.compile(optimizer='adam',
                          loss='categorical_crossentropy',
                          metrics=['accuracy'])
print("Training classifier without GAN augmentation:")
classifier_no_aug.fit(classification_train_ds,
                      validation_data=classification_valid_ds,
                      epochs=CLASSIFIER_EPOCHS)
test_loss_no_aug, test_acc_no_aug = classifier_no_aug.evaluate(classification_test_ds)
print(f"Test Accuracy without augmentation: {test_acc_no_aug:.2f}")

# -------------------------------
# 6. Create an Augmented Training Dataset (Real + Synthetic)
# -------------------------------
# IMPORTANT: Since the unconditional GAN does not produce labeled images,
# here we assume for demonstration that the synthetic images represent a specific cancer type.
# For example, assume they represent "adenocarcinoma". If your classes are in the order:
# [adenocarcinoma, large.cell.carcinoma, normal, squamous.cell.carcinoma],
# then "adenocarcinoma" corresponds to index 0.
synthetic_label = tf.one_hot(0, depth=4)  # One-hot label for adenocarcinoma
synthetic_labels = tf.repeat(tf.expand_dims(synthetic_label, 0), num_synthetic, axis=0)

# Convert synthetic images (scaled to [0, 255]) to [0, 1] and create a dataset
synthetic_images_norm = tf.cast(synthetic_images, tf.float32) / 255.0
synthetic_ds = tf.data.Dataset.from_tensor_slices((synthetic_images_norm, synthetic_labels))
synthetic_ds = synthetic_ds.batch(BATCH_SIZE)

# Combine the real training dataset with the synthetic dataset
augmented_train_ds = classification_train_ds.concatenate(synthetic_ds)
# Shuffle the combined dataset for training
augmented_train_ds = augmented_train_ds.shuffle(buffer_size=1000)

# -------------------------------
# 7. Train and Evaluate Classifier WITH Augmentation
# -------------------------------
classifier_with_aug = build_classifier_model()
classifier_with_aug.compile(optimizer='adam',
                            loss='categorical_crossentropy',
                            metrics=['accuracy'])
print("Training classifier with GAN augmentation:")
classifier_with_aug.fit(augmented_train_ds,
                        validation_data=classification_valid_ds,
                        epochs=CLASSIFIER_EPOCHS)
test_loss_with_aug, test_acc_with_aug = classifier_with_aug.evaluate(classification_test_ds)
print(f"Test Accuracy with augmentation: {test_acc_with_aug:.2f}")

# -------------------------------
# 8. Compare the Results
# -------------------------------
print("\nSummary of Classification Results:")
print(f"Accuracy without augmentation: {test_acc_no_aug:.2f}")
print(f"Accuracy with GAN augmentation: {test_acc_with_aug:.2f}")

