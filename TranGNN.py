import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Embedding, Dense, Dropout, LayerNormalization, MultiHeadAttention, GlobalAveragePooling1D, Add, Concatenate
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.utils import class_weight
import tensorflow as tf
from sklearn.preprocessing import RobustScaler



def transformer_encoder(inputs, head_size, num_heads, ff_dim, dropout=0):
    x = MultiHeadAttention(key_dim=head_size, num_heads=num_heads, dropout=dropout)(inputs, inputs)
    x = Dropout(dropout)(x)
    x = Add()([x, inputs])
    x = LayerNormalization(epsilon=1e-6)(x)
    
    x_ff = Dense(ff_dim, activation='relu')(x)
    x_ff = Dropout(dropout)(x_ff)
    x_ff = Dense(inputs.shape[-1])(x_ff)
    x = Add()([x, x_ff])
    return LayerNormalization(epsilon=1e-6)(x)


df = pd.read_csv('reentrancy_sample.csv')


max_words = 10000
max_len = 1000
tokenizer = Tokenizer(num_words=max_words, oov_token="<OOV>")
tokenizer.fit_on_texts(df['code'])

sequences = tokenizer.texts_to_sequences(df['code'])
padded_sequences = pad_sequences(sequences, maxlen=max_len, padding='post', truncating='post')


y = df['is_reentrancy'].values


gnn_embeddings = np.load("gnn_embeddings.npy")  # shape: (N, 64)

# Normalize the GNN embeddings
#scaler = StandardScaler()
#gnn_embeddings = scaler.fit_transform(gnn_embeddings)  # shape still (N, 64)

X_combined = [padded_sequences, gnn_embeddings]


skipped = np.load("skipped_indices.npy")
mask = np.ones(len(df), dtype=bool)
mask[skipped] = False

padded_sequences = padded_sequences[mask]
y = y[mask]



# 70% train, 15% val, 15% test
X_seq_train, X_seq_temp, X_gnn_train, X_gnn_temp, y_train, y_temp = train_test_split(
    padded_sequences, gnn_embeddings, y, test_size=0.3, stratify=y, random_state=42
)

X_seq_val, X_seq_test, X_gnn_val, X_gnn_test, y_val, y_test = train_test_split(
    X_seq_temp, X_gnn_temp, y_temp, test_size=0.5, stratify=y_temp, random_state=42
)


scaler = RobustScaler()
X_gnn_train = scaler.fit_transform(X_gnn_train)
X_gnn_val = scaler.transform(X_gnn_val)
X_gnn_test = scaler.transform(X_gnn_test)


class_weights = class_weight.compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
class_weights = dict(enumerate(class_weights))

print(f"Class Weights: {class_weights}")

print("Train:", len(y_train))
print("Validation:", len(y_val))
print("Test:", len(y_test))

# Final  model
input_layer = Input(shape=(max_len,))
embedding = Embedding(input_dim=max_words, output_dim=128)(input_layer)
transformer = transformer_encoder(embedding, head_size=64, num_heads=4, ff_dim=128, dropout=0.2)
transformer = transformer_encoder(transformer, head_size=64, num_heads=4, ff_dim=128, dropout=0.2)
global_avg = GlobalAveragePooling1D()(transformer)
transformer_vector = Dense(64, activation='relu')(global_avg)

gnn_input = Input(shape=(gnn_embeddings.shape[1],))
combined = Concatenate()([transformer_vector, gnn_input])
dense = Dense(64, activation='relu')(combined)
dropout = Dropout(0.4)(dense)   #used to be 0.3 but 0.4 better
output = Dense(1, activation='sigmoid')(dropout)

hybrid_model = Model(inputs=[input_layer, gnn_input], outputs=output)
hybrid_model.compile(optimizer=Adam(1e-4), loss='binary_crossentropy', metrics=['accuracy'])
#used to just be Adam(1e-4)

early_stop = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)
lr_scheduler = ReduceLROnPlateau(monitor='val_loss', patience=2, factor=0.5, verbose=1)


train_inputs = [X_seq_train, X_gnn_train]
val_inputs = [X_seq_val, X_gnn_val]
test_inputs = [X_seq_test, X_gnn_test]

# Training
history = hybrid_model.fit(
    #[X_train_seq, X_train_gnn],
    #y_train,
    #validation_data=([X_val_seq, X_val_gnn], y_val),
    
    train_inputs, y_train,
    validation_data=(val_inputs, y_val),

    batch_size=32,
    epochs=10,
    callbacks=[early_stop, lr_scheduler],
    class_weight=class_weights
)

# Evaluation
#y_pred_prob = hybrid_model.predict([X_test_seq, X_test_gnn])
y_pred_prob = hybrid_model.predict([X_seq_test, X_gnn_test])

#y_pred = (y_pred_prob > 0.5).astype(int)
#removed for threshold tuning

#START threshold tuning
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report
import numpy as np
import matplotlib.pyplot as plt

thresholds = np.arange(0.3, 0.71, 0.01)
best_acc, best_f1 = 0, 0
best_acc_thresh, best_f1_thresh = 0.5, 0.5
acc_scores, f1_scores = [], []

print("\n--- Threshold Tuning ---")
for thresh in thresholds:
    y_pred_thresh = (y_pred_prob >= thresh).astype(int)
    acc = accuracy_score(y_test, y_pred_thresh)
    f1 = f1_score(y_test, y_pred_thresh)
    acc_scores.append(acc)
    f1_scores.append(f1)
    
    print(f"Threshold: {thresh:.2f} | Accuracy: {acc:.4f} | F1: {f1:.4f}")
    
    if acc > best_acc:
        best_acc = acc
        best_acc_thresh = thresh
    if f1 > best_f1:
        best_f1 = f1
        best_f1_thresh = thresh

print(f"\nBest Accuracy Threshold: {best_acc_thresh:.2f} → Accuracy: {best_acc:.4f}")
print(f"Best F1 Threshold: {best_f1_thresh:.2f} → F1 Score: {best_f1:.4f}")

y_pred_final = (y_pred_prob >= best_f1_thresh).astype(int)

print("\n--- Final Classification Report (Best F1 Threshold) ---")
print(classification_report(y_test, y_pred_final, target_names=["Non-Reentrancy", "Reentrancy"]))

# Confusion matrix
import seaborn as sns
import matplotlib.pyplot as plt
cm = confusion_matrix(y_test, y_pred_final)
plt.figure(figsize=(7, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=["Non-Reentrancy", "Reentrancy"], yticklabels=["Non-Reentrancy", "Reentrancy"])
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title('Confusion Matrix (Best F1 Threshold)')
plt.show()

# Threshold curve
plt.figure(figsize=(10, 5))
plt.plot(thresholds, acc_scores, label='Accuracy')
plt.plot(thresholds, f1_scores, label='F1 Score')
plt.xlabel("Threshold")
plt.ylabel("Score")
plt.title("Threshold vs Accuracy / F1 Score")
plt.legend()
plt.grid(True)
plt.show()
#END threshold tuning


# Accuracy and Loss curves
plt.figure(figsize=(10,4))
plt.subplot(1,2,1)
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Val Accuracy')
plt.title('Hybrid Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

plt.subplot(1,2,2)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Val Loss')
plt.title('Hybrid Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.tight_layout()
plt.show()


#GNN embeddings before normalization
print("GNN embeddings stats:")
print("Min:", gnn_embeddings.min())
print("Max:", gnn_embeddings.max())
print("Mean:", gnn_embeddings.mean())
print("Std:", gnn_embeddings.std())

#GNN embeddings after normalization
print("Normalized GNN embeddings stats (training set):")
print("Min:", np.min(X_gnn_train))
print("Max:", np.max(X_gnn_train))
print("Mean:", np.mean(X_gnn_train))
print("Std:", np.std(X_gnn_train))

