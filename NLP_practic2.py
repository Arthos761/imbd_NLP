from collections import Counter

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset


# 1. ЗАГРУЗКА И ПРОСМОТР ДАННЫХ
full_df = pd.read_csv('IMDB Dataset.csv')
print(full_df.shape)
print(full_df.head())
print(full_df['sentiment'].value_counts()) # сколько positive и negative

# 2. БЕРЕМ 50 000 ОТЗЫВОВ(25 000 ПОЛОЖИТЕЛЬНЫХ; 25 000 НЕГАТИВНЫХ)
df_positive = full_df[full_df['sentiment'] == 'positive'].sample(25000, random_state=52)
df_negative = full_df[full_df['sentiment'] == 'negative'].sample(25000, random_state=52)

df = (
  pd.concat([df_positive, df_negative]).sample(frac=1, random_state=52).reset_index(drop=True) #объеденили и перемешали
     )

print(df.shape)

# 3. ПРЕВРАЩЕМ "sentiment" В ЧИСЛА
df['sentiment'] = df['sentiment'].map({'positive': 1, 'negative': 0})

 # Превращаем столбцы DataFrame в списки Python
texts = df['review'].tolist()
labels = df['sentiment'].tolist()

# 4. ТОКЕНИЗАЦИЯ.
def tokenize(text):
    return text.lower().split()[:100]

all_words = []
for text in texts:
    all_words.extend(tokenize(text))

 # Строим словарь: слово → индекс. Но с ограничением на 40 000 самых частых слов

word_counts = Counter(all_words) # Считаем, сколько раз каждое слово встречается в all_words.
vocab = {word: idx for idx, (word, _) in enumerate(word_counts.most_common(50000))} # Создаем сам словарь с 50000 частых слов
vocab['<UNK>'] = len(vocab) # Добавляем специальный токен <UNK> для незнакомых слов.

 # Превратим текст в индексы

def encode_text(text, vocab):
    return [vocab.get(word, vocab['<UNK>']) for word in tokenize(text)]

# 5. ПАДДИНГ(уравняем длины)
max_len = 100

def pad_sequence(seq, max_len=max_len):
    if len(seq) < max_len:
        return seq + [0] * (max_len - len(seq))
    return seq[:max_len]

encoded_texts = [encode_text(text, vocab) for text in texts]
padded_texts = [pad_sequence(seq, max_len) for seq in encoded_texts]

# 6. РАЗДЕЛЯЕМ НА ТРЕНИРОВОЧНЫЕ И ТЕСТОВЫЕ ВЫБОРКИ(train/test)
X_train, X_test, y_train, y_test = train_test_split(padded_texts,labels, test_size=0.2, random_state=52)

#Переведем в тензоры
X_train = torch.tensor(X_train)
X_test = torch.tensor(X_test)
y_train = torch.tensor(y_train, dtype=torch.float32).unsqueeze(-1)
y_test = torch.tensor(y_test, dtype=torch.float32).unsqueeze(-1)

# 7. DATA LOADER
train_dataset = TensorDataset(X_train, y_train)
test_dataset = TensorDataset(X_test, y_test)

#Разбивем на батчи по 64
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)



# 8. МОДЕЛЬ
class NLP(nn.Module):
    def __init__(self, vocab_size, embed_size, hidden_size, output_size):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_size) #индексы в векторы
        self.lstm = nn.LSTM(embed_size, hidden_size, batch_first=True) #обрабатываем послед-ть
        self.dropout = nn.Dropout(0.55) # это слой, который случайно выключает часть нейронов во время обучения.
        self.fc = nn.Linear(hidden_size, output_size) #классифициуем

    def forward(self, x):
        x = self.embedding(x)
        out, _ = self.lstm(x)
        out = out[:, -1, :]
        out = self.dropout(out)
        return self.fc(out)

model = NLP(len(vocab), embed_size=64, hidden_size=128, output_size=1)
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.01, weight_decay=1e-5)


# 9. ОБУЧЕНИЕ
epochs = 10
for epoch in range(epochs):
    model.train()
    running_loss = 0.0
    for X_batch, y_batch in train_loader:
        optimizer.zero_grad()
        y_pred = torch.sigmoid(model(X_batch))
        loss = criterion(y_pred, y_batch)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    print(f'Epoch {epoch + 1} | Loss: {running_loss / len(train_loader):.3f}')

# 10. Проверка
model.eval() # Переключает модель в режим тестирования.
correct = 0
total = 0
with torch.no_grad():
    for X_batch, y_batch in test_loader:
        y_pred = torch.sigmoid(model(X_batch))
        y_pred_class = (y_pred > 0.5).float()
        correct += (y_pred_class == y_batch).sum().item()
        total += y_batch.size(0)

print(f"Accuracy на тесте: {100 * correct / total:.2f}%")
















