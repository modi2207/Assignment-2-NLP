corpus = []
with open('..//data//testing_data.txt', 'r', encoding='utf-8') as file:
    for line in file:
        corpus.append(line.strip().split(', '))  # Split by ', ' to separate words

corpus = [word for sentence in corpus for word in sentence]

word_counts = {}

for word in corpus:
    if word in word_counts:
        word_counts[word] += 1
    else:
        word_counts[word] = 1

print("len: ",len(word_counts))