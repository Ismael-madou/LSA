from sklearn.datasets import fetch_20newsgroups
import os

# Télécharger le dataset
dataset = fetch_20newsgroups(subset='all', remove=('headers', 'footers', 'quotes'))

base_dir = "corpus_20newsgroups_txt"
os.makedirs(base_dir, exist_ok=True)

for i, (text, target) in enumerate(zip(dataset.data, dataset.target)):
    class_name = dataset.target_names[target]
    class_dir = os.path.join(base_dir, class_name)
    os.makedirs(class_dir, exist_ok=True)

    filename = os.path.join(class_dir, f"doc_{i:05d}.txt")
    with open(filename, "w", encoding="utf-8") as f:
        f.write(text)

print("Corpus exporté avec succès.")