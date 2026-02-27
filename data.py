import os, random, shutil

# 🔹 Absolute path to your dataset folder
# Update this if your folder is in a different location
data_dir = r"C:\Users\srilakshmi\OneDrive\Desktop\CELL_IMAGES"

parasitized = os.path.join(data_dir, "Parasitized")
uninfected = os.path.join(data_dir, "Uninfected")

# 🔹 Output folder for subset
subset_dir = os.path.join(data_dir, "data_subset")
os.makedirs(os.path.join(subset_dir, "Parasitized"), exist_ok=True)
os.makedirs(os.path.join(subset_dir, "Uninfected"), exist_ok=True)

# 🔹 Select 499 parasitized + 499 uninfected
parasitized_files = random.sample(os.listdir(parasitized), 499)
uninfected_files = random.sample(os.listdir(uninfected), 499)

# 🔹 Copy files into subset folder
for f in parasitized_files:
    shutil.copy(os.path.join(parasitized, f),
                os.path.join(subset_dir, "Parasitized", f))

for f in uninfected_files:
    shutil.copy(os.path.join(uninfected, f),
                os.path.join(subset_dir, "Uninfected", f))

print("✅ Subset ready: 499 parasitized + 499 uninfected")
print("Saved in:", subset_dir)
