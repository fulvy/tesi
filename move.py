import os
import shutil

#%% MOVE .mat FILE IN ANOTHER DIRCTORY
source_dir = r"C:\Users\fulvi\DataspellProjects\tesi\features_mat"
destination_dir = r"C:\Users\fulvi\DataspellProjects\tesi\mat"

if not os.path.exists(destination_dir):
    os.makedirs(destination_dir)

for file_name in os.listdir(source_dir):
    if file_name.endswith('.mat'):
        # Costruzione del percorso completo dei file di origine e destinazione
        source_file = os.path.join(source_dir, file_name)
        destination_file = os.path.join(destination_dir, file_name)
        # Spostamento del file
        shutil.move(source_file, destination_file)

print("Spostamento completato con successo.")


#%% CREATE Probe AND Gallery FOLDER

# Definire i percorsi delle cartelle
base_path = r'C:\Users\fulvi\DataspellProjects\tesi'
features_mat_path = os.path.join(base_path, 'features_mat')
gallery_path = os.path.join(base_path, 'gallery')
probe_path = os.path.join(base_path, 'probe')

# Creare le cartelle 'gallery' e 'probe' se non esistono
os.makedirs(gallery_path, exist_ok=True)
os.makedirs(probe_path, exist_ok=True)

# Ottenere la lista di tutti i file nella cartella 'features_mat'
file_list = sorted(os.listdir(features_mat_path))

# Variabili per contare i file copiati
gallery_count = 0
probe_count = 0

# Iterare sui file in blocchi di 20
for i in range(0, len(file_list), 20):
    gallery_files = file_list[i:i+10]
    probe_files = file_list[i+10:i+20]

    # Copiare i primi 10 file nella cartella 'gallery'
    for file_name in gallery_files:
        src_path = os.path.join(features_mat_path, file_name)
        dest_path = os.path.join(gallery_path, file_name)
        shutil.copyfile(src_path, dest_path)
        gallery_count += 1

    # Copiare i successivi 10 file nella cartella 'probe'
    for file_name in probe_files:
        src_path = os.path.join(features_mat_path, file_name)
        dest_path = os.path.join(probe_path, file_name)
        shutil.copyfile(src_path, dest_path)
        probe_count += 1

print(f"Files copied to gallery: {gallery_count}")
print(f"Files copied to probe: {probe_count}")
