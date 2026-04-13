import kagglehub
import os
import shutil

def setup_data():
    # Télécharge la version la plus populaire du dataset Lending Club
    path = kagglehub.dataset_download("wordsforthewise/lending-club")
    
    # On déplace le fichier dans notre dossier local 'data' pour plus de clarté
    local_data_dir = "data"
    os.makedirs(local_data_dir, exist_ok=True)
    
    for file in os.listdir(path):
        if file.endswith(".csv"):
            shutil.move(os.path.join(path, file), os.path.join(local_data_dir, file))
            print(f"Fichier déplacé : {file}")

if __name__ == "__main__":
    setup_data()