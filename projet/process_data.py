import polars as pl
import os

def convert_to_parquet():
    csv_path = "projet/data/accepted_2007_to_2018q4.csv/accepted_2007_to_2018Q4.csv"
    parquet_path = "projet/data/lending_club.parquet"

    if os.path.exists(csv_path):
        print("Conversion CSV -> Parquet en cours (utilisation de Polars)...")
        df = pl.scan_csv(csv_path, infer_schema_length=10000, low_memory=True, ignore_errors=True)
        df.sink_parquet(parquet_path)
        print(f"Terminé ! Fichier Parquet créé : {parquet_path}")
        
        # Optionnel : Supprimer le CSV pour gagner de la place
        # os.remove(csv_path)

if __name__ == "__main__":
    convert_to_parquet()