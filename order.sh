# Databases
python src/databases/find_MIKC.py
python src/databases/all_interpro_domains.py
python src/databases/add_uniprot_data.py
python src/databases/download_databases_PPIs.py

# Sources
src/sources/merge_sources.ipynb
src/sources/statistics_database.ipynb
python src/sources/add_kahip.py
src/sources/plot_database.ipynb
sh job.sh -u gpu_a100 -t 00:10:00 -f src/sources/add_embeddings.py
python src/sources/add_domains.py
python src/sources/add_structure.py
python src/sources/add_distance_map.py
sh job.sh -u rome -t 00:05:00 -f src/sources/add_interface_features.py --array 0-5724%128