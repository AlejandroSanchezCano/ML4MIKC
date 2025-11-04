# Databases
python src/databases/find_MIKC.py
python src/databases/add_uniprot_data.py
python src/databases/add_interpro_domains.py
python src/databases/add_closest_arabidopsis.py
sh job.sh -u gpu_a100 -t 00:25:00 -f src/databases/add_embeddings.py