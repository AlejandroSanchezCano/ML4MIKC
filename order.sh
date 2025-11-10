# MIKC
python src/families/MIKC/find_MIKC.py
python src/families/MIKC/add_uniprot_data.py
python src/families/MIKC/add_interpro_domains.py
python src/families/MIKC/add_closest_arabidopsis.py
sh job.sh -u gpu_a100 -t 00:25:00 -f src/families/MIKC/add_embeddings.py
python src/families/MIKC/plot_embedding.py

# M
python src/families/M/find_M.py
python src/families/M/add_uniprot_data.py
sh job.sh -u gpu_a100 -t 00:25:00 -f src/families/M/add_embeddings.py