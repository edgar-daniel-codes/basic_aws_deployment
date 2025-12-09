#!/usr/bin/env zsh

# MCD - ITAM
# Computational Foundations for DS 
# Edgar Daniel Diaz Nava - 174415


###############################################################################
### 1.- Descarga y Preparación de Archivos
###############################################################################

# Descargamos los archivos
wget -O ./data/raw/auto-mpg.data -q https://archive.ics.uci.edu/ml/machine-learning-databases/auto-mpg/auto-mpg.data


# Limpiamos los datos crudos 
cat ./data/raw/auto-mpg.data | tr -s ' ' | sed 's/\t/ /g' | sed 's/ /,/g' | grep -v "?" | sed 's/^,//g' > './data/clean/auto_mpg_noheader.csv'

# Creamos el header 
cat << EOF > ./data/clean/headers.csv
mpg,cylinders,displacement,horsepower,weight,acceleration,model_year,origin,car_name
EOF

# Creamos la base de datos 
cat ./data/clean/headers.csv ./data/clean/auto_mpg_noheader.csv > './data/clean/auto_mpg.csv'


