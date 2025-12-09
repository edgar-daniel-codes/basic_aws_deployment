# Imagen con requerimientos para Shiny apps 
FROM rocker/shiny:latest

# Trabajamos en el default de Shiny Server 
WORKDIR /srv/shiny-server

# Hacemos una copia de la app donde se espera por default 
COPY app.R /srv/shiny-server/app.R

# Exponemos el puerto 3838
EXPOSE 3838

# Se corre la app por Default en la imagen
# No hay necesidad de sobre escribir CMD 
