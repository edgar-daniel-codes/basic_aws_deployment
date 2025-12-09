# Imagen con requerimientos para Shiny apps 
FROM rocker/shiny:latest

# Instalamos dependencias 
RUN R -e "install.packages(c('shinydashboard', 'dplyr', 'readr'), repos = 'https://cloud.r-project.org')"

# Trabajamos en el default de Shiny Server 
WORKDIR /srv/shiny-server

# Hacemos una copia de la app donde se espera por default 
COPY app.R /srv/shiny-server/app.R

# Exponemos el puerto 3838
EXPOSE 3838

# Se corre la app por Default en la imagen
# No hay necesidad de sobre escribir CMD 
