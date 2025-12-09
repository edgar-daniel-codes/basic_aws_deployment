# main.R 

### MCD ITAM - Otoño 2025
### Análisis de Datos Espaciales 
### Examen Final 
### Edgar Daniel Diaz Nava (174415)

# En el presente archivo se realiza un Analisis Exploratorio de Datos 
# para el conjunto de datos Auto MPG del repositorio UCI Machine Learning
# imprimiendo y guardando los resultados relevantes. 


### ----------------------------------------------------------------------------
### Librrías Necesarias  -------------------------------------------------------

library(tidyverse)
library(dplyr)
library(ggplot2)
library(GGally) 
library(arrow)


### ----------------------------------------------------------------------------
### PARAMETROS Y DATOS ---------------------------------------------------------


RAW_DATA <- "./data/raw/"
PREP_DATA <- "./data/prep/"
CLEAN_DATA <- "./data/clean/"

FILE_NAME <- "auto_mpg.csv"
FILE_FEATHER <- "auto_mpg.feather"

OUT_DIR <- './figures/'

# Se lee el archivo
data <- read.csv(paste0(CLEAN_DATA, FILE_NAME))

# Convertir 'origin' (1, 2, 3) a Factor
# 1: USA, 2: Europe, 3: Japan
data$origin <- factor(data$origin, 
                      levels = c(1, 2, 3), 
                      labels = c("USA", "Europe", "Japan"))


# Garantizamos que horsepower sea numerico 
data$horsepower <- as.numeric(data$horsepower)

# Imprimimos las variables
head(data)


### ----------------------------------------------------------------------------
### Analisis Exploratorio ------------------------------------------------------


### Calcular estadísticas univariadas

# Resumen rápido
summary(data)  

# Estadisticas particulares
num_data <- data[, sapply(data, is.numeric)]

summary_stats <- data.frame(
  Mínimo = sapply(num_data, min, na.rm = TRUE),
  Máximo = sapply(num_data, max, na.rm = TRUE),
  Media = sapply(num_data, mean, na.rm = TRUE),
  Mediana = sapply(num_data, median, na.rm = TRUE),
  Desviación_Estándar = sapply(num_data, sd, na.rm = TRUE),
  Q1 = sapply(num_data, quantile, probs = 0.25, na.rm = TRUE),
  Q3 = sapply(num_data, quantile, probs = 0.75, na.rm = TRUE)
)


# Transponer para que cada variable sea una fila
summary_stats <- t(summary_stats)
print(summary_stats)

# Repetimos el análisis por factor 

for(f in c("USA", "Europe", "Japan")){
  
  data_aux <- data |> filter(origin == f)
  
  num_data <- data_aux[, sapply(data_aux, is.numeric)]

  summary_stats <- data.frame(
    Mínimo = sapply(num_data, min, na.rm = TRUE),
    Máximo = sapply(num_data, max, na.rm = TRUE),
    Media = sapply(num_data, mean, na.rm = TRUE),
    Mediana = sapply(num_data, median, na.rm = TRUE),
    Desviación_Estándar = sapply(num_data, sd, na.rm = TRUE),
    Q1 = sapply(num_data, quantile, probs = 0.25, na.rm = TRUE),
    Q3 = sapply(num_data, quantile, probs = 0.75, na.rm = TRUE)
  )
  
  summary_stats <- t(summary_stats)
  
  cat("\nEstadisticas controladas por factor origen ", f, ": ------------- \n")
  print(summary_stats)
}


### Realizar análisis bivariado

num_data <- data |> 
  select(where(is.numeric))


## Correlacion global 

# Calcular matriz de correlación
cor_global <- cor(num_data, use = "pairwise.complete.obs")

# Convertir a formato largo para heatmap
cor_df <- cor_global |>
  as.data.frame() |>
  tibble::rownames_to_column("var1") |>
  pivot_longer(
    cols = -var1,
    names_to = "var2",
    values_to = "corr"
  )

# Heatmap de correlaciones global
p_cor_global <- ggplot(cor_df, aes(x = var1, y = var2, fill = corr)) +
  geom_tile() +
  scale_fill_gradient2(
    low = "blue", mid = "white", high = "red",
    midpoint = 0
  ) +
  labs(
    title = "Matriz de correlación (global)",
    x = "Variable",
    y = "Variable",
    fill = "Correlación"
  ) +
  theme_minimal() +
  theme(axis.text.x = element_text(angle = 45, hjust = 1))

p_cor_global

# Guardar heatmap global
ggsave(
  filename = file.path(OUT_DIR, "correlacion_global.png"),
  plot     = p_cor_global,
  width    = 8,
  height   = 6,
  dpi      = 300
)


## Analisis Bivariado Global 

# Matriz de gráficos (pairs) para todas las numéricas
p_pairs_global <- ggpairs(num_data)

p_pairs_global

ggsave(
  filename = file.path(OUT_DIR, "bivariado_global.png"),
  plot     = p_pairs_global,
  width    = 10,
  height   = 10,
  dpi      = 300
)


## Analisis por origen 

# Matriz bivariada (ggpairs) coloreada por origin
p_pairs_origin <- ggpairs(
  data,
  columns = which(sapply(data, is.numeric)),  # solo columnas numéricas
  aes(colour = origin, alpha = 0.6)
)

p_pairs_origin

ggsave(
  filename = file.path(OUT_DIR, "bivariado_por_origin.png"),
  plot     = p_pairs_origin,
  width    = 10,
  height   = 10,
  dpi      = 300
)


## Correlaciones por origen 

# Lista de valores únicos de origin
orígenes <- sort(unique(data$origin))

# Bucle sobre cada origen para calcular y guardar su heatmap
for (ori in orígenes) {
  # Filtrar un origen
  df_ori <- data |>
    filter(origin == ori) |>
    select(where(is.numeric))
  
  # Si hay menos de 2 variables numéricas, saltar
  if (ncol(df_ori) < 2) next
  
  # Matriz de correlación para este origen
  cor_ori <- cor(df_ori, use = "pairwise.complete.obs")
  
  # Pasar a formato largo
  cor_ori_df <- cor_ori |>
    as.data.frame() |>
    tibble::rownames_to_column("var1") |>
    pivot_longer(
      cols = -var1,
      names_to = "var2",
      values_to = "corr"
    )
  
  # Heatmap para este origen
  p_cor_ori <- ggplot(cor_ori_df, aes(x = var1, y = var2, fill = corr)) +
    geom_tile() +
    scale_fill_gradient2(
      low = "blue", mid = "white", high = "red",
      midpoint = 0
    ) +
    labs(
      title = paste("Matriz de correlación - origin:", ori),
      x = "Variable",
      y = "Variable",
      fill = "Correlación"
    ) +
    theme_minimal() +
    theme(axis.text.x = element_text(angle = 45, hjust = 1))
  
  # Guardar por origen (nombre limpio en archivo)
  fname <- paste0("correlacion_origin_", as.character(ori), ".png")
  
  show(p_cor_ori)
  
  ggsave(
    filename = file.path(OUT_DIR, fname),
    plot     = p_cor_ori,
    width    = 8,
    height   = 6,
    dpi      = 300
  )
}

## Regresiones bivariadas con Eficiencia (mpg)

# Seleccionar variables numéricas excepto mpg
num_vars <- data |>
  select(where(is.numeric)) |>
  select(-mpg) |>
  names()

# Loop: mpg ~ variable_numérica
for (v in num_vars) {
  
  p <- ggplot(data, aes(x = .data[[v]], y = mpg)) +
    geom_point(alpha = 0.6) +
    geom_smooth(method = "lm", se = TRUE) +
    labs(
      title = paste("Relación:", v, "vs Eficiencia (mpg)"),
      x = v,
      y = "Eficiencia (mpg)"
    ) +
    theme_minimal()
  
  show(p)
  
  ggsave(
    filename = file.path(OUT_DIR, paste0("scatter_reg_mpg_vs_", v, ".png")),
    plot = p,
    width = 7,
    height = 5,
    dpi = 300
  )
}


### Gráficos

# Boxplot de Eficiencia (mpg) por Origen (origin)
ggplot(data, aes(x = origin, y = mpg)) +
  geom_boxplot() +
  labs(
    title = "Distribución de eficiencia (mpg) por origen",
    x = "Origen",
    y = "Eficiencia (mpg)"
  ) +
  theme_minimal()

# Scatterplot de Peso(weight) vs Eficiencia (mpg)
ggplot(data, aes(x = weight, y = mpg)) +
  geom_point(alpha = 0.6) +
  labs(
    title = "Relación entre peso y eficiencia",
    x = "Peso",
    y = "Eficiencia (mpg)"
  ) +
  theme_minimal()



### ----------------------------------------------------------------------------
### Guardamos Data Limpía ------------------------------------------------------

# Guardamos en formato feather 
write_feather(data, paste0(CLEAN_DATA, FILE_FEATHER))


