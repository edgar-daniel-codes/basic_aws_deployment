# app.R

### MCD ITAM - Otoño 2025
### Análisis de Datos Espaciales 
### Examen Final 
### Edgar Daniel Diaz Nava (174415)

#  En el presente scrip se define y configura un tablero interactivo a partir 
#  de los datos limpios usados como base de entrenamiento y almacenados en 
#  formato feather. La intención es generar un dashboard en una shiny app. 


### ----------------------------------------------------------------------------
### Importamos las librerias necesarias 
library(shiny)
library(shinydashboard)
library(dplyr)
library(readr)
library(ggplot2)

# Parametros 
CLEAN_DATA <- "./data/clean/"
FILE_NAME  <- "auto_mpg.csv"

# Cargar datos 
df <- read_csv(paste0(CLEAN_DATA, FILE_NAME))

### ----------------------------------------------------------------------------
### UI
ui <- dashboardPage(
  dashboardHeader(title = "Auto MPG "),
  
  dashboardSidebar(
    sidebarMenu(
      menuItem("Matriz ", tabName = "matrix", icon = icon("braille")),
      menuItem("Distribuciones", tabName = "dist", icon = icon("chart-bar"))
    ),
    br(),
    # Filtros globales
    selectInput(
      "origin_filter",
      "Filtrar por origen:",
      choices  = sort(unique(df$origin)),
      selected = unique(df$origin),
      multiple = TRUE
    ),
    sliderInput(
      "year_filter",
      "Rango de año del modelo:",
      min   = min(df$model_year, na.rm = TRUE),
      max   = max(df$model_year, na.rm = TRUE),
      value = c(min(df$model_year, na.rm = TRUE),
                max(df$model_year, na.rm = TRUE)),
      sep = ""
    ),
    sliderInput(
      "cyl_filter",
      "Rango de cilindros:",
      min   = min(df$cylinders, na.rm = TRUE),
      max   = max(df$cylinders, na.rm = TRUE),
      value = c(min(df$cylinders, na.rm = TRUE),
                max(df$cylinders, na.rm = TRUE)),
      step  = 1
    )
  ),
  
  dashboardBody(
    # KPIs superiores tipo 
    fluidRow(
      valueBoxOutput("kpi_mpg", width = 4),
      valueBoxOutput("kpi_hp",  width = 4),
      valueBoxOutput("kpi_wt",  width = 4)
    ),
    
    tabItems(
      # Tab de matriz estilo BCG 
      tabItem(
        tabName = "matrix",
        fluidRow(
          box(
            width = 12,
            title = "Matriz de importancia: MPG vs Horsepower",
            status = "primary",
            solidHeader = TRUE,
            collapsible = TRUE,
            p("• Eje X: Potencia (horsepower)",
              br(),
              "• Eje Y: Eficiencia (mpg)",
              br(),
              "• Líneas punteadas: medianas de horsepower y mpg separando cuadrantes."),
            plotOutput("bcgPlot", height = "500px")
          )
        )
      ),
      
      # Tab de distribuciones 
      tabItem(
        tabName = "dist",
        fluidRow(
          box(
            width = 6,
            title = "Distribución de MPG",
            status = "info",
            solidHeader = TRUE,
            plotOutput("mpgHist", height = "350px")
          ),
          box(
            width = 6,
            title = "MPG vs Peso",
            status = "info",
            solidHeader = TRUE,
            plotOutput("mpgWeight", height = "350px")
          )
        ),
        fluidRow(
          box(
            width = 12,
            title = "MPG vs Cilindros por origen",
            status = "info",
            solidHeader = TRUE,
            plotOutput("mpgCylinders", height = "350px")
          )
        )
      )
    )
  )
)

### ----------------------------------------------------------------------------
### Server: configuracion
server <- function(input, output, session) {
  
  # Datos filtrados en función de los controles
  data_filtered <- reactive({
    df %>%
      filter(
        origin %in% input$origin_filter,
        model_year >= input$year_filter[1],
        model_year <= input$year_filter[2],
        cylinders  >= input$cyl_filter[1],
        cylinders  <= input$cyl_filter[2]
      )
  })
  
  # KPIs generales 
  output$kpi_mpg <- renderValueBox({
    d <- data_filtered()
    valueBox(
      value = round(mean(d$mpg, na.rm = TRUE), 1),
      subtitle = "MPG promedio (objetivo)",
      icon = icon("gauge-high"),
      color = "green"
    )
  })
  
  output$kpi_hp <- renderValueBox({
    d <- data_filtered()
    valueBox(
      value = round(mean(d$horsepower, na.rm = TRUE), 1),
      subtitle = "Horsepower promedio",
      icon = icon("car"),
      color = "yellow"
    )
  })
  
  output$kpi_wt <- renderValueBox({
    d <- data_filtered()
    valueBox(
      value = round(mean(d$weight, na.rm = TRUE), 0),
      subtitle = "Peso promedio",
      icon = icon("weight-hanging"),
      color = "aqua"
    )
  })
  
  # Plot estilo BCG 
  output$bcgPlot <- renderPlot({
    d <- data_filtered()
    req(nrow(d) > 0)
    
    x_med <- median(d$horsepower, na.rm = TRUE)
    y_med <- median(d$mpg,        na.rm = TRUE)
    
    ggplot(d, aes(x = horsepower, y = mpg, color = origin, size = weight)) +
      geom_point(alpha = 0.7) +
      geom_vline(xintercept = x_med, linetype = "dashed") +
      geom_hline(yintercept = y_med, linetype = "dashed") +
      annotate("text",
               x = x_med * 0.5,
               y = y_med * 1.05,
               label = "Alta eficiencia / Baja potencia",
               hjust = 0, vjust = -0.5, size = 3.5) +
      annotate("text",
               x = x_med * 1.1,
               y = y_med * 1.05,
               label = "Alta eficiencia / Alta potencia",
               hjust = 0, vjust = -0.5, size = 3.5) +
      annotate("text",
               x = x_med * 0.5,
               y = y_med * 0.9,
               label = "Baja eficiencia / Baja potencia",
               hjust = 0, vjust = 1.5, size = 3.5) +
      annotate("text",
               x = x_med * 1.1,
               y = y_med * 0.9,
               label = "Baja eficiencia / Alta potencia",
               hjust = 0, vjust = 1.5, size = 3.5) +
      scale_size_continuous(name = "Peso") +
      labs(
        x = "Horsepower",
        y = "MPG (objetivo)",
        color = "Origen",
        title = "Matriz estilo BCG: Eficiencia vs Potencia",
        subtitle = "Líneas punteadas: medianas de horsepower y mpg"
      ) +
      theme_minimal()
  })
  
  # Distribuciones de variables 
  output$mpgHist <- renderPlot({
    d <- data_filtered()
    req(nrow(d) > 0)
    
    ggplot(d, aes(x = mpg, fill = origin)) +
      geom_histogram(bins = 20, alpha = 0.7, position = "identity") +
      labs(
        x = "MPG",
        y = "Frecuencia",
        fill = "Origen"
      ) +
      theme_minimal()
  })
  
  output$mpgWeight <- renderPlot({
    d <- data_filtered()
    req(nrow(d) > 0)
    
    ggplot(d, aes(x = weight, y = mpg, color = origin)) +
      geom_point(alpha = 0.7) +
      labs(
        x = "Peso",
        y = "MPG",
        color = "Origen",
        title = "Relación MPG vs Peso"
      ) +
      theme_minimal()
  })
  
  output$mpgCylinders <- renderPlot({
    d <- data_filtered()
    req(nrow(d) > 0)
    
    ggplot(d, aes(x = factor(cylinders), y = mpg, fill = origin)) +
      geom_boxplot(alpha = 0.8) +
      labs(
        x = "Cilindros",
        y = "MPG",
        fill = "Origen",
        title = "MPG por # de cilindros y origen"
      ) +
      theme_minimal()
  })
}

shinyApp(ui, server)
