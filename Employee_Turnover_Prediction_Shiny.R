###--------------------------------------------------------------------------###
###          COMM 301 | Machine Learning for Communication Management        ###
###                    Take Home Assignment | Group 6                        ###
###                         Shiny App Component                              ###
###--------------------------------------------------------------------------###

#### Setting Discrepancies ####
install.packages(c("shiny",
                   "shinydashboard")
)

pacman::p_load(shiny, shinydashboard)

#### QUESTION 8 ----

skim(turnover_cleaned)

#### MODEL ----
FINAL_MODEL %>% 
  saveRDS("algo_turnover_Group6_Final.RDS")

MODEL <- readRDS("algo_turnover_Group6_Final.RDS")
MODEL

#### UI ----

ui <- 
  dashboardPage(
    
    dashboardHeader(title = "Employee Turnover Prediction App"),
    
    dashboardSidebar(
      menuItem("Employee Turnover Prediction App",
               tabName = "turnover_tab")
    ),
    
    dashboardBody(
      # Step 1
      tabItem(tabName = "attribute_tab",
              # # 1st input box 
              selectInput(inputId = "Education",
                          label = "Education",
                          choices = c("Bachelors",
                                      "Masters",
                                      "PHD")
              ),
              # # Second Input Box
              selectInput(inputId = "City",
                          label = "City",
                          choices = c("Singapore",
                                      "Seoul",
                                      "Hong Kong")
              ),
              # # Third Input Box
              selectInput(inputId = "PaymentTier",
                          label = "Payment Tier",
                          choices = c("1",
                                      "2",
                                      "3")
              ),
              # # Fourth Input Box
              selectInput(inputId = "Gender",
                          label = "Gender of Employee",
                          choices = c("Male",
                                      "Female")
              ),
              # # Fifth Input Box
              selectInput(inputId = "EverBenched",
                          label = "Was Employee Kept out of Project > 1 Month",
                          choices = c("Yes",
                                      "No")
              ),
              # Sixth input box    
              box(sliderInput(inputId = "JoiningYear",
                              label = "Joining Year",
                              value = 2015, #default value,
                              min = 2012,
                              max = 2018,
                              sep = "",
                              step = 1)
              ),
              # Seventh input box
              box(sliderInput(inputId = "Age",
                              label = "Employee Age",
                              value = 29,
                              min = 22,
                              max = 41)
              ),
              # Eighth input Box
              box(sliderInput(inputId = "ExperienceInCurrentDomain",
                              label = "Experience in Current Domain",
                              value = 3,
                              min = 0,
                              max = 7)
              )
      ),
      
      # Step 4
      tabItem(tabName = "prediction_tab",
              box(valueBoxOutput("turnover_prediction")
              )
      )
    )
  )

#### SERVER ----

server <- 
  function(input, output) 
  { 
    output$turnover_prediction <- 
      # Step 2
      renderValueBox(
        {
          predicted_values <- 
            MODEL %>% 
            predict(tibble("JoiningYear" = input$JoiningYear,
                           "Age" = input$Age,
                           "Education" = input$Education,
                           "City" = input$City,
                           "PaymentTier" = input$PaymentTier,
                           "Gender" = input$Gender,
                           "EverBenched" = input$EverBenched,
                           "ExperienceInCurrentDomain" = input$ExperienceInCurrentDomain
            )
            ) %>% 
            select(.pred_class)
          # Step 3    
          valueBox(value = paste0(predicted_values$.pred_class),
                   subtitle = "Should Retention Package be Provided?"
          )
        }
      )     
  }

#### Run Shiny App ----

shinyapp_turnover <-
  shinyApp(ui, server)

shinyapp_turnover

# Thank you for taking the time and effort to look through our Shiny App :)


