###--------------------------------------------------------------------------###
###          COMM 301 | Machine Learning for Communication Management        ###
###                         Take Home Assignment                             ###
###                      Machine Learning Component                          ###
###--------------------------------------------------------------------------###

#### Setting Discrepancies ####

pacman::p_load(tidyverse, lubridate, # Tidy Data Science Practice
               tidymodels, # Tidy Machine Learning
               usemodels, ranger, doParallel, # xgboost, 
               skimr, GGally, Hmisc, broom, modelr, # EDA
               jtools, huxtable, interactions, # EDA
               ggfortify, ggstance, scales, gridExtra, ggthemes, # ggplot2:: add-ons
               DT, plotly, # Interactive Data Display
               janitor, # Data Wrangling
               factoextra, cluster, tidyclust, ggradar
)
sessionInfo()

load("Employee_Turnover_Prediction.RData")

#### DEFINE ####
# Predict a multinational company's employee turnover

#### IMPORT ----

turnover <-
  read.csv("https://talktoroh.squarespace.com/s/Employee_Turnover_MNC.csv")

skim(turnover)

#### QUESTION 1 ####

#### TRANSFORM ----

turnover_cleaned <-
  turnover %>% 
  mutate_if(is.character, as.factor) %>% 
  mutate(Turnover = ifelse(Turnover == "1", "Yes", "No")) %>% 
  mutate(Turnover = as.factor(Turnover),
         PaymentTier = as.factor(PaymentTier)) %>% 
  mutate(Turnover = fct_relevel(Turnover, "Yes"))

skim(turnover_cleaned)

# Correlation Matrix

### EDA PROCESS ####

turnover %>% 
  select(Turnover, # fct.
         where(is.numeric) # 
  ) %>% 
  mutate(Turnover = as.numeric(Turnover)
  ) %>% 
  as.matrix() %>% 
  rcorr() %>%  # Hmisc::
  tidy() %>% # broom:: # tidy() returns tbl_df
  mutate(absCORR = abs(estimate)
  ) %>% 
  dplyr::select(column1, column2, absCORR) %>% 
  datatable() %>% 
  formatRound(columns = "absCORR",
              digits = 3
  )
 
# Recipe Planning For EDA
 
recipe_EDA <-
   recipe(formula = Turnover ~ .,
          data = turnover_cleaned) %>% 
   step_normalize(all_numeric_predictors()
   ) %>% 
   step_dummy(all_nominal_predictors()
   )
 
 # Recipe Execution For EDA
 
baked_EDA <- 
   recipe_EDA %>% 
   prep(verbose = T) %>% 
   bake(new_data = turnover_cleaned)
 
baked_EDA %>% 
   skim()
 
# Visiualising Correlation
 
# Method 1: Data Table with absCORR

Corr_Table <-
  baked_EDA %>% 
  select(Turnover, # fct.
         where(is.numeric) # 
  ) %>% 
  mutate(Turnover = as.numeric(Turnover)
  ) %>% 
   as.matrix(.) %>% 
   rcorr(.) %>% 
   tidy(.) %>% 
   mutate(absCORR = abs(estimate)
   ) %>% 
   select(-n, -p.value, -estimate) %>% 
   datatable() %>% 
   formatRound(columns = "absCORR",
               digits = 3)

# Visualising Data

glimpse(turnover_cleaned)

plot_payment_tier <-
  turnover_cleaned %>% 
  ggplot(aes(fill = Turnover, 
             x = PaymentTier)) +
  geom_bar()

plot_city <-
  turnover_cleaned %>% 
  ggplot(aes(fill = Turnover, 
             x = City)) +
  geom_bar() 

plot_gender <-
  turnover_cleaned %>% 
  ggplot(aes(fill = Turnover, 
             x = Gender)) +
  geom_bar()
    
plot_joiningyear <-
  turnover_cleaned %>% 
  ggplot(aes(fill = Turnover, 
             x = as.factor(JoiningYear))) +
  geom_bar() 

compare_plots <-
  grid.arrange(plot_payment_tier, plot_city,
             plot_gender, plot_joiningyear)

# GGally

skim(turnover_cleaned)

GGally_plot <- 
  turnover_cleaned %>% 
  select(Turnover, PaymentTier,
         Gender, City,
         JoiningYear, ExperienceInCurrentDomain) %>% 
  ggpairs(mapping = aes(color = Turnover))

## Joining year seems to be an important feature. Employees are more
## likely to turnover if they joined in 2018 compared to other years
 
 #### SPLIT ----

skim(turnover_cleaned)

set.seed(100851)
turnover_split <-
  turnover_cleaned %>% 
  initial_split(prop = .75,
                strata = "Turnover")

turnover_train <- training(turnover_split)
turnover_test <- testing(turnover_split)

#### QUESTION 2 ####

#### Feature Engineering ----

skim(turnover_train)

# Recipe 1: Poly + Dummy, Joining Year
turnover_poly_dummy <-
  recipe(Turnover ~ .,
         data = turnover_train) %>% 
  step_dummy(all_nominal_predictors()) %>% 
  step_poly(JoiningYear,
            degree = 2,
            role = "predictor") 

# Recipe 2: Norm + Poly + Dummy + KNN, Joining Year
turnover_norm_poly_dummy_knn <-
  recipe(formula = Turnover ~ ., 
         data = turnover_train) %>% 
  step_impute_knn(PaymentTier) %>% 
  step_dummy(all_nominal_predictors()) %>% 
  step_normalize(all_numeric()) %>% 
  step_poly(JoiningYear,
            degree = 2,
            role = "predictor")

# Recipe 3: Norm + Poly + Dummy, Joining Year
turnover_norm_poly_dummy <-
  recipe(formula = Turnover ~ ., 
         data = turnover_train) %>% 
  step_dummy(all_nominal_predictors()) %>% 
  step_normalize(all_numeric()) %>% 
  step_poly(JoiningYear,
            degree = 2,
            role = "predictor")

#### QUESTION 3 ####

#### FITTING ----

model_RF <-
  rand_forest() %>% 
  set_mode("classification") %>% 
  set_engine("ranger",
             importance = "impurity") %>% 
  set_args(mtry = tune())

#### TUNING ----

# Set Workflow ----
# Recipe 1: Poly + Dummy, Joining Year
poly_dummy_rf <-
  workflow() %>% 
  add_recipe(turnover_poly_dummy) %>% 
  add_model(model_RF)

# Recipe 2: Norm + Poly + Dummy + KNN, Joining Year
norm_poly_dummy_knn_rf <-
  workflow() %>% 
  add_recipe(turnover_norm_poly_dummy_knn) %>% 
  add_model(model_RF)

# Recipe 3: Norm + Poly + Dummy, Joining Year
norm_poly_dummy_rf <-
  workflow() %>% 
  add_recipe(turnover_norm_poly_dummy) %>% 
  add_model(model_RF)

# Cross Validation ----

set.seed(100852)

cv10 <- 
  turnover_train %>% 
  vfold_cv(v = 10)

# Set up Proccessing ----

install.packages("doParallel")
library(doParallel)

registerDoParallel()

grid_RF <- 
  expand.grid(mtry = c(3, 4, 5, 6)
  )

# Tune Grid ----

# Recipe 1: Poly + Dummy, Joining Year
set.seed(100861)

poly_dummy_rf_TUNED <-
  poly_dummy_rf %>% 
  tune_grid(resamples = cv10,
            grid = grid_RF,
            metrics = metric_set(accuracy,
                                 roc_auc,
                                 f_meas))

# Recipe 2: Norm + Poly + Dummy + KNN, Joining Year
set.seed(100862)

norm_poly_dummy_knn_rf_TUNED <-
  norm_poly_dummy_knn_rf %>% 
  tune_grid(resamples = cv10,
            grid = grid_RF,
            metrics = metric_set(accuracy,
                                 roc_auc,
                                 f_meas))

# Recipe 3: Norm + Poly + Dummy, Joining Year
set.seed(100863)

norm_poly_dummy_rf_TUNED <-
  norm_poly_dummy_rf %>% 
  tune_grid(resamples = cv10,
            grid = grid_RF,
            metrics = metric_set(accuracy,
                                 roc_auc,
                                 f_meas))

# select_best() ----

# Recipe 1: Poly + Dummy, Joining Year
poly_dummy_rf_PARAM <-
  poly_dummy_rf_TUNED %>% 
  select_best(metric = "roc_auc")

poly_dummy_rf_PARAM

# Recipe 2: Norm + Poly + Dummy + KNN, Joining Year
norm_poly_dummy_knn_rf_PARAM <-
  norm_poly_dummy_knn_rf_TUNED %>% 
  select_best(metric = "roc_auc")

norm_poly_dummy_knn_rf_PARAM

# Recipe 3: Norm + Poly + Dummy, Joining Year
norm_poly_dummy_rf_PARAM <-
  norm_poly_dummy_rf_TUNED %>% 
  select_best(metric = "roc_auc")

norm_poly_dummy_rf_PARAM

# Step 4.5. finalize_workflow() ----

# Recipe 1: Poly + Dummy, Joining Year
poly_dummy_FINAL <-
  poly_dummy_rf %>% 
  finalize_workflow(poly_dummy_rf_PARAM)

poly_dummy_FINAL

# Recipe 2: Norm + Poly + Dummy + KNN, Joining Year
norm_poly_dummy_knn_FINAL <-
  norm_poly_dummy_knn_rf %>% 
  finalize_workflow(norm_poly_dummy_knn_rf_PARAM)

norm_poly_dummy_knn_FINAL

# Recipe 3: Norm + Poly + Dummy, Joining Year
norm_poly_dummy_FINAL <-
  norm_poly_dummy_rf %>% 
  finalize_workflow(norm_poly_dummy_rf_PARAM)

norm_poly_dummy_FINAL

# last_fit() ----

# Recipe 1: Poly + Dummy, Joining Year
poly_dummy_FIT <-
  poly_dummy_FINAL %>% 
  last_fit(turnover_split)

# Recipe 2: Norm + Poly + Dummy + KNN, Joining Year
norm_poly_dummy_knn_FIT <-
  norm_poly_dummy_knn_FINAL %>% 
  last_fit(turnover_split)

# Recipe 3: Norm + Poly + Dummy, Joining Year
norm_poly_dummy_FIT <-
  norm_poly_dummy_FINAL %>% 
  last_fit(turnover_split)

poly_dummy_FIT
norm_poly_dummy_knn_FIT
norm_poly_dummy_FIT

#### ASSESS ####

# Recipe 1: Poly + Dummy, Joining Year
poly_dummy_PERFORM <-
  poly_dummy_FIT %>% 
  collect_metrics() %>% 
  mutate(algorithm = "Random Forest (with Poly and Dummy, Joining Year)")

# Recipe 2: Norm + Poly + Dummy + KNN, Joining Year
norm_poly_dummy_knn_PERFORM <-
  norm_poly_dummy_knn_FIT %>% 
  collect_metrics() %>% 
  mutate(algorithm = "Random Forest (with Norm, Poly, and Dummy, Joinig Year + KNN)")

# Recipe 3: Norm + Poly + Dummy, Joining Year
norm_poly_dummy_PERFORM <-
  norm_poly_dummy_FIT %>% 
  collect_metrics() %>% 
  mutate(algorithm = "Random Forest (with Norm, Poly and Dummy, Joining Year)")

poly_dummy_PERFORM
norm_poly_dummy_knn_PERFORM
norm_poly_dummy_PERFORM

# Comparison Table

Comparison_Table <-
  bind_rows(poly_dummy_PERFORM,
            norm_poly_dummy_knn_PERFORM,
            norm_poly_dummy_PERFORM) %>% 
  select(-.config) %>% 
  pivot_wider(names_from = .metric,
              values_from = .estimate) %>% 
  DT::datatable() %>% 
  DT::formatRound(columns = c("accuracy",
                              "roc_auc"),
                  digits = 3)


#### QUESTION 4 ####

# From ROC_AUC, RF with Poly and Dummy, Joining Year Performed best

# Individual Prediction

norm_poly_dummy_knn_PRED_TRUTH <-
  norm_poly_dummy_knn_FIT %>% 
  collect_predictions()

PRED_TRUTH <-
  norm_poly_dummy_knn_PRED_TRUTH %>% 
  select(.pred_class, Turnover) %>% 
  rename(actual_Y = Turnover)

skim(PRED_TRUTH)

# Confusion Matrix Function Found Below ----

Confusion_Matrix <-
  PRED_TRUTH %>% 
  CM_builder_for_comm301()


#### QUESTION 5 ####

# Recipe 1: Poly + Dummy, Joining Year
poly_dummy_PRED_TRUTH_compare <-
  poly_dummy_FIT %>% 
  collect_predictions() %>% 
  mutate(algorithm = "Random Forest (Artificial Intelligence - Norm & Dummy)")

# Recipe 2: Norm + Poly + Dummy + KNN, Joining Year
norm_poly_dummy_knn_PERFORM_compare <-
  norm_poly_dummy_knn_FIT %>% 
  collect_predictions() %>% 
  mutate(algorithm = "Random Forest (Artificial Intelligence - Poly & Dummy, Age)")

# Recipe 3: Norm + Poly + Dummy, Joining Year
norm_poly_dummy_PRED_TRUTH_compare <-
  norm_poly_dummy_FIT %>% 
  collect_predictions() %>% 
  mutate(algorithm = "Random Forest (Artificial Intelligence - Poly & Dummy, Joining Year)")

Lets_Compare <-
  bind_rows(poly_dummy_PRED_TRUTH_compare,
            norm_poly_dummy_knn_PERFORM_compare,
            norm_poly_dummy_PRED_TRUTH_compare)

AUC_curve <-
  Lets_Compare %>% 
  group_by(algorithm) %>% 
  roc_curve(Turnover,
            .pred_Yes) %>% 
  autoplot() +
  theme_bw() +
  theme(legend.position = c(.65, .25))

#### QUESTION 6 ####

# Fit Final Model

Feature_Importance <-
  norm_poly_dummy_knn_FINAL %>% 
  fit(turnover_cleaned) %>% 
  extract_fit_parsnip() %>% 
  vip::vip()

#### QUESTION 7 ####

# Assign Final Model
FINAL_MODEL <-
  norm_poly_dummy_knn_FINAL %>% 
  fit(turnover_cleaned)

target_employee <- 
  tribble(
    ~Education,
    ~JoiningYear,               
    ~City,          
    ~PaymentTier,         
    ~Age,          
    ~Gender,     
    ~EverBenched,
    ~ExperienceInCurrentDomain,
    "Masters",
    2018,
    "Singapore",
    2,
    27,
    "Female",
    "Yes",
    3) %>% 
  mutate(PaymentTier = as.factor(PaymentTier))
         

skim(target_employee)
datatable(target_employee)

Indi_Pred_Test <-
  FINAL_MODEL %>% 
  predict(target_employee) %>% 
  mutate(Outcome = ifelse(.pred_class == "Yes", 
                          "Employee predicted to leave, provide retention package",
                          "Employee NOT predicted to leave, no action required")) %>% 
  glue::glue_data("{.pred_class}, {Outcome}.")

# Confusion Matrix Function

CM_builder_for_comm301 <- 
  function(data, actual_Y)
  { 
    {data} %>% 
      conf_mat(estimate = .pred_class,
               truth = {actual_Y}
      ) %>% 
      pluck(1) %>% 
      as_tibble() %>% 
      mutate(cm_colors = ifelse(Truth == "Yes" & Prediction == "Yes", "True Positive",
                                ifelse(Truth == "Yes" & Prediction == "No", "False Negative",
                                       ifelse(Truth == "No" & Prediction == "Yes", 
                                              "False Positive", 
                                              "True Negative")
                                )
      )
      ) %>% 
      ggplot(aes(x = Prediction, y = Truth)) + 
      geom_tile(aes(fill = cm_colors), show.legend = F) +
      scale_fill_manual(values = c("True Positive" = "green1",
                                   "False Negative" = "red1",
                                   "False Positive" = "red1",
                                   "True Negative" = "green2")
      ) + 
      geom_text(aes(label = n), color = "white", size = 10) + 
      geom_label(aes(label = cm_colors), vjust = 2
      ) + 
      theme_fivethirtyeight() + 
      theme(axis.title = element_text()
      )
    }

# Thank you for taking the time and effort to look through our script :)

#### SAVE IMAGE ####

save.image("Employee_Turnover_Prediction.RData")



