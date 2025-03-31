#Trains weights for our four features to predict which pitch type will be thrown for the first pitch of a game given the pitcher
library(dplyr)
library(tidyr)
library(rsample) #Data splitting
library(future.apply) #Parallel processing
library(progressr) #Progess bar

set.seed(64)
handlers(global = TRUE)

Data = read.csv("/mnt/c/Users/Caldwell/Documents/Fanduel Take Home/mlb_pitch_velo_assessment.csv") 
head(Data)

#Drop some columns that are not needed
Data = Data %>% select(-home_team_id, -away_team_id, -venue_id, -is_top_half, -pitcher_id, -batter_id)

#Unique pitch types for each pitcher
PitcherArsenals <- Data %>% group_by(pitcher_name) %>% summarize(pitch_types = list(unique(pitch_type)), .groups = "drop")

#First Pitch of every game
FirstPitches_Game <- Data %>% filter(pitch_number ==1) #Should adjust this to be the first pitch for the away team too?

#Model for predicting the pitch type of the first pitch of the game
Predict_FirstPitch <- function(Pitcher, Batter, W_1, W_2, W_3, W_4, PitcherGameFreq, PitcherPAFreq, BatterPAFreq, BatterAnyFreq) {
  PitcherThrows = unlist(PitcherArsenals[PitcherArsenals$pitcher_name == Pitcher, "pitch_types"][[1]])

  #Filter the frequency tables for the specific pitcher and batter
  PitcherGame <- PitcherGameFreq %>% filter(pitcher_name == Pitcher)
  PitcherPA <- PitcherPAFreq %>% filter(pitcher_name == Pitcher)
  BatterPA <- BatterPAFreq %>% filter(batter_name == Batter, pitch_type %in% PitcherThrows)
  BatterAny <- BatterAnyFreq %>% filter(batter_name == Batter, pitch_type %in% PitcherThrows)
  
  WeightedProbabilities_DF <- data.frame(pitch_type = PitcherThrows, stringsAsFactors = FALSE) %>%
    left_join(PitcherGame, by = "pitch_type") %>%
    left_join(PitcherPA, by = "pitch_type") %>%
    left_join(BatterPA, by = "pitch_type") %>%
    left_join(BatterAny, by = "pitch_type") %>%
    replace_na(list(PitcherGameFreq = 0, PitcherPAFreq = 0, BatterPAFreq = 0, BatterAnyFreq = 0)) %>%
    select(pitch_type, PitcherGameFreq, PitcherPAFreq, BatterPAFreq, BatterAnyFreq) %>%
    mutate(WeightedProb = W_1 * PitcherGameFreq +
                          W_2 * PitcherPAFreq +
                          W_3 * BatterPAFreq +
                          W_4 * BatterAnyFreq)

  #If the sum of the weights is not 1, normalize them
  TotalWeight = sum(WeightedProbabilities_DF$WeightedProb)
  if(TotalWeight != 1 && TotalWeight > 0) {
    WeightedProbabilities_DF <- WeightedProbabilities_DF %>%
      mutate(WeightedProb = WeightedProb / TotalWeight)
  }
  
  #Find highest probability pitch type across all dimensions
  best_pitch <- NA #WeightedProbabilities_DF %>% slice_max(WeightedProb, n = 1, with_ties = FALSE) %>% pull(pitch_type)
   
  return(list(best_pitch = best_pitch, probability_table = WeightedProbabilities_DF))
}

#Training the weights

#Score function for KFold
score_fold <- function(split, W_1, W_2, W_3, W_4) {
  Test_FirstPitches_Game  <- assessment(split)
  Train_FirstPitches_Game <- analysis(split)

  # Recompute frequency tables on training data
  TrainData <- Data %>% anti_join(Test_FirstPitches_Game, by = "pitch_id")

  #Feature 1: Probability Distribution for the first pitch of the game BY pitcher
  Pitcher_FirstPitch_Game_Freq = Train_FirstPitches_Game %>% group_by(pitcher_name, pitch_type) %>% summarise(PitcherGameCount = n(), .groups = "drop") %>% 
    group_by(pitcher_name) %>% mutate(PitcherGameFreq  = PitcherGameCount/sum(PitcherGameCount))

  #First Pitch of every plate appearance
  FirstPitches_PA <- TrainData %>% group_by(game_id,batter_name) %>% arrange(game_id, pitch_number) %>% 
    slice(1) %>% ungroup() %>% as.data.frame() %>% filter(pre_pitch_balls == 0 & pre_pitch_strikes == 0)

  #Feature 2: Probability Distribution for the first pitch of any at bat BY pitcher
  Pitcher_FirstPitch_PA_Freq <- FirstPitches_PA %>% group_by(pitcher_name, pitch_type) %>% summarise(PitcherPACount = n(), .groups = "drop") %>% 
    group_by(pitcher_name) %>% mutate(PitcherPAFreq  = PitcherPACount/sum(PitcherPACount))
  
  #Feature 3:  Probability Distribution for the first pitch of any at bat BY Batter
  Batter_FirstPitch_PA_Freq <- FirstPitches_PA %>% group_by(batter_name, pitch_type) %>% summarise(BatterPACount = n(), .groups = "drop") %>% 
    group_by(batter_name) %>% mutate(BatterPAFreq  = BatterPACount/sum(BatterPACount))

  #Feature 4: Probability Distribution for any pitch of any at bat BY Batter
  Batter_AnyPitch_Freq <- TrainData %>% group_by(batter_name, pitch_type) %>% summarise(BatterAnyCount = n(), .groups = "drop") %>% 
    group_by(batter_name) %>% mutate(BatterAnyFreq  = BatterAnyCount/sum(BatterAnyCount)) 

  #Score on test set
  ScoreFunction_KFold <- function(TestData) {
    mean(
      sapply(1:nrow(TestData), function(i) {
        Probability <- Predict_FirstPitch(TestData$pitcher_name[i], TestData$batter_name[i], W_1, W_2, W_3, W_4, 
        Pitcher_FirstPitch_Game_Freq, Pitcher_FirstPitch_PA_Freq, Batter_FirstPitch_PA_Freq, Batter_AnyPitch_Freq)$probability_table %>% #Run the prediction function for each row
          filter(pitch_type == TestData$pitch_type[i]) %>% pull(WeightedProb) #Pull the predicted probability of the actual pitch type thrown
        if (length(Probability) == 0) Probability <- 0
        return(Probability)
      })
    )
  }
  ScoreFunction_KFold(Test_FirstPitches_Game)
}

N <- 30  #Initial sample size
InitialSamples <- data.frame(W_1 = runif(N, 0, 1), W_2 = runif(N, 0, 1), W_3 = runif(N, 0, 1), W_4 = runif(N, 0, 1)) %>%
  rowwise() %>% mutate(total = W_1 + W_2 + W_3 + W_4) %>% mutate(
    W_1 = W_1 / total, W_2 = W_2 / total,
    W_3 = W_3 / total, W_4 = W_4 / total) %>% ungroup()

Results <- list()
SampleBatches <- list(InitialSamples)
for (i in 1:5) { #Run 5 batches
  CurrentBatch <- SampleBatches[[i]]
  message("Running batch ", i, " of 10")

  with_progress({ 
    p <- progressor(steps = nrow(current_batch)) #Initialize progress bar for total number of rows in current batch
    batch_results <- future_lapply(1:nrow(current_batch), function(j) {
      W1 <- CurrentBatch$W_1[j]
      W2 <- CurrentBatch$W_2[j]
      W3 <- CurrentBatch$W_3[j]
      W4 <- CurrentBatch$W_4[j]
      Score <- mean(sapply(kFolds$splits, score_fold, W_1 = W1, W_2 = W2, W_3 = W3, W_4 = W4))

      p() #Update progress bar
      data.frame(W_1 = W1, W_2 = W2, W_3 = W3, W_4 = W4, Score = Score)
    })
  })

  Results[[i]] <- bind_rows(batch_results)
  AllResults <- bind_rows(Results)

  #Save best weights so far
  TopSoFar <- AllResults %>% slice_max(Score, n = max(1, floor(nrow(AllResults) * 0.05)))

  message("Top Score So Far: ", max(TopSoFar$Score, na.rm = TRUE))

  #If TopSoFar is empty for any reason, break
  if (nrow(TopSoFar) == 0) {
    message("No top results found. Stopping at batch ", i)
    break
  }

  #Generate next batch by perturbing top weights
  Next_Batch <- TopSoFar %>% slice_sample(n = 100, replace = TRUE) %>%
    mutate(
      W_1 = W_1 + rnorm(n(), 0, 0.05),
      W_2 = W_2 + rnorm(n(), 0, 0.05),
      W_3 = W_3 + rnorm(n(), 0, 0.05),
      W_4 = W_4 + rnorm(n(), 0, 0.05)) %>%
    mutate(across(starts_with("W_"), ~ pmin(pmax(., 0), 1))) %>%  #Ensure each of the weight are in [0, 1]
    rowwise() %>% mutate(Total = W_1 + W_2 + W_3 + W_4) %>%
    mutate( #Normalize weights so they add up to 1
      W_1 = W_1 / Total,
      W_2 = W_2 / Total,
      W_3 = W_3 / Total,
      W_4 = W_4 / Total) %>% ungroup()
      
  SampleBatches[[i + 1]] <- Next_Batch
}

# Final combined results
CrossValidation_Results <- bind_rows(Results)
#write.csv(CrossValidation_Results, "smart_search_results.csv", row.names = FALSE)

# Output top-performing weight combo
max_score_row <- CrossValidation_Results %>% slice_max(Score, n = 1)
print(max_score_row)


#Final output for optimal weights is W_1 = 1, W_2 = 0, W_3 = 0, W_4 = 0
