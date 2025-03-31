#Construct velocity distributions for each pitch type a pitcher might throw for the first pitch of the game
#Compute total probability of velocity being above a threshold for any pitcher and pitch type
library(dplyr)
library(tidyr)
library(mgcv) #Generalized Additive Models
library(future.apply) #Parallel processing
library(progressr) #Progess bar

Data = read.csv("/mnt/c/Users/Caldwell/Documents/Fanduel Take Home/mlb_pitch_velo_assessment.csv")
head(Data)

set.seed(64)

#Drop some columns that are not needed
Data = Data %>% select(-home_team_id, -away_team_id, -venue_id, -is_top_half, -pitcher_id, -batter_id)

#Unique pitch types for each pitcher
PitcherArsenals <- Data %>% filter(!is.na(pitch_type) & pitch_type != "") %>% distinct(pitcher_name, pitch_type)

#### Fit Generalized Additive Model to model decrease in velocity by pitch number for each pitch for each pitcher
#Data preprocessing
ReleaseSpeed_by_PitchNumber = Data %>% filter(!is.na(release_speed), !is.na(pitch_type), pitch_type != "") %>%
    group_by(date, pitcher_name) %>% arrange(pitch_number) %>%
    mutate(Accurate_PitchNumber = row_number()) %>% ungroup() %>% #Recreate the pitch number column excluding other pitchers
    group_by(pitcher_name, pitch_type, Accurate_PitchNumber) %>% 
    summarise(Average_ReleaseSpeed = mean(release_speed, na.rm = TRUE), #Calculate average release speed
        Count = n(), .groups = "drop") %>% #Include the number of times this pitch was thrown at that pitch number
    split(f =  interaction(.$pitcher_name, .$pitch_type, drop = TRUE))
  

#Training the models
plan(multisession, workers = 12)

with_progress({
    p <- progressor(steps = length(ReleaseSpeed_by_PitchNumber))

    #Train a weighted GAM with k=3 and REML to find average release speed by pitch number for each pitcher pitch type combo
    GAM_ReleaseSpeed_by_PitchNumber = lapply(ReleaseSpeed_by_PitchNumber, function(df){
        p() #Update progress bar

        tryCatch({
            mgcv::gam(Average_ReleaseSpeed ~ s(Accurate_PitchNumber, k=3), data = df,
                weights = df$Count, #Weight by sample size
                method = "REML") #Flatten edges 
        }, error = function(e) { #If not enough data to fit the model
            message("Error in GAM fitting: ", e$message, " for pitcher: ", unique(df$pitcher_name), " pitch type: ", unique(df$pitch_type))
            NULL
        })
    })
})
plan(sequential)

#### Next we can use the fitted GAM to predict the expected velocity for each pitch number for each pitcher for each pitch type
names(GAM_ReleaseSpeed_by_PitchNumber) <- names(ReleaseSpeed_by_PitchNumber) #Syncing names

Data_valid <- Data %>% filter(paste0(pitcher_name, ".", pitch_type) %in% names(GAM_ReleaseSpeed_by_PitchNumber)) #Filter data to only valid models

Data_valid <- Data_valid %>% group_by(date, pitcher_name) %>% arrange(pitch_number) %>% 
  mutate(Accurate_PitchNumber = row_number()) %>% ungroup() #Add accurate pitch number for all pitches

#Split by pitcher and pitch type
SplitData <- split(Data_valid, f = interaction(Data_valid$pitcher_name, Data_valid$pitch_type, drop = TRUE))

plan(multisession, workers = 12)
#Parallel batch prediction for expected velocity
with_progress({
    p <- progressor(steps = length(SplitData))
    Predicted <- future_lapply(SplitData, function(df) {
        local_p <- p
        if(runif(1) < 0.05) local_p()
        
        model <- GAM_ReleaseSpeed_by_PitchNumber[[paste0(df$pitcher_name[1], ".", df$pitch_type[1])]]

        if(!is.null(model)){
            X <- predict.gam(model, newdata = df, type = "lpmatrix")
            beta <- coef(model)
            df$Expected_ReleaseSpeed <- as.vector(X %*% beta)
        } else{
           df$Expected_ReleaseSpeed <- NA
        }
    return(df)
    }, future.seed = TRUE)
})
plan(sequential)

Data_valid <- as.data.frame(bind_rows(Predicted))
head(Data_valid)

#Now we have the "expected velocity" for each pitch number

#Now lets get the smoothed standard deviation for velocity at each pitch number
Standard_Deviation_by_PitchNumber = Data_valid %>% mutate(Key = paste0(pitcher_name, ".", pitch_type),
    Squared_Residual = (release_speed - Expected_ReleaseSpeed)^2) %>% #Squared residuals for expected value for each pitch
    group_by(Key, Accurate_PitchNumber) %>% 
    summarise(Average_Squared_Residual = mean(Squared_Residual, na.rm = TRUE), #Average squared residuals for each pitcher and pitch type at each pitch number
        Count = n(), .groups = "drop") %>% #Include the number of times this pitch was thrown at that pitch number
    mutate(Approximated_StandardDeviation = sqrt(Average_Squared_Residual))  #Approximate standard deviation

Split_SD_by_PitchNumber <- split(Standard_Deviation_by_PitchNumber, Standard_Deviation_by_PitchNumber$Key)

#From here we can fit a GAM to the standard deviation at each pitch number for each pitcher and pitch type
#This will give us a smoothed standard deviation for each pitch number
#It is the same as the GAM we fit to the average release speed
plan(multisession, workers = 12)

with_progress({
  p <- progressor(steps = length(Split_SD_by_PitchNumber))

  GAM_SD_by_PitchNumber <- future_lapply(Split_SD_by_PitchNumber, function(df) {
    p()
    tryCatch({ 
      gam(Approximated_StandardDeviation ~ s(Accurate_PitchNumber, k = 3), data = df,
          weights = df$Count, #Weight by sample size
          method = "REML") #Flatten edges
    }, error = function(e) NULL)  #If not enough data to fit the model
  })
})
plan(sequential)

names(GAM_SD_by_PitchNumber) <- names(Split_SD_by_PitchNumber) #Syncing names


#We repeat the process to predict the expected standard deviation at each pitch number for each pitcher, pitch type combo
SplitData <- split(Data_valid, f = interaction(Data_valid$pitcher_name, Data_valid$pitch_type, drop = TRUE))

plan(multisession, workers = 12)

with_progress({
    p <- progressor(steps = length(SplitData))
    Predicted_SD <- future_lapply(SplitData, function(df) {
        local_p <- p
        if(runif(1) < 0.05) local_p()
        
        model <- GAM_SD_by_PitchNumber[[paste0(df$pitcher_name[1], ".", df$pitch_type[1])]]

        if(!is.null(model)){
            X <- predict.gam(model, newdata = df, type = "lpmatrix") #I did this because... the other thing wasnt running and its better because ..
            beta <- coef(model)
            df$Expected_StandardDevation_ReleaseSpeed <- as.vector(X %*% beta)
        } else{
           df$Expected_StandardDevation_ReleaseSpeed <- NA
        }
    return(df)
    }, future.seed = TRUE)
})
plan(sequential)

Data_valid <- bind_rows(Predicted_SD)
head(Data_valid)

#### we can use this to normalize each pitch they've thrown to the expected velocity if it was the first pitch

#We find the z-score of each pitch thrown
Data_valid <- Data_valid %>% mutate(Pitch_ZScore = (release_speed - Expected_ReleaseSpeed) / Expected_StandardDevation_ReleaseSpeed) %>% 
    filter(!is.na(Pitch_ZScore)) #Filter out any NA values

#Dataframe with Expected_ReleaseSpeed and Expected_StandardDevation_ReleaseSpeed for just the first pitches
FirstPitches <- Data_valid %>% filter(pitch_number == 1) %>%
    distinct(pitcher_name, pitch_type, .keep_all = TRUE) %>%
    select(pitcher_name, pitch_type, Accurate_PitchNumber, Expected_ReleaseSpeed, Expected_StandardDevation_ReleaseSpeed)

#For pitches in a pitcher's arsenal that have never been thrown as the first pitch
FirstPitches_Missing <- PitcherArsenals %>% anti_join(FirstPitches, by = c("pitcher_name", "pitch_type")) %>%
    left_join(Data_valid, by = c("pitcher_name", "pitch_type")) %>%
    group_by(pitcher_name, pitch_type) %>%
    arrange(date, pitch_number) %>% slice(1) %>% ungroup() %>%
    select(pitcher_name, pitch_type, Accurate_PitchNumber, Expected_ReleaseSpeed, Expected_StandardDevation_ReleaseSpeed)

FirstPitches <- bind_rows(FirstPitches, FirstPitches_Missing) #combine them

#We fit a KDE across all of the z-scores for each pitcher and pitch type
SplitData <- split(Data_valid, f = interaction(Data_valid$pitcher_name, Data_valid$pitch_type, drop = TRUE))

Pitcher_KDE <- list()
for (key in names(SplitData)) { #for every pitcher, pitch type combo
    df <- SplitData[[key]]

    #This fits a KDE to the "normalized" velocities
    KDE_Temp <- density(df$Pitch_ZScore, bw = "nrd0") #Using normal reference distribution

    #Find all the first pitches for this pitcher, pitch type combo
    FirstPitches_Temp <- FirstPitches %>% filter(pitcher_name == strsplit(key, "\\.")[[1]][1], pitch_type == strsplit(key, "\\.")[[1]][2])

    #Rescale the KDE frrom z-score space back to velocity space
    #x-axis: z*sigma+mu
    Pitcher_KDE[[key]] <- list(KDE_Temp$x * FirstPitches_Temp$Expected_StandardDevation_ReleaseSpeed + FirstPitches_Temp$Expected_ReleaseSpeed, 
        KDE_Temp$y / FirstPitches_Temp$Expected_StandardDevation_ReleaseSpeed) #y-axis: pdf(z) / sigma (preserve total probability)
}

CumulativeProbability = function(pitcher_name, pitch_type, Threshold, Pitcher_KDE = Pitcher_KDE) {
    Selected_KDE = Pitcher_KDE[[paste0(pitcher_name, ".", pitch_type)]]

    #Smoothing the KDE
    Spline <- splinefun(Selected_KDE[[1]], Selected_KDE[[2]], method = "natural")

    Maximum_ReleaseSpeed <- max(Selected_KDE[[1]])
    if(Threshold > Maximum_ReleaseSpeed) return(0)

    #integrate over the smoothed KDE from our lower threshold to upper
    Probability <- integrate(Spline, lower = Threshold, upper = Maximum_ReleaseSpeed)$value
    return(Probability)
}

CumulativeProbability("Aaron Nola", "FF", 89.95, Pitcher_KDE = Pitcher_KDE)





