library(data.table)
library(tidyverse)
library(lubridate)
rm(list=ls())

tz = "America/New_York"

out_psg_raw <- "path to raw_psg"
feature_path <- "path to feature_output"

SAMPLING_RATE <- 50
# Helper Function
# Zero Cross Rate
calculate_zcr <- function(v) {
  v_centered <- v - mean(v)
  zcr <- sum(diff(sign(v_centered)) != 0) / (length(v_centered) - 1)
  return(zcr)
}
# FFT features
get_fft_features <- function(v, sampling_rate) {
  n <- length(v)
  if (n == 0) return(list(dom_freq = NA, energy = NA))
  fft_result <- fft(v)
  magnitudes <- Mod(fft_result[1:(n/2)])
  frequencies <- (0:(n/2 - 1)) * sampling_rate / n
  dom_freq <- frequencies[which.max(magnitudes)]
  energy <- sum(magnitudes^2)
  return(list(dom_freq = dom_freq, energy = energy))
}
get_mode <- function(v) {
  freq_table <- table(v)
  return(names(freq_table)[which.max(freq_table)])
}
# Auto-correlation
get_autocor <- function(x,n) {
  y <- data.table::shift(x,n)
  y <- y[(n+1):length(y)]
  x <- x[(n+1):length(x)]
  cor(x,y)
}


# Check existed data and only update the new data files
out_psg_raw_list <- list.files(out_psg_raw, ".csv")
out_psg_raw_list_name <- word(list.files(out_psg_raw, ".csv"),1 , sep = "_acc")

existed_list <- word(list.files(feature_path, ".csv"),1, sep = "_feature")
updated_raw_list_name <- setdiff(out_psg_raw_list_name,existed_list)

# --- Create Epochs and Aggregate Features ---
for(i in 1:length(updated_raw_list_name)){
  psg_raw_select <- updated_raw_list_name[i]
  data_name_select <- out_psg_raw_list[grep(paste0("^",psg_raw_select,"_"),out_psg_raw_list)]
  
  dt <- fread(file.path(out_psg_raw,data_name_select))
  if(nrow(dt) >0) {
  ID <- unique(dt$ID)
  # Create a new column 'epoch' by rounding the timestamp down to the nearest 30-second interval
  dt[, epoch := floor_date(timestamp, unit = "30 seconds")]
  #Group by each epoch and calculate all features at once.
  feature_dt <- dt[, {
    
    mag <- sqrt(x^2 + y^2 + z^2)
    
    # FFT features (same as before)
    fft_x <- get_fft_features(x, SAMPLING_RATE)
    fft_y <- get_fft_features(y, SAMPLING_RATE)
    fft_z <- get_fft_features(z, SAMPLING_RATE)
    fft_mag <- get_fft_features(mag, SAMPLING_RATE)
    
    autocortwo_x <- get_autocor(x,200)
    autocoreight_x <- get_autocor(x,800)
    autocortwo_y <- get_autocor(y,200)
    autocoreight_y <- get_autocor(y,800)
    autocortwo_z <- get_autocor(z,200)
    autocoreight_z <- get_autocor(z,800)
    
    .(
      label = get_mode(label),
      # --- Time-Domain Statistical Features ---
      # (Mean, SD, and Range are the same)
      mean_x = mean(x), sd_x = sd(x), range_x = max(x) - min(x),
      mean_y = mean(y), sd_y = sd(y), range_y = max(y) - min(y),
      mean_z = mean(z), sd_z = sd(z), range_z = max(z) - min(z),
      mean_mag = mean(mag), sd_mag = sd(mag), range_mag = max(mag) - min(mag),
      
      # Median
      median_x = median(x), median_y = median(y), median_z = median(z), median_mag = median(mag),
      
      # Interquartile Range (IQR)
      iqr_x = IQR(x), iqr_y = IQR(y), iqr_z = IQR(z), iqr_mag = IQR(mag),
      
      #  Mode (most frequent value)
      mode_x = as.numeric(get_mode(round(x, 2))), # Round to make mode meaningful
      mode_y = as.numeric(get_mode(round(y, 2))),
      mode_z = as.numeric(get_mode(round(z, 2))),
      mode_mag = as.numeric(get_mode(round(mag, 2))),
      
      # --- Frequency-Domain Features ---
      zcr_x = calculate_zcr(x), zcr_y = calculate_zcr(y), zcr_z = calculate_zcr(z),
      dom_freq_x = fft_x$dom_freq, energy_x = fft_x$energy,
      dom_freq_y = fft_y$dom_freq, energy_y = fft_y$energy,
      dom_freq_z = fft_z$dom_freq, energy_z = fft_z$energy,
      dom_freq_mag = fft_mag$dom_freq, energy_mag = fft_mag$energy
    )
  }, by = epoch]
  
  fwrite(feature_dt, file.path(feature_path,paste0(ID,"_feature.csv")))
  rm(feature_dt)
  rm(dt)
  }
}
