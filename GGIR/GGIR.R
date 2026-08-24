
# Define the required version
required_ggir_version <- "3.3-4"

ggir_installed <- FALSE

# Check if GGIR is installed at all
if (requireNamespace("GGIR", quietly = TRUE)) {
  # If it is, check if the version is correct
  if (packageVersion("GGIR") == required_ggir_version) {
    cat("GGIR version", required_ggir_version, "is already installed.\n")
    ggir_installed <- TRUE
  } else {
    cat("GGIR is installed, but it's the wrong version (", 
        as.character(packageVersion("GGIR")), "). Re-installing.\n")
  }
} else {
  cat("GGIR is not installed. Installing now.\n")
}

# Install only if it's missing or the wrong version
if (!ggir_installed) {
  if (!requireNamespace("remotes", quietly = TRUE)) {
    install.packages("remotes", repos = "http://cran.us.r-project.org")
  }
  remotes::install_github("wadpac/GGIR", ref = required_ggir_version)
}

if(!require(data.table)){
  install.packages("data.table", repos = "http://cran.us.r-project.org")
  library(data.table)
} else {
  library(data.table)
}
if(!require(tidyverse)){
  install.packages("tidyverse", repos = "http://cran.us.r-project.org")
  library(tidyverse)
} else {
  library(tidyverse)
}

library(GGIR)
rm(list = ls())


# AGV < ----------------------


tz = "America/New_York"

datadir = "path_to_raw_actigraphy (.bin files)"
GGIR_result = "path_to_output_results"

testing_name_list = c("/GGIR_334_CK/","/GGIR_334_Sadeh/")
algo_name_list = c("ColeKripke1992","Sadeh1994")
for (test_n in 1:length(testing_name_list)){
  
  testing_name = testing_name_list[test_n]
  outputdir = paste0(GGIR_result,testing_name)
  dir.create(outputdir,recursive = TRUE)
  desiredtz = "America/New_York"
  
  algo_name = algo_name_list[test_n]
  Sadeh_axis = "Y" # this is needed for Cole-Kripke
  
  # Parameters -----------------------
    GGIR(
        nonwear_approach = "2023",
        mode = c(1:5),
        datadir = datadir, 
        outputdir = outputdir,  
        do.report = c(2,4,5), 
        desiredtz = desiredtz,
        HASIB.algo = algo_name, 
        Sadeh_axis = Sadeh_axis,
        # Part 1 =====================
        do.cal = T, 
        do.imp = T, 
        do.enmo = T,
        do.anglez= T, 
        chunksize= 1,
        do.parallel=T,
        minimumFileSizeMB = 1,
        
        #=====================
        # Part 2
        #=====================
        strategy = 1, 
        hrs.del.start = 0,          hrs.del.end = 0,
        maxdur = 0,  
        includedaycrit = 1, 
        qwindow = c(0,24), 
        mvpathreshold = c(100), 
        excludefirstlast = FALSE, 
        includenightcrit = 1, 
        cosinor = TRUE,
        #=====================
        # Part 3 + 4
        #=====================
        def.noc.sleep = 1, 
        outliers.only = TRUE, 
        criterror = 4, 
        #=====================
        # Part 5
        #=====================
        threshold.lig = c(30), threshold.mod = c(100),  threshold.vig = c(400),
        boutcriter = 0.8,      boutcriter.in = 0.9,     boutcriter.lig = 0.8,
        boutcriter.mvpa = 0.8, boutdur.in = c(1,10,30), boutdur.lig = c(1,10, 30),
        boutdur.mvpa = c(1, 5, 10), 
        includedaycrit.part5 = 1/10, 
        iglevels = c(seq(0,4000,by=25),8000), 
        qlevels = c(c(1380/1440),c(1410/1440),c(1430/1440)),
        part6HCA = F,
        #=====================
        # Visual report
        #=====================
        do.visual = T, 
        timewindow = c("WW"), 
        visualreport = T, 
        old_visualreport = F
    )
}

# Van Hees Run

testing_name = "/GGIR_334_vH/"
outputdir = paste0(GGIR_result,testing_name)
dir.create(outputdir,recursive = TRUE)
desiredtz = "America/New_York"

algo_name = "vanHees2015"
# Parameters -----------------------
GGIR(
  nonwear_approach = "2023",
  mode = c(1:5),
  datadir = datadir, 
  outputdir = outputdir,  
  do.report = c(2,4,5), 
  desiredtz = desiredtz,
  HASIB.algo = algo_name, 
  # Part 1 =====================
  do.cal = T, 
  do.imp = T, 
  do.enmo = T,
  do.anglez= T, 
  chunksize= 1,
  do.parallel=T,
  minimumFileSizeMB = 1,
  
  #=====================
  # Part 2
  #=====================
  strategy = 1, 
  hrs.del.start = 0,          hrs.del.end = 0,
  maxdur = 0,  
  includedaycrit = 1, 
  qwindow = c(0,24), 
  mvpathreshold = c(100), 
  excludefirstlast = FALSE, 
  includenightcrit = 1, 
  cosinor = TRUE,
  #=====================
  # Part 3 + 4
  #=====================
  def.noc.sleep = 1, 
  outliers.only = TRUE, 
  criterror = 4, 
  #=====================
  # Part 5
  #=====================
  threshold.lig = c(30), threshold.mod = c(100),  threshold.vig = c(400),
  boutcriter = 0.8,      boutcriter.in = 0.9,     boutcriter.lig = 0.8,
  boutcriter.mvpa = 0.8, boutdur.in = c(1,10,30), boutdur.lig = c(1,10, 30),
  boutdur.mvpa = c(1, 5, 10), 
  includedaycrit.part5 = 1/10, 
  iglevels = c(seq(0,4000,by=25),8000), 
  qlevels = c(c(1380/1440),c(1410/1440),c(1430/1440)),
  part6HCA = F,
  #=====================
  # Visual report
  #=====================
  do.visual = T, 
  timewindow = c("WW"), 
  visualreport = T, 
  old_visualreport = F

) 



