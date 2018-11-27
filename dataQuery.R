library(tidyverse)
library(data.table) #fread
library(DBI) #get BQ data
library(dbplyr) #BQ
library(bigrquery) 
library(cloudml)

options(httr_oob_default = TRUE)
con <- dbConnect(
  bigrquery::bigquery(),
  project = "gn-data-science-project02",
  billing = "gn-data-science-project02", dataset = "jb"
)

demand_forecast_view <- tbl(con, "gn-data-science-project02.jb.demand_forecast_view")

sql <- "SELECT `store_number`, `product_id`, `fiscal_week_in_year`, `fiscal_year`, `total_units`, `biz_cd`, CONCAT(cast(`fiscal_year` as string), '_', cast(`fiscal_week_in_year` as string)) AS `year_week`
FROM (SELECT *
FROM `gn-data-science-project02.jb.demand_forecast_view`)"

#store as table object to export to cloud 
tb <- bq_project_query("gn-data-science-project02", sql)


#BQ API has a 1GB limit on file export size
#so putting an asterisk will let it output many files
bq_table_save(tb, "gs://jb_ds/demand_forecast_query/*.csv", 
              destination_format = "CSV")

gs_copy("gs://jb_ds/demand_forecast_query/*.csv", "query_files/")





  


