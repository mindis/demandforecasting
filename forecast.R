library(tidyverse)
library(data.table)

demand <- lapply(list.files("query_files/", full.names = T), fread, sep = ",") %>% 
  rbindlist() %>%
  mutate(fiscal_week_in_year = as.character(fiscal_week_in_year)) %>%
  mutate(fiscal_week_in_year = ifelse(nchar(fiscal_week_in_year) == 2, fiscal_week_in_year,
                                      paste0("0", fiscal_week_in_year))) %>%
  mutate(year_week = as.integer(paste0(fiscal_year, fiscal_week_in_year)))


start <- 201601
stop <- 201727
n_time_steps <- stop - start

prod_demand_16 <- demand %>%
  filter(between(year_week, start, stop)) %>% 
  data.table() %>%
  .[, sum(total_units, na.rm = TRUE), by = list(product_id, store_number)] %>%
  filter(V1 >= 40) %>% # Arbitrary value to filer out products without much sales information
  rename(total_sales = V1)

setDT(demand)
demand_wide <- demand %>% 
  filter(fiscal_year == 2016 & between(fiscal_week_in_year, "01", "26")) %>%
  select(-fiscal_week_in_year, -fiscal_year) %>%
  data.table() %>%
  dcast.data.table(formula = product_id + store_number + biz_cd ~ year_week, 
                   fun.aggregate = sum, 
                   value.var = "total_units") %>% 
  full_join(demand %>% 
              filter(fiscal_year == 2016 & between(fiscal_week_in_year, "27", "52")) %>%
              select(-fiscal_week_in_year, -fiscal_year) %>%
              data.table() %>%
              dcast.data.table(formula = product_id + store_number + biz_cd ~ year_week, 
                               fun.aggregate = sum, 
                               value.var = "total_units")) %>%
  full_join(demand %>% 
              filter(fiscal_year == 2017 & between(fiscal_week_in_year, "01", "26")) %>%
              select(-fiscal_week_in_year, -fiscal_year) %>%
              data.table() %>%
              dcast.data.table(formula = product_id + store_number + biz_cd ~ year_week, 
                               fun.aggregate = sum, 
                               value.var = "total_units")) %>%
  
  left_join(prod_demand_16) %>% 
  filter(!is.na(total_sales)) %>%
  mutate_all(funs(replace(., which(is.na(.)), 0))) %>%
  arrange(product_id, store_number) %>% 
  mutate(scale_factor = 1 + total_sales / n_time_steps) %>%
  mutate(sampling_probability = scale_factor / sum(scale_factor)) %>%
  mutate(biz_cd_int = as.integer(as.factor(biz_cd)))

  
# saveRDS(demand_wide, "demand_wide.RDS")


