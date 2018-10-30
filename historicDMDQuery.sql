select * 
from
merch_user.stu_demand_4_1_3@ddwp; --historical demand forecast table

--Dmdunit – Demand unit is mastersku – colorcode.  This is the lowest level of product the demand planners forecast.  Any Dmdunit that is only the master sku is style level (8 digits).  
--DMDGROUP – all is DSG
--LOC – 5 digit store number.  Ecomm locations all start with ‘8888-‘ and then chain.  So 821 is ‘8888-DSG’.  ALL = chain level forecast.  
--FCSTDATE – Week of actual forecast
--Qty – forecast qty
--Dtype – forecast
--Data_date – date the data was grabbed

--There are two lags in this data, a 3 week and a 1 week.  
--There are also multiple levels of forecast data.  
--Demand planners forecast 3 levels: style/chain, style/color/chain and style/color/store.  
--You will want the style/color/store, and at the 3 lag.  The query for that is:
select *
from merch_user.stu_demand_4_1_3@ddwp
where loc <> 'ALL' 
AND LENGTH(DMDUNIT) > 8
AND FCSTDATE - DATA_DATE > 14
--and FCSTDATE < '02-FEB-2018'
