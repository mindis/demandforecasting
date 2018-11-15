select * from all_tab_columns where owner LIKE 'DDW%' and column_name LIKE '%PRODUCT_ID%';

select *
from ddw.mdm_product_attr_detail; --product attribute data

select *
from ddw.mdm_location_attr_detail; --location attribute data

select *
from ddw.omcl_sale_fct; --store sku sales

select *
from ddw.product

select *
from
(select (department_number || '.' || sub_department_number || '.' || class_number) as biz_cd, ph.*
from ddw.product_hierarchy_dim ph)
join 
<<<<<<< HEAD
where biz_cd in ('260.10.1', '600.24.4', '520.1.4', '500.1.1', '700.24.1', '311.1.7', '315.1.3'); --just PoC areas
=======
where biz_cd in ('260.10.1', '600.24.4', '350.5.1', '500.1.1', '700.24.1', '160.4.1', '315.1.3'); --just PoC areas

-- STORE / SKU / FY Week sales for 2015 - 2017.
-- Not complete with all covariates which will eventually be required.

select dd.fiscal_week_in_year ,ohd.store_number, ohd.store_name, omcl.product_id, phd.department_number, 
phd.sub_department_number, phd.class_number,
sum(omcl.net_sales_units) as tot_units 
from ddw.omcl_sale_fct omcl
inner join ddw.organization_hierarchy_dim ohd
on omcl.o_organization_hierarchy_id = ohd.organization_hierarchy_id
and ohd.chain_number = 1 -- DSG
and omcl.date_id >= 20150101
and omcl.date_id <= 20171231
and omcl.return_units = 0 -- Excluding returns for now??
join ddw.date_dim dd
on dd.date_id = omcl.date_id
join ddw.product_dim pd
on omcl.product_id = pd.product_id
join ddw.product_hierarchy_dim phd
on pd.product_hierarchy_id = phd.product_hierarchy_id
group by dd.fiscal_week_in_year, ohd.store_number, omcl.product_id, ohd.store_name, phd.department_number, 
phd.sub_department_number, phd.class_number
order by tot_units desc;
>>>>>>> 5d7e0f7fd4d56b00691d543788d6067ab54186f0
