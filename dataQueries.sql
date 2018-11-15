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
where biz_cd in ('260.10.1', '600.24.4', '520.1.4', '500.1.1', '700.24.1', '311.1.7', '315.1.3'); --just PoC areas