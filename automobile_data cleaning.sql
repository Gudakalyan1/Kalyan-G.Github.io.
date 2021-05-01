use kalyan;
select * from automobile_data;
select distinct(make) FROM automobile_data;
select distinct(fuel_type) from automobile_data;
select length from automobile_data order by length asc;
select max(length) as max_length,min(length) min_length from automobile_data;
select * from automobile_data where num_of_doors is null;
select distinct num_of_cylinders from automobile_data;
set sql_safe_updates=0;
update automobile_data set num_of_cylinders="two" where num_of_cylinders="tow";
select distinct compression_ratio from automobile_data;
delete  from automobile_data where compression_ratio="70";
select max(price) as max_price,min(price) as min_price from automobile_data;
select avg(price) from automobile_data;
update automobile_data set price =12978 where price=0;
select distinct(drive_wheels) from automobile_data ;
select length(drive_wheels) from automobile_data group by drive_wheels;
select drive_wheels from automobile_data where drive_wheel=4;

update automobile_data set drive_wheels=trim(drive_wheels) where drive_wheels >3;



