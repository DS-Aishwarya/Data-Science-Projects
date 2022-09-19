show databases;
use orders;
show tables;
describe product_class;

--Problem 1:Write a query to Display the product details (product_class_code, product_id, product_desc, product_price,)
-- as per the following criteria and sort them in descending order of category: 
--a. If the category is 2050, increase the price by 2000 
--b. If the category is 2051, increase the price by 500 
--c. If the category is 2052, increase the price by 600. 

--Solution:
SELECT PRODUCT_CLASS_CODE AS 'Product Catagory',
PRODUCT_ID AS 'Product ID',
PRODUCT_DESC AS 'Product Description',
PRODUCT_PRICE AS 'Actual Price', 
CASE PRODUCT_CLASS_CODE
WHEN 2050 THEN PRODUCT_PRICE+2000 
WHEN 2051 THEN PRODUCT_PRICE+500 
WHEN 2052 THEN PRODUCT_PRICE+600 
ELSE PRODUCT_PRICE 
END AS 'Calculated Price'
FROM PRODUCT 
ORDER BY PRODUCT_CLASS_CODE DESC;

--Problem 2 : Write a query to display (product_class_desc, product_id, product_desc, product_quantity_avail ) 
--and Show inventory status of products as below as per their available quantity:
--a. For Electronics and Computer categories, if available quantity is <= 10, show 'Low stock', 11 <= qty <= 30, show 'In stock', >= 31, show 'Enough stock' 
--b. For Stationery and Clothes categories, if qty <= 20, show 'Low stock', 21 <= qty <= 80, show 'In stock', >= 81, show 'Enough stock' 
--c. Rest of the categories, if qty <= 15 – 'Low Stock', 16 <= qty <= 50 – 'In Stock', >= 51 – 'Enough stock' 
--For all categories, if available quantity is 0, show 'Out of stock'. 

--Solution:
SELECT PC.PRODUCT_CLASS_DESC AS 'Product Category',
P.PRODUCT_ID AS 'Product ID',
P.PRODUCT_DESC AS 'Product Description',
P.PRODUCT_QUANTITY_AVAIL AS 'Product Availability',
CASE 
WHEN PC.PRODUCT_CLASS_CODE IN (2050,2053) THEN
CASE
WHEN P.PRODUCT_QUANTITY_AVAIL =0 THEN 'Out of stock' 
WHEN P.PRODUCT_QUANTITY_AVAIL <= 10 THEN 'Low stock'
WHEN (P.PRODUCT_QUANTITY_AVAIL >= 11 AND P.PRODUCT_QUANTITY_AVAIL <= 30) THEN 'In stock'
WHEN (PRODUCT_QUANTITY_AVAIL >= 31) THEN 'Enough stock'
END
WHEN PC.PRODUCT_CLASS_CODE IN (2052,2056) THEN
CASE
WHEN P.PRODUCT_QUANTITY_AVAIL =0 THEN 'Out of stock' 
WHEN P.PRODUCT_QUANTITY_AVAIL <= 20 THEN 'Low stock'
WHEN (P.PRODUCT_QUANTITY_AVAIL >= 21 AND P.PRODUCT_QUANTITY_AVAIL <= 80) THEN 'In stock'
WHEN (PRODUCT_QUANTITY_AVAIL >= 81) THEN 'Enough stock'
END
ELSE
CASE
WHEN P.PRODUCT_QUANTITY_AVAIL =0 THEN 'Out of stock' 
WHEN P.PRODUCT_QUANTITY_AVAIL <= 15 THEN 'Low stock'
WHEN (P.PRODUCT_QUANTITY_AVAIL >= 16 AND P.PRODUCT_QUANTITY_AVAIL <= 50) THEN 'In stock'
WHEN (PRODUCT_QUANTITY_AVAIL >=51) THEN 'Enough stock'
END
END AS 'Inventory Status'
FROM PRODUCT P
INNER JOIN PRODUCT_CLASS PC ON P.PRODUCT_CLASS_CODE = PC.PRODUCT_CLASS_CODE
ORDER BY P.PRODUCT_CLASS_CODE,P.PRODUCT_QUANTITY_AVAIL DESC;

--Problem 3: Write a query to show the number of cities in all countries other than USA & MALAYSIA, with more than 1 city, in the descending order of CITIES. (2 rows)

--Solution:
      SELECT COUNTRY, COUNT(DISTINCT(CITY)) as Number_of_Cities 
      from ADDRESS 
      WHERE COUNTRY NOT IN ('USA', 'Malaysia') 
      GROUP BY COUNTRY 
      HAVING COUNT(CITY)>1 
      ORDER BY Number_of_Cities DESC;


--Problem 4: Write a query to display the customer_id,customer full name ,city,pincode,and order details (order id, product class desc, product desc, subtotal(product_quantity * product_price)) for orders shipped to cities whose pin codes do not have any 0s in them. 
--Sort the output on customer name and subtotal. (52 ROWS)

--Solution:
SELECT  oc.customer_id,    CONCAT(oc.customer_fname,  '  ',  oc.customer_lname)  as  Fullname,  a.city, 
a.pincode,     oh.order_id,oh.order_date,     pc.product_class_desc,     p.product_desc, 
oi.product_quantity*p.product_price as Subtotal  
FROM online_customer oc   
INNER JOIN address a    
ON oc.address_id=a.address_id   
INNER JOIN order_header oh   
ON oc.customer_id=oh.customer_id   
INNER JOIN order_items oi   
ON oh.order_id=oi.order_id   
INNER JOIN product p   
ON oi.product_id=p.product_id   
INNER JOIN product_class pc   
ON p.product_class_code=pc.product_class_code   
WHERE oh.order_status='Shipped' AND a.PINCODE NOT LIKE  "%0%"   
ORDER BY Fullname,oh.order_date, subtotal;  


--Problem 5: Write a Query to display product id,product description,totalquantity(sum(product quantity) for a given item whose product id is 201 and which item has been bought along with it maximum no. of times. 
--Display only one record which has the maximum value for total quantity in this scenario.

--Solution:
SELECT p.product_id, product_desc, SUM(product_quantity) AS tot_qty 
FROM Order_Items oi, Product p 
WHERE order_id IN 
(SELECT order_id FROM Order_Items  
WHERE product_id = 201) 
AND p.product_id != 201 
AND oi.product_id = p.product_id 
GROUP BY p.product_id, product_desc 
ORDER BY tot_qty DESC LIMIT 1;



--Problem 6:Write a query to display the customer_id,customer name, email and order details (order id, product desc,product qty, subtotal(product_quantity * product_price)) for all customers even if they have not ordered any item.(225 ROWS) 
--[NOTE: TABLE TO BE USED - online_customer, order_header, order_items, product]

--Solution:
SELECT CONCAT(oc.customer_fname, ' ', oc.customer_lname) AS fullname, 
customer_email, oh.order_id, p.product_desc, oi.product_quantity AS prod_qty, 
oi.product_quantity * p.product_price AS subtotal 
FROM online_customer oc LEFT JOIN order_header oh 
ON oc.customer_id = oh.customer_id 
LEFT JOIN order_items oi 
ON oh.order_id = oi.order_id 
LEFT JOIN product p 
ON oi.product_id = p.product_id 
ORDER BY oc.customer_id, oh.order_id, p.product_desc; 


--Problem 7:Write a query to display carton id, (len*width*height) as carton_vol and identify the optimum carton (carton with the least volume whose volume is greater than the total volume of all items (len * width * height * product_quantity)) for a given order whose order id is 10006, Assume all items of an order are packed into one single carton (box). (1 ROW) 
--[NOTE: CARTON TABLE]

--Solution:

SELECT C.CARTON_ID , 
 (C.LEN*C.WIDTH*C.HEIGHT)AS Carton_Volume 
FROM ORDERS.CARTON C 
WHERE (C.LEN*C.WIDTH*C.HEIGHT) >= (
SELECT SUM(P.LEN*P.WIDTH*P.HEIGHT*OI.PRODUCT_QUANTITY) AS VOL 
 FROM 
ORDERS.ORDER_ITEMS OI 
INNER JOIN ORDERS.PRODUCT P ON OI.PRODUCT_ID = P.PRODUCT_ID  
WHERE OI.ORDER_ID =10006 )
ORDER BY (C.LEN*C.WIDTH*C.HEIGHT) ASC
LIMIT 1; 

--Problem 8: Write a query to display details (customer id,customer fullname,order id,product quantity) of customers who bought more than ten (i.e. total order qty) products with credit card or Net banking as the mode of payment per shipped order. (6 ROWS) 
--[NOTE: TABLES TO BE USED - online_customer, order_header, order_items,]

--Solution:

SELECT OC.CUSTOMER_ID AS Customer_ID, 
CONCAT(CUSTOMER_FNAME,' ',CUSTOMER_LNAME) AS Customer_FullName,
OH.ORDER_ID AS Order_ID,
 SUM(OI.PRODUCT_QUANTITY) AS Total_Order_Quantity,
 OH.PAYMENT_MODE FROM ONLINE_CUSTOMER OC
INNER JOIN ORDER_HEADER OH ON OC.CUSTOMER_ID = OH.CUSTOMER_ID 
INNER JOIN ORDER_ITEMS OI ON OH.ORDER_ID = OI.ORDER_ID 
WHERE OH.ORDER_STATUS = 'Shipped' AND (OH.PAYMENT_MODE = 'net banking' or OH.PAYMENT_MODE = 'credit card') 
GROUP BY OH.ORDER_ID 
HAVING Total_Order_Quantity > 10;

--Problem 9:Write a query to display the order_id, customer id and cutomer full name of customers starting with the alphabet "A" along with (product_quantity) as total quantity of products shipped for order ids > 10030. (5 ROWS) 
--[NOTE: TABLES TO BE USED - online_customer, order_header, order_items]

--Solution:
SELECT OC.CUSTOMER_ID AS Customer_ID, 
CONCAT(CUSTOMER_FNAME,' ',CUSTOMER_LNAME) AS Customer_FullName,
 OH.ORDER_ID AS Order_ID,
SUM(OI.PRODUCT_QUANTITY) AS Total_Order_Quantity
FROM ONLINE_CUSTOMER OC
INNER JOIN ORDER_HEADER OH ON OH.CUSTOMER_ID = OC.CUSTOMER_ID -- To connect the Order and Customer details.
INNER JOIN ORDER_ITEMS OI ON OH.ORDER_ID = OI.ORDER_ID -- To fetch the Product Quantity.
WHERE OH.ORDER_STATUS = 'Shipped' AND OH.ORDER_ID > 10030 -- To check for order_status whether it is shipped.
GROUP BY ORDER_ID 
HAVING Customer_FullName like 'A%' 
order by customer_fullname;

--Problem 10:Write a query to display product class description ,total quantity (sum(product_quantity),Total value (product_quantity * product price) and show which class of products have been shipped highest(Quantity) to countries outside India other than USA? Also show the total value of those items. (1 ROWS)
--[NOTE:PRODUCT TABLE,ADDRESS TABLE,ONLINE_CUSTOMER TABLE,ORDER_HEADER TABLE,ORDER_ITEMS TABLE,PRODUCT_CLASS TABLE]

--Solution:
SELECT product_class_desc, SUM(oi.product_quantity) AS total_qty,  
SUM(oi.product_quantity * p.product_price) AS total_value 
FROM Address a inner join Online_Customer oc on oc.address_id=a.address_id inner join 
Order_Header oh on oc.customer_id = oh.customer_id  
inner join Order_Items oi on oh.order_id = oi.order_id 
inner join Product p on oi.product_id = p.product_id 
inner join Product_class pc  
on p.product_class_code = pc.product_class_code 
WHERE a.country != 'India' 
AND a.country != 'USA' 
AND order_status = "shipped" 
GROUP BY product_class_desc 
ORDER BY total_qty DESC limit 1;
