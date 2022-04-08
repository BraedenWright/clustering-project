Project Goal
---------------------


    We want to be able to predict the property tax assessed values ('taxvaluedollarcnt') of Single Family Properties that had a transaction during 2017.

    We have a model already, but we are hoping your insights can help us improve it. I need recommendations on a way to make a better model. Maybe you will create a new feature out of existing ones that works better, try a non-linear regression algorithm, or try to create a different model for each county. Whatever you find that works (or doesn't work) will be useful. Given you have just joined our team, we are excited to see your outside perspective.

    One last thing, Zach lost the email that told us where these properties were located. Ugh, Zach :-/. Because property taxes are assessed at the county level, we would like to know what states and counties these are located in.

-- The Zillow Data Science Team



    Find the key drivers of property value for single family properties and construct an ML Regression model that will predict property tax assessed values (tax_value) for those homes.

Initial Questions

    Does more square footage have a linear relation to tax value, as I suspect it to?
    Would the year the house was built positively or negatively affect tax value?
    Will the geographical location(fips) be useful, or does it skew the data?




Initial Questions
---------------------


    Does more square footage have a linear relation to tax value, as I suspect it to?
    
    Would the year the house was built positively or negatively affect tax value?
    
    Will the geographical location(fips) be useful, or does it skew the data?





Data Dictionary
---------------------

 	        Feature 	                           Description
3 	'bathroomcnt'               	Number of bathrooms in home including fractio...
4 	'bedroomcnt' 	                Number of bedrooms in home
10 	'finishedfloor1squarefeet'   	Size of the finished living area on the first...
17 	'fips'                  	    Federal Information Processing Standard code ...
50 	'yearbuilt' 	                The Year the principal residence was built
51 	'taxvaluedollarcnt'         	The total tax assessed value of the parcel
54 	'taxamount' 	                The total property tax assessed for that asses...



