
# Data Science Shopping

# Supermarket Analysis

Suppose that the supermarket “ _The Best_ ” hired you to analyze their customer and sales data
and suggest different configurations for the products in the shelfs, so to maximize the profits
of the supermarket.

This exercise regards the “ _Data Science Hands-on Project_ ” of the Data Science course and
it is intended to cover the most important topics lectured in the whole course itself.

Consider the following physical configuration of the supermarket, where the positions
shaded in gray correspond to the shelfs where the products should be shown, the positions
in white are the corridors, while the green/red positions denote the supermarket entrance/exit
doors.

The supermarket sells a total of 165 different products, that should be exposed in the 2 48
shelfs available. For each product you will have the following information, given in the
“Products.txt” file. The ID of the product corresponds to its corresponding position in this
file, starting in “1”.

- Family: integer identifying the group where the product belongs;
- Name: designation of the product;


- Price: product price;
- Profit margin: profit to the supermarket (%) when the corresponding product is sold;
- Total shelfs: Number of shelfs that should be occupied by the product;
- Prior Probability: prior probability for selling the corresponding product. It is given
    by the difference between the value in each row and the value immediately above (
    for the first row).

