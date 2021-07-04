
# Data Science Shopping

# Practical Project: Supermarket Analysis

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

There are a number of tasks demanded to the “Data Scientist”. You should provide your
solution to each one, along with a report explaining the most important insights/conclusions:

1. Provide a solution for the products that should be exposed in each shelf, in order to
    maximize the sales. The solution is simply a vector of 248 integers in a “csv” file
    (comma delimited), where each element is the ID of the product that should be
    exposed in the shelf given by the corresponding position in the list.

```
Example: 2, 5, 2, 3, ...
```
```
This solution states that:
```
```
o Product “2” should be exposed in the first shelf;
o Product “5” should be exposed in the second shelf;
o Product “2” should be exposed in the third shelf;
o Product “3” should be exposed in the fourth shelf;
o ...
```
2. Provide a solution for the products that should be exposed in each shelf, in order to
    maximize the profit of the supermarket.
3. Identify rules of the form “AàB” for items that are typically sold together,
    considering:

```
a. Support 1%;
b. Support 5%;
c. Support 10%;
d. Support 50%;
```
The rules should be given ordered by decreasing support value.

4. Produce a list of the top 100 clients, considering the total sales value.


5. Produce a list of the best 100 clients, considering the total profit that each client gave
    to the supermarket.

**Hints:**

- Consider the usual behavior of customers in a supermarket. Typically, each client
    has a “wish list” and traverses the supermarket according to the best path, to pick
    each one.
- Note that all clients know the positions of all products in the supermarket.
- When a customer picks a product, he looks to the surrounding products and “maybe”
    he picks it also, if he likes it.
- When moving along the corridors, each customer also looks for the available
    products in the shelfs, and even if they are not in his wish list, there is a certain
    probability of also buying them.
- The customers are only able to “look” in North/South/East/West directions of radius
    1, with respect to their current position.
- Often, after enough time, all clients feel tired and leave the supermarket, even if all
    products in their wish list were not bought.
- In case of equal costs between two paths [aàb], the used algorithm privileges the
    next indices to “a” with small values.

```
Example: supposing that [6, 8, 15, 19 21] and [6, 9, 16, 18, 21] refer to equal optimum
paths between vertices “6” and “21”. The used algorithm will follow the underlined
path (because 8<9).
```

