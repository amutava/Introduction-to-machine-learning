# Introduction-to-machine-learning
## Data Science

Libraries :

 * Numpy
 The core data structure in NumPy is the ndarray object, which stands for N-dimensional array. An array is a collection of values, similar to a list. N-dimensional refers to the number of indices needed to select individual values from the object.
 If A is an array. A one dimensional array is accessed A[0], this is a one dimensional array. In a case B which is a two dimensional array is B[0][1], this is accessing an element in a two dimensional array.



A 1-dimensional array is often referred to as a `vector` while a 2-dimensional array is often referred to as a `matrix`. Both of these terms are both borrowed from a branch of mathematics called linear algebra. They're also often used in data science literature, so we'll use these words throughout this course.

To use NumPy, we first need to import it into our environment. NumPy is commonly imported using the alias np:

`import numpy as np`

We can directly construct arrays from lists using the numpy.array() function. To construct a vector, we need to pass in a single list (with no nesting):

`vector = np.array([5, 10, 15, 20])`

The `numpy.array()` function also accepts a list of lists, which we use to create a matrix:

`matrix = np.array([[5, 10, 15], [20, 25, 30], [35, 40, 45]])`

Arrays have a certain number of elements.

Matrices instead use rows and columns, which matches how we thought about datasets.

It's often useful to know how many elements an array contains. We can use the ndarray.shape property to figure out how many elements are in the array.
For vectors, the shape property contains a tuple with 1 element. A tuple is a kind of list where the elements can't be changed.

``` vector = numpy.array([1, 2, 3, 4])
    print(vector.shape)
```
The code above would result in the tuple (4,). This tuple indicates that the array vector has one dimension, with length 4, which matches our intuition that vector has 4 elements.

For matrices, the shape property contains a tuple with 2 elements.

``` 
    matrix = numpy.array([[5, 10, 15], [20, 25, 30]])
    print(matrix.shape)
```

The above code will result in the tuple (2,3) indicating that matrix has 2 rows and 3 columns.

We can read in datasets using the `numpy.genfromtxt()` function. Our dataset, world_alcohol.csv is a comma separated value dataset. We can specify the delimiter using the delimiter parameter:
```
import numpy
nfl = numpy.genfromtxt("data.csv", delimiter=",")
```
The above code would read in a file named data.csv file into a NumPy array. NumPy arrays are represented using the numpy.ndarray class. We'll refer to ndarray objects as NumPy arrays in our material.
Here are the first few rows of the dataset we'll be working with:

| Year | WHO Region | Country|Beverage Type |Display Value|
| -------- | ------------- | --------- |----|-------------|
| 1986 | West Pacific  | Viet Nam|Wine|0|
| 1986 | Americas | Uruguay|Wine|0.5|
| 1986 | Africa| Cote d'Ivoire|other|1.62|

Each row specifies how many liters of a type of alcohol each citizen of a country drank in a given year. The first row shows how many liters of wine an average person in Vietnam drank in 1986.

Here's what each column represents:

* Year -- the year the data in the row is for.
* WHO Region -- the region in which the country is located.
* Country -- the country the data is for.
* Beverage Types -- the type of beverage the data is for.
* Display Value -- the number of liters, on average, of the beverage type a citizen of the country drank in the given year.

Each value in a NumPy array has to have the same data type. NumPy data types are similar to Python data types, but have slight differences. You can find a full list of NumPy data types (here)[https://docs.scipy.org/doc/numpy-1.10.1/user/basics.types.html]. 

NumPy will automatically figure out an appropriate data type when reading in data or converting lists to arrays. You can check the data type of a NumPy array using the dtype property.

```
numbers = np.array([1, 2, 3, 4])
numbers.dtype

```
Because numbers only contains integers, its data type is int64.

Here's how NumPy represents the first few rows of the dataset:
```
array([[             nan,              nan,              nan,              nan,              nan],
       [  1.98600000e+03,              nan,              nan,              nan,   0.00000000e+00],
       [  1.98600000e+03,              nan,              nan,              nan,   5.00000000e-01]])
```
There are a few concepts we haven't been introduced to yet that we'll dive into:

* Many items in world_alcohol are `nan`, including the entire first row. `nan`, which stands for "not a number", is a data type used to represent missing values.
* Some of the numbers are written like 1.98600000e+03.

The data type of world_alcohol is `float`. Because all of the values in a NumPy array have to have the same data type, NumPy attempted to convert all of the columns to floats when they were read in. The numpy.genfromtxt() function will attempt to guess the correct data type of the array it creates.

In this case, the WHO Region, Country, and Beverage Types columns are actually strings, and couldn't be converted to floats. When NumPy can't convert a value to a numeric data type like float or integer, it uses a special `nan` value that stands for "not a number". NumPy assigns an `na` value, which stands for "not available", when the value doesn't exist.` nan` and `na` values are types of missing data. We'll dive more into how to deal with missing data in later missions.

The whole first row of world_alcohol.csv is a header row that contains the names of each column. This is not actually part of the data, and consists entirely of strings. Since the strings couldn't be converted to floats properly, NumPy uses nan values to represent them.

If you haven't seen (scientific notation) [https://en.wikipedia.org/wiki/Scientific_notation] before, you might not recognize numbers like `1.98600000e+03`. Scientific notation is a way to condense how very large or very precise numbers are displayed. We can represent 100 in scientific notation as `1e+02`. The `e+02` indicates that we should multiply what comes before it by 10 ^ 2(10 to the power 2, or 10 squared). This results in 1 * 100, or 100. Thus, `1.98600000e+03` is actually 1.986 * 10 ^ 3, or 1986. 1000000000000000 can be written as 1e+15.

In this case, `1.98600000e+03` is actually longer than 1986, but NumPy displays numeric values in scientific notation by default to account for larger or more precise numbers.

When reading in the data using the `numpy.genfromtxt()` function, we can use parameters to customize how we want the data to be read in. While we're at it, we can also specify that we want to skip the header row of world_alcohol.csv.

* To specify the data type for the entire NumPy array, we use the keyword argument dtype and set it to "U75". This specifies that we want to read in each value as a 75 byte unicode data type. We'll dive more into unicode and bytes later on, but for now, it's enough to know that this will read in our data properly.

* To skip the header when reading in the data, we use the skip_header parameter. The skip_header parameter accepts an integer value, specifying the number of lines from the top of the file we want NumPy to ignore.

Now that the data is in the right format, let's learn how to explore it. We can index NumPy arrays similarly to how we index regular Python lists. Here's how we would index a NumPy vector:

```
  vector = np.array([5, 10, 15, 20])
  print(vector[0])

```

The above code would print the first element of vector, or 5.

Indexing matrices is similar to indexing lists of lists. Here's a refresher on indexing lists of lists:

```
list_of_lists = [
        [5, 10, 15], 
        [20, 25, 30]
       ]

```

The first item in list_of_lists is [5, 10, 15]. If we wanted to access the element 15, we could do this:

```

first_item = list_of_lists[0]
first_item[2]
We could also condense the notation like this:

```

 list_of_lists[0][2]

```

We can index matrices in a similar way, but we place both indices inside square brackets. The first index specifies which row the data comes from, and the second index specifies which column the data comes from:

```
 matrix = np.array([
                        [5, 10, 15], 
                        [20, 25, 30]
                     ])
 matrix[1,2]
30

```

In the above code, we pass two indices into the square brackets when we index matrix.

We can use value slices to select subsets of arrays just like we can with lists:

```
  vector = np.array([5, 10, 15, 20])
  vector[0:3]
  array([ 5, 10, 15])

```

Like lists, vector slicing is from the first index up to but not including the second index. Matrix slicing is a bit more complex, and has four forms:

* When we want to select one entire dimension, and a single element from the other.
* When we want to select one entire dimension, and a slice of the other.
* When you want to select a slice of one dimension, and a single element from the other.
* When we want to slice both dimensions.

We'll dive into the first form in this screen. When we want to select one whole dimension, and an element from the other, we can do this:

```
 matrix = np.array([
                    [5, 10, 15], 
                    [20, 25, 30],
                    [35, 40, 45]
                 ])
matrix[:,1]
array([10, 25, 40])

```

This will select all of the rows, but only the column with index 1. The colon by itself : specifies that the entirety of a single dimension should be selected. Think of the colon as selecting from the first element in a dimension up to and including the last element.

When we want to select one whole dimension, and a slice of the other, we need to use special notation:

```
 matrix = np.array([
                    [5, 10, 15], 
                    [20, 25, 30],
                    [35, 40, 45]
                 ])
matrix[:,0:2]
array([[ 5, 10],
       [20, 25],
       [35, 40]])
```

We can select rows by specifying a colon in the columns area. The code below selects rows 1 and 2, and all of the columns.

```
matrix[1:3,:]
array([[20, 25, 30],
       [35, 40, 45]])
```

We can also select a single value along an entire dimension. The code belows selects rows 1 and 2 and column 1:

```
matrix[1:3,1]
array([25, 40])
```

We can also slice along both dimensions simultaneously. The following code selects rows with index 1 and 2, and columns with index 0 and 1:

```
matrix = np.array([
                    [5, 10, 15], 
                    [20, 25, 30],
                    [35, 40, 45]
                 ])
matrix[1:3,0:2]
array([[20, 25],
       [35, 40]])
```

One of the most powerful aspects of the NumPy module is the ability to make comparisons across an entire array. These comparisons result in Boolean values.

Here's an example of how we can do this with a vector:

```
vector = numpy.array([5, 10, 15, 20])
vector == 10
```

If you'll recall from an earlier mission, the double equals sign (==) compares two values. When used with NumPy, it will compare the second value to each element in the vector. If the value are equal, the Python interpreter returns True; otherwise, it returns False. It stores the Boolean results in a new vector.

For example, the code above will generate the vector `[False, True, False, False]`, since only the second element in vector equals `10`.

Here's an example with a matrix:

```
matrix = numpy.array([
                    [5, 10, 15], 
                    [20, 25, 30],
                    [35, 40, 45]
                 ])
matrix == 25

```

The final statement will compare 25 to every element in matrix. The result will be a matrix where elements are True or False:

```

[
    [False, False, False], 
    [False, True,  False],
    [False, False, False]
]

```

We mentioned that comparisons are very powerful, but it may not have been obvious why on the last screen. Comparisons give us the power to select elements in arrays using Boolean vectors. This allows us to conditionally select certain elements in vectors, or certain rows in matrices.

Here's an example of how we would do this with a vector:

```

vector = numpy.array([5, 10, 15, 20])
equal_to_ten = (vector == 10)

print(vector[equal_to_ten])

```

The code above:

Creates vector.
Compares vector to the value 10, which generates a Boolean vector [False, True, False, False]. It assigns the result to equal_to_ten.

Uses equal_to_ten to only select elements in vector where equal_to_ten is True. This results in the vector [10].

We can use the same principle to select rows in matrices:

```

matrix = numpy.array([
                [5, 10, 15], 
                [20, 25, 30],
                [35, 40, 45]
             ])
    second_column_25 = (matrix[:,1] == 25)
    print(matrix[second_column_25, :])

```

The code above:

* Creates matrix.
* Uses second_column_25 to select any rows in matrix where second_column_25 is True.
We end up with this matrix:

```

[
    [20, 25, 30]
]

```

We selected a single row from matrix, which was returned in a new matrix.

#### Comparison
On the last screen, we made comparisons based on a single condition. We can also perform comparisons with multiple conditions by specifying each one separately, then joining them with an ampersand (&). When constructing a comparison with multiple conditions, it's critical to put each one in parentheses.

Here's an example of how we would do this with a vector:
```
vector = numpy.array([5, 10, 15, 20])
equal_to_ten_and_five = (vector == 10) & (vector == 5)

```

In the above statement, we have two conditions, `(vector == 10)` and `(vector == 5)`. We use the ampersand `(&)` to indicate that both conditions must be `True` for the final result to be `True`. The statement returns `[False, False, False, False]`, because none of the elements can be `10` and `5` at the same time. Here's a diagram of the comparison logic:
We can also use the pipe symbol `(|)` to specify that either one condition or the other should be `True`:

```
vector = numpy.array([5, 10, 15, 20])
equal_to_ten_or_five = (vector == 10) | (vector == 5)
```
The code above will result in `[True, True, False, False]`.
We can also use comparisons to replace values in an array, based on certain conditions. Here's an example of how we would do this for a vector:
```vector = numpy.array([5, 10, 15, 20])
   equal_to_ten_or_five = (vector == 10) | (vector == 5)
   vector[equal_to_ten_or_five] = 50
   print(vector)
```
This code will complete the following steps:

* Create an array vector.
* Compare vector to `10` and `5`, and generate a vector that's True where vector is equal to either value.
* Select only the elements in vector where equal_to_ten_or_five is True.
* Replace the selected values with the value 50.
The result will be `[50, 50, 15, 20]`.

We can perform the same replacement on a matrix. To do this, we'll need to use indexing to select a column or row first:

```
matrix = numpy.array([
            [5, 10, 15], 
            [20, 25, 30],
            [35, 40, 45]
         ])
    second_column_25 = matrix[:,1] == 25
    matrix[second_column_25, 1] = 10
```
The code above will result in:
```
[
    [5, 10, 15], 
    [20, 10, 30],
    [35, 40, 45]
]
```
We'll soon be working with the Display Value column, which shows how much alcohol the average citizen of a country drinks. However, because world_alcohol currently has a unicode datatype, all of the values in the column are strings. To add these values together or perform any other mathematical operations on them, we'll have to convert the data in the column to floats.

Before we can do this, we need to address the empty string values ('') that appear where there was no original data for the given country and year. If we try to convert the data in the column to floats without removing these values first, we'll get a ValueError. Thankfully, we can remove these items using the replacement technique we learned on the last screen.

We can convert the data type of an array with the astype() method. Here's an example of how this works:
```vector = numpy.array(["1", "2", "3"])
   vector = vector.astype(float)
```
The code above will convert all of the values in vector to floats: [1.0, 2.0, 3.0].

We'll do something similar with the fifth column of world_alcohol, which contains information on how much alcohol the average citizen of a country drank in a given year. To determine which country drinks the most, we'll have to convert the values in this column to float values. That's because we can't add or perform calculations on these values while they're strings.

Now that alcohol_consumption consists of numeric values, we can perform computations on it. NumPy has a few built-in methods that operate on arrays. You can view all of them [in the documentation] (https://docs.scipy.org/doc/numpy-1.10.1/index.html). For now, here are a few important ones:

* sum() -- Computes the sum of all the elements in a vector, or the sum along a dimension in a matrix
* mean() -- Computes the average of all the elements in a vector, or the average along a dimension in a matrix
* max() -- Identifies the maximum value among all the elements in a vector, or the maximum along a dimension in a matrix
Here's an example of how we'd use one of these methods on a vector:
```
vector = numpy.array([5, 10, 15, 20])
vector.sum()
```
This would add together all of the elements in vector, and result in 50.
With a matrix, we have to specify an additional keyword argument, axis. The axis dictates which dimension we perform the operation on. 1 means that we want to perform the operation on each row, and 0 means on each column. The example below performs an operation across each row:
```
matrix = numpy.array([
                [5, 10, 15], 
                [20, 25, 30],
                [35, 40, 45]
             ])
    matrix.sum(axis=1)
```

Each country is associated with several rows for different types of beverages:
```[['1986', 'Americas', 'Canada', 'Other', ''],
   ['1986', 'Americas', 'Canada', 'Spirits', '3.11'],
   ['1986', 'Americas', 'Canada', 'Beer', '4.87'],
   ['1986', 'Americas', 'Canada', 'Wine', '1.33']]
```
To find the total amount the average person in Canada drank in 1986, for example, we'd have to add up all 4 of the rows shown above, then repeat this process for each country.

Now that we know how to calculate the average consumption of all types of alcohol for a single country and year, we can scale up the process and make the same calculation for all countries in a given year. Here's a rough process:

* Create an empty dictionary called totals.
* Select only the rows in world_alcohol that match a given year. Assign the result to year.
* Loop through a list of countries. For each country:
* Select only the rows from year that match the given country.
* Assign the result to country_consumption.
* Extract the fifth column from country_consumption.
* Replace any empty string values in the column with the string 0.
* Convert the column to the float data type.
* Find the sum of the column.
* Add the sum to the totals dictionary, with the country name as the key.
* After the code executes, you'll have a dictionary containing all of the country names as keys, with the associated alcohol consumption totals as the values.

Now that we've computed total alcohol consumption for each country in 1989, we can loop through the totals dictionary to find the country with the highest value.

The process we've outlined below will help you find the key with the highest value in a dictionary:

* Create a variable called highest_value that will keep track of the highest value. Set its value to 0.
* Create a variable called highest_key that will keep track of the key associated with the highest value. Set its value to None.
* Loop through each key in the dictionary.
- If the value associated with the key is greater than highest_value, assign the value to highest_value, and assign the key to highest_key.
* After the code runs, highest_key will be the key associated with the highest value in the dictionary.

# PANDAS

Pandas is a library that unifies the most common workflows that data analysts and data scientists previously relied on many different libraries for. Pandas has quickly became an important tool in a data professional's toolbelt and is the most popular library for working with tabular data in Python. Tabular data is any data that can be represented as rows and columns. The CSV files we've worked with in previous missions are all examples of tabular data.

To represent tabular data, pandas uses a custom data structure called a dataframe. A dataframe is a highly efficient, 2-dimensional data structure that provides a suite of methods and attributes to quickly explore, analyze, and visualize data. The dataframe is similar to the NumPy 2D array but adds support for many features that help you work with tabular data.

One of the biggest advantages that pandas has over NumPy is the ability to store mixed data types in rows and columns. Many tabular datasets contain a range of data types and pandas dataframes handle mixed data types effortlessly while NumPy doesn't. Pandas dataframes can also handle missing values gracefully using a custom object, NaN, to represent those values. A common complaint with NumPy is its lack of an object to represent missing values and people end up having to find and replace these values manually. In addition, pandas dataframes contain axis labels for both rows and columns and enable you to refer to elements in the dataframe more intuitively. Since many tabular datasets contain column titles, this means that dataframes preserve the metadata from the file around the data.

In this mission, you'll learn the basics of pandas while exploring a dataset from the [United States Department of Agriculture (USDA)] (http://www.ars.usda.gov/Services/docs.htm?docid=8964). This dataset contains nutritional information on the most common foods Americans consume. Each column in the dataset shows a different attribute of the foods and each row describes a different food item.

Here are some of the columns in the dataset:

* NDB_No - unique id of the food.
* Shrt_Desc - name of the food.
* Water_(g) - water content in grams.
* Energ_Kcal - energy measured in kilo-calories.
* Protein_(g) - protein measured in grams.
* Cholestrl_(mg) - cholesterol in milligrams.

To use the Pandas library, we need to import it into the environment using the import keyword:
We can then refer to the module using pandas and use dot notation to call its methods. To read a CSV file into a dataframe, we use the pandas.read_csv() function and pass in the file name as a string:

Now that we've read the dataset into a dataframe, we can start using the dataframe methods to explore the data. To select the first 5 rows of a dataframe, use the dataframe method head(). When you call the head() method, pandas will return a new dataframe containing just the first 5 rows:

`first_rows = food_info.head()`

If you peek at the [documentation] (http://pandas.pydata.org/pandas-docs/version/0.17.1/generated/pandas.DataFrame.head.html), you'll notice that you can pass in an integer (n) into the head() method to display the first n rows instead of the first 5:

```# First 3 rows.
print(food_info.head(3))```

Because this dataframe contains many columns and rows, pandas uses ellipsis (...) to hide the columns and rows in the middle. Only the first few and the last few columns and rows are displayed to conserve space.
To access the full list of column names, use the columns attribute:
`column_names = food_info.columns`
Lastly, you can use the shape attribute to understand the dimensions of the dataframe. The shape attribute returns a tuple of integers representing the number of rows followed by the number of columns:

```
# Returns the tuple (8618,36) and assigns to `dimensions`.
dimensions = food_info.shape
# The number of rows, 8618.
num_rows = dimensions[0]
# The number of columns, 36.
num_cols = dimensions[1]

```
When you read in a file into a dataframe, pandas uses the values in the first row (also known as the header) for the column labels and the row number for the row labels. Collectively, the labels are referred to as the index. dataframes contain both a row index and a column index. Here's a diagram that displays some of the column and row labels for food_info:

The Series object is a core data structure that pandas uses to represent rows and columns. A Series is a labelled collection of values similar to the NumPy vector. The main advantage of Series objects is the ability to utilize non-integer labels. NumPy arrays can only utilize integer labels for indexing.

Pandas utilizes this feature to provide more context when returning a row or a column from a dataframe. For example, when you select a row from a dataframe, instead of just returning the values in that row as a list, pandas returns a Series object that contains the column labels as well as the corresponding values:

While we use bracket notation to access elements in a NumPy array or a standard list, we need to use the pandas method loc[] to select rows in a dataframe. The loc[] method allows you to select rows by row labels. Recall that when you read a file into a dataframe, pandas uses the row number (or position) as each row's label. Pandas uses zero-indexing, so the first row is at index 0, the second row at index 1, and so on.
```
# Series object representing the row at index 0.
food_info.loc[0]

# Series object representing the seventh row.
food_info.loc[6]

# Will throw an error: "KeyError: 'the label [8620] is not in the [index]'"
food_info.loc[8620]
```
When accessing an individual row, pandas returns a Series object containing the column names and that row's value for each column. In the following code cell, we select the first and seventh rows and display them using the print() function.

When you displayed individual rows, represented as Series objects, you may have noticed the text "dtype: object" after the last value. dtype: object refers to the data type, or dtype, of that Series. The object dtype is equivalent to the string type in Python. Pandas borrows from the NumPy type system and contains the following dtypes:

* object - for representing string values.
* int - for representing integer values.
* float - for representing float values.
* datetime - for representing time values.
* bool - for representing Boolean values.

When reading a file into a dataframe, pandas analyzes the values and infers each column's types. To access the types for each column, use the DataFrame.dtypes attribute to return a Series containing each column name and its corresponding type. Read more about data types on the Pandas [documentation] (http://pandas.pydata.org/pandas-docs/stable/basics.html#dtypes).

If you're interested in accessing multiple rows of the dataframe, you can pass in either a slice of row labels or a list of row labels and pandas will return a dataframe. Note that unlike slicing lists in Python, a slice of a dataframe using .loc[] will include both the start and the end row:

```
# DataFrame containing the rows at index 3, 4, 5, and 6 returned.
food_info.loc[3:6]

# DataFrame containing the rows at index 2, 5, and 10 returned. Either of the following work.
# Method 1
two_five_ten = [2,5,10] 
food_info.loc[two_five_ten]

# Method 2
food_info.loc[[2,5,10]]
```
When accessing a column in a dataframe, pandas returns a Series object containing the row label and each row's value for that column. To access a single column, use bracket notation and pass in the column name as a string.

```
# Series object representing the "NDB_No" column.
ndb_col = food_info["NDB_No"]

# You can instead access a column by passing in a string variable.
col_name = "NDB_No"
ndb_col = food_info[col_name]
```

To select multiple columns, pass in a list of strings representing the column names and pandas will return a dataframe containing only the values in those columns. The following code returns a dataframe containing the "Zinc_(mg)" and "Copper_(mg)" columns, in that order:
```
columns = ["Zinc_(mg)", "Copper_(mg)"]
zinc_copper = food_info[columns]

# Skipping the assignment.
zinc_copper = food_info[["Zinc_(mg)", "Copper_(mg)"]]
```
When selecting multiple columns, the order of the columns in the returned dataframe matches the order of the column names in the list of strings that you passed in. This allows you to easily explore specific columns that may not be positioned next to each other in the dataframe.

##### Exercise
* Select and display only the columns that use grams for measurement (that end with "(g)"). To accomplish this:
* Use the columns attribute to return the column names in food_info and convert to a list by calling the method tolist()
* Create a new list, gram_columns, containing only the column names that end in "(g)". The string method endswith() returns True if the string object calling the method ends with the string passed into the parentheses.
* Pass gram_columns into bracket notation to select just those columns and assign the resulting dataframe to gram_df
* Then use the dataframe method head() to display the first 3 rows of gram_df.
##### Answer

```
  print(food_info.columns)
  print(food_info.head(2))
  col_names = food_info.columns.tolist()
  gram_columns = []

  for c in col_names:
      if c.endswith("(g)"):
          gram_columns.append(c)
  gram_df = food_info[gram_columns]
  print(gram_df.head(3))
  
```


