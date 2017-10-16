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
