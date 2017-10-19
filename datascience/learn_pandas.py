import pandas

food_info = pandas.read_csv("food_info.csv")
print(type(food_info))
print(food_info.head(3))
dimensions = food_info.shape
print(dimensions)
num_rows = dimensions[0]
print(num_rows)
num_cols = dimensions[1]
print(num_cols)

# prints the first twenty rows. 
first_twenty = food_info.head(20)

#prints the hundredth row.
hundredth_row = food_info.loc[99]
print(hundredth_row)

# print the last 5 items in the data
print("Rows 3, 4, 5 and 6")
print(food_info.loc[3:6])

print("Rows 2, 5, and 10")
two_five_ten = [2,5,10]
print(food_info.loc[two_five_ten])

num_rows = food_info.shape[0]
last_rows = food_info.loc[num_rows-5:num_rows-1]

# prints the column in the range.
# Series object.
ndb_col = food_info["NDB_No"]
print(ndb_col)

# Display the type of the column to confirm it's a Series object.
print(type(ndb_col))

saturated_fat = food_info["FA_Sat_(g)"]
cholesterol = food_info["Cholestrl_(mg)"]

# selecting two columns.
zinc_copper = food_info[["Zinc_(mg)", "Copper_(mg)"]]

columns = ["Zinc_(mg)", "Copper_(mg)"]
zinc_copper = food_info[columns]

selenium_thiamin = food_info[["Selenium_(mcg)", "Thiamin_(mg)"]]