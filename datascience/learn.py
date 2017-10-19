import numpy as np

world_alcohol = np.genfromtxt(
    'world_alcohol.csv', dtype="U75", delimiter=",", skip_header=1)
world_alcohol_dtype = world_alcohol.dtype
print(world_alcohol)

countries_canada = (world_alcohol[:, 2] == "Canada")
years_1984 = (world_alcohol[:, 0] == "1984")

country_is_algeria = world_alcohol[:, 2] == "Algeria"
country_algeria = world_alcohol[country_is_algeria, :]

is_algeria_and_1986 = (world_alcohol[:, 0] == '1986') & (
    world_alcohol[:, 2] == 'Algeria')
rows_with_algeria_and_1986 = world_alcohol[is_algeria_and_1986, :]

world_alcohol[:, 0][world_alcohol[:, 0] == '1986'] = '2014'
world_alcohol[:, 3][world_alcohol[:, 3] == 'Wine'] = 'Grog'

is_value_empty = world_alcohol[:, 4] == ''
world_alcohol[is_value_empty, 4] = '0'

alcohol_consumption = world_alcohol[:, 4]
alcohol_consumption = alcohol_consumption.astype(float)

total_alcohol = alcohol_consumption.sum()
average_alcohol = alcohol_consumption.mean()

is_canada_1986 = (world_alcohol[:, 2] == "Canada") & (
    world_alcohol[:, 0] == '1986')
canada_1986 = world_alcohol[is_canada_1986, :]
canada_alcohol = canada_1986[:, 4]
empty_strings = canada_alcohol == ''
canada_alcohol[empty_strings] = "0"
canada_alcohol = canada_alcohol.astype(float)
total_canadian_drinking = canada_alcohol.sum()

totals = {}
is_year = world_alcohol[:, 0] == "1989"
year = world_alcohol[is_year, :]

for country in countries:
    is_country = year[:, 2] == country
    country_consumption = year[is_country, :]
    alcohol_column = country_consumption[:, 4]
    is_empty = alcohol_column == ''
    alcohol_column[is_empty] = "0"
    alcohol_column = alcohol_column.astype(float)
    totals[country] = alcohol_column.sum()

highest_value = 0
highest_key = None
for country in totals:
    consumption = totals[country]
    if highest_value < consumption:
        highest_value = consumption
        highest_key = country
