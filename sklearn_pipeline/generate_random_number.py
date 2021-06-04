import random
import string

random_float_list = []
random_string_list = []
random_numeric_list = []
for i in range(0, 10):
    x = round(random.uniform(50.50, 500.50), 2)
    random_float_list.append(x)
    length=10
    result = ''.join(
        (random.choice(string.ascii_lowercase) for x in range(length)))  # run loop until the define length
    random_string_list.append(result)
    y = round(random.uniform(100, 4000))
    random_numeric_list.append(y)

print(random_float_list, random_string_list, random_numeric_list)




# def Upper_Lower_string(length):  # define the function and pass the length as argument
#     # Print the string in Lowercase
#     result = ''.join(
#         (random.choice(string.ascii_lowercase) for x in range(length)))  # run loop until the define length
#     print(" Random string generated in Lowercase: ", result)
#
#     # Print the string in Uppercase
#     result1 = ''.join(
#         (random.choice(string.ascii_uppercase) for x in range(length)))  # run the loop until the define length
#     print(" Random string generated in Uppercase: ", result1)
#
#
# Upper_Lower_string(10)  # define the length