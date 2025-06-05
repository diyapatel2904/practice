
# Count how many times the letter 'a' appears in a user input string

user_input = input("Enter a string: ")
count = 0

for char in user_input:
    if char == 'a' or char == 'A':
        count += 1

print("Number of times 'a' appears:", count)

