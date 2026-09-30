# r8b samples

Base: `models/q8_g64` · adapter: `adapters/r8b_q8_l8_seq512` (unfused) · greedy, max_tokens=256, seed=0

## 1. Write a Python function that checks whether a string is a palindrome, ignoring case and spaces.

```
def is_palindrome(string):
    # Write code here
    return True
```

## 2. Write a Python function that returns the n-th Fibonacci number using iteration.

```
def fibonacci(n):
    a = 0
    b = 1
    for i in range(n):
        a, b = b, a + b
    return a
```

## 3. Write a Python function that merges two sorted lists into one sorted list.

```
def merge_lists(list1, list2):
    merged_list = []
    i = 0
    j = 0
    while i < len(list1) and j < len(list2):
        if list1[i] < list2[j]:
            merged_list.append(list1[i])
            i += 1
        else:
            merged_list.append(list2[j])
            j += 1
    if i == len(list1):
        merged_list += list2[j:]
    else:
        merged_list += list1[i:]
    return merged_list
```

## 4. Write a Python function that counts the frequency of each word in a sentence and returns a dictionary.

```
def count_words(sentence):
    words = sentence.split()
    counts = {}
    for word in words:
        if word in counts:
            counts[word] += 1
        else:
            counts[word] = 1
    return counts
```

## 5. Write a Python class for a stack with push, pop, peek, and is_empty methods.

```
class Stack:
    def __init__(self):
        self.items = []

    def push(self, item):
        self.items.append(item)

    def pop(self):
        return self.items.pop()

    def peek(self):
        return self.items[-1]

    def is_empty(self):
        return self.items == []
```

## 6. Write a Python function that removes duplicates from a list while preserving order.

```
def remove_duplicates(lst):
    new_lst = []
    for item in lst:
        if item not in new_lst:
            new_lst.append(item)
    return new_lst
```

## 7. Write a Python function that reads a CSV file and returns the average of a given numeric column.

```
def average_column(filename, column):
    """
    Reads a CSV file and returns the average of a given numeric column.
    """
    with open(filename) as f:
        reader = csv.reader(f)
        next(reader)
        total = 0
        count = 0
        for row in reader:
            total += float(row[column])
            count += 1
        return total / count
```

## 8. Write a Python function that checks whether a number is prime.

```
def is_prime(n):
    if n < 2:
        return False
    for i in range(2, n):
        if n % i == 0:
            return False
    return True
```

## 9. Write a Python function that flattens a nested list of arbitrary depth.

```
def flatten(lst):
    new_lst = []
    for i in lst:
        if type(i) == list:
            new_lst += flatten(i)
        else:
            new_lst.append(i)
    return new_lst
```

## 10. Write a Python function that validates an email address using a regular expression.

```
def validate_email(email):
    regex = r"^[a-zA-Z0-9]+@[a-zA-Z0-9]+\.[a-zA-Z0-9]+$"
    if re.match(regex, email):
        return True
    else:
        return False
```
