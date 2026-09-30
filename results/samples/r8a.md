# r8a samples

Base: `models/q8_g64` · adapter: `adapters/r8a_q8_l16_seq1024` (unfused) · greedy, max_tokens=256, seed=0

## 1. Write a Python function that checks whether a string is a palindrome, ignoring case and spaces.

```
def is_palindrome(string):
    string = string.lower().replace(" ", "")
    return string == string[::-1]
```

## 2. Write a Python function that returns the n-th Fibonacci number using iteration.

```
def fibonacci(n):
	a, b = 0, 1
	for i in range(n):
		a, b = b, a + b
	return a
```

## 3. Write a Python function that merges two sorted lists into one sorted list.

```
def merge_sorted_lists(list1, list2):
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
    merged_list += list1[i:]
    merged_list += list2[j:]
    return merged_list
```

## 4. Write a Python function that counts the frequency of each word in a sentence and returns a dictionary.

```
def count_words(sentence):
    words = sentence.split()
    word_freq = {}
    for word in words:
        if word in word_freq:
            word_freq[word] += 1
        else:
            word_freq[word] = 1
    return word_freq
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
    new_list = []
    for item in lst:
        if item not in new_list:
            new_list.append(item)
    return new_list
```

## 7. Write a Python function that reads a CSV file and returns the average of a given numeric column.

```
def read_csv(filename):
    with open(filename, 'r') as f:
        reader = csv.reader(f)
        data = list(reader)
        return sum([float(row[0]) for row in data]) / len(data)
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
def flatten(nested_list):
    flattened_list = []
    for item in nested_list:
        if isinstance(item, list):
            flattened_list.extend(flatten(item))
        else:
            flattened_list.append(item)
    return flattened_list
```

## 10. Write a Python function that validates an email address using a regular expression.

```
import re

def validate_email(email):
    pattern = re.compile(r"^[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+$")
    return pattern.match(email)
```
