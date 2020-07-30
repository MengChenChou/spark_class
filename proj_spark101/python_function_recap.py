from urllib.request import urlopen


# define a function
def fetch_words(url):
    page = urlopen(url)
    words = []
    for line in page:
        line_words = line.decode('utf-8').split()
        for word in line_words:
            words.append(word)
    return words


# use the function
fetch_words("https://google.com")

# functions are objects
type(fetch_words)

# function objects can be treated as other data objects
# re-assign the function object to fetch_words2
fetch_words2 = fetch_words

type(fetch_words2)


# take an function as an argument
def fetcher(target, fetch_function):
    return fetch_function(target)


fetcher("https://google.com", fetch_words)


# anonymous function
def get_first_name(full_name):
    return full_name.split()[0]


get_first_name("Peter Parker")

# define a lambda function (anonymous function)
# body is a single expression
# No return statement is needed. the result of the expression will be returned automatically.
lambda full_name: full_name.split()[0]

my_function = lambda full_name: full_name.split()[0]

my_function("Peter Parker")

# sorted() built-in sort function
characters = ['b', 'c', 'D', 'A', 'B', 'd']

sorted(characters)
sorted(characters, reverse=True)
sorted(characters, key=lambda c: c.lower())
sorted(characters, key=lambda c: c.lower(), reverse=True)


students = [
    ('john', 'A', 15),
    ('kevin', 'B', 12),
    ('dave', 'B', 10),
]

sorted(students)
sorted(students, key=lambda student: student[2])   
