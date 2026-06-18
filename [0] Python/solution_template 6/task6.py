def check(s, filename):
    words = s.lower().split(" ")
    dictionary = {}
    for word in words:
        if word in dictionary:
            dictionary[word] += 1
        else:
            dictionary[word] = 1
    with open(filename, "w") as f:
        for word in sorted(dictionary.keys()):
            f.write(f"{word} {dictionary[word]}\n")
