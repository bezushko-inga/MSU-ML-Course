def hello(x = None):
    if not x:
        return "Hello!"
    else:
        return f"Hello, {x}!"
        
def int_to_roman(n):
    roman = ""
    int_to_rom = [[1000, "M"], [900, "CM"], [500, "D"], [400, "CD"],
        [100, "C"], [90, "XC"], [50, "L"], [40, "XL"],
        [10, "X"], [9, "IX"], [5, "V"], [4, "IV"], [1, "I"]]
    i = 0
    while n > 0:
        if n >= int_to_rom[i][0]:
            roman += int_to_rom[i][1]
            n -= int_to_rom[i][0]
        else:
            i += 1
    return roman

def longest_common_prefix(x):
    y = [word.lstrip() for word in x]
    prefix = ""
    if x:
        first = y[0]
        for i in range(len(first)):
            char = first[i]
            for word in range(1, len(y)):
                if i >= len(y[word]):
                    return prefix
                if y[word][i] != char:
                    return prefix
            prefix += char
    return prefix

class BankCard:
    def __init__(self,total_sum, balance_limit = None):
        self.total_sum = total_sum
        self.balance_limit = balance_limit
    def __call__(self, sum_spent):
        if sum_spent > self.total_sum:
            raise ValueError("Not enough money to spend sum_spent dollars.")
        self.total_sum -= sum_spent
        print("You spent ", sum_spent, " dollars.")
    def __str__(self):
        return "To learn the balance call balance."
    def __add__(self, card):
        return BankCard(self.total_sum + card.total_sum, max(self.balance_limit, card.balance_limit))
    def put(self, sum_put):
        self.total_sum += sum_put
        print("You put ", sum_put, " dollars.")
    @property
    def balance(self):
        if self.balance_limit is None:
            return self.total_sum
        if self.balance_limit == 0:
            raise ValueError("Balance check limits exceeded.")
        else:
            self.balance_limit -= 1
            return self.total_sum

def primes():
    n = 2
    while True:
        ok = True
        for i in range(2, int(n**0.5) + 1):
            ok = n % i != 0
            if not ok: break
        if ok: yield n;
        n += 1
