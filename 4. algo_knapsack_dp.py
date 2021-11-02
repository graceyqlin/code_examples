# Q1. Description
# Given n items with an integer m denotes the size of a backpack. How full you can fill this backpack?

#example: input:
# array = [3,4,8,5]
# backpack size = 10

# output: 9 ( Load 4 and 5.)

def get_max(input_array, backpack):
  if backpack == 0:
    return 0
  if len(input_array) == 0:
    return 0

  n = len(input_array)
  m = backpack
  dp = [[0] * (backpack + 1) for _ in range(n+1)]

  dp[0][0] = 1 # i don't know why the initialization is 1

  for row in range(1, n+1):
    for col in range(1, m+1):
      if input_array[row-1] <= col:
        dp[row][col] = max(dp[row-1][col - input_array[row-1]] + input_array[row-1], dp[row-1][col])
      else:
        dp[row][col] = dp[row-1][col]
  print(dp)
  return dp[-1][-1]

input_array = [3, 4, 8, 5]
backpack = 10

print(get_max(input_array, backpack))




#Q2. There are n items and a backpack with size m. 
# Given array A representing the size of each item and array V representing the value of each item.
# What's the maximum value can you put into the backpack?

# example input
# m = 10
# A = [2, 3, 5, 7]
# V = [1, 5, 2, 4]

# example output :9
# dp[i][j]: max value of using first i items and total weight is exact j

def get_max_value(m, A, V):
  dp = [[0] * (m+1) for _ in range(len(A) + 1)]
  dp[0][0] = 0
  for row in range(len(A) + 1):
    for col in range(m+1):
      if A[row-1] <= col:
        dp[row][col] = max(dp[row-1][col - A[row-1]] + V[row-1], dp[row-1][col])
      else:
        dp[row][col] = dp[row-1][col]
  print(dp)

  return dp[-1][-1]


m = 10
A = [2, 3, 5, 7]
V = [1, 5, 2, 4]

print(get_max_value(m, A, V))

