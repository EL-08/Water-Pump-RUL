
def kadane(arr):
    max_sum = arr[0]
    curr = arr[0]
    start = end = s = 0

    for i in range(1,len(arr)):
        if curr + arr[i] < arr[i]:
            curr = arr[i]
            s = i
        else:
            curr += arr[i]

        if curr > max_sum:
            max_sum = curr
            start = s
            end = i

    return start,end,max_sum
