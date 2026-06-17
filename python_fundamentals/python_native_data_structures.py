###############################################
#                                             #
#          DICTIONARY / DEFAULTDICT           #
#                                             #
###############################################


"""
Problem 1 - Find longest duration user:
You are given a list of log entries. Each log entry is a string in the format:

"user_id action timestamp"

logs = [
    "u1 login 100",
    "u2 login 200",
    "u1 logout 300",
    "u3 login 400",
    "u2 logout 500"
]

Write a function longest_logged_in(logs) that returns the user who was logged in for the longest total duration.

Assumptions:
Logs are NOT guaranteed to be sorted by timestamp
A user may log in and out multiple times
Every login has a corresponding logout
Input is space-delimited and always complete
Timestamps are integers

Requirements:
Solve in pure Python — no pandas
Use appropriate data structures from collections if needed
Handle multiple sessions per user
Handle unsorted logs


Follow-ups to answer in comments after your solution:
Write 3 test cases covering: basic, multiple sessions, unsorted logs
What is the time complexity?
What is the space complexity?
"""

# Solution:
# When logs are not guaranteed to be sorted
from collections import defaultdict
def longest_logged_in(logs):
  events = []
  for log in logs:
    uid, action, time = log.split() # split is by default with white space
    events.append((int(time), action, uid)) # putting time first to sort by default .sort() - in_place
  events.sort() # sort the logs first and in this case by first item i.e. timestamp
  
  login_times = {} # No defaultdict as we can rely on existence of login times due to sorting
  durations = defaultdict(int)
  for time, action, uid in events:
    if action == 'login':
      login_times[uid] = time
    if action == 'logout':
      start_time = login_times.pop(uid) # sort guarantees a login_item and pop also removes it.
      durations[uid] += time - start_time
   return max(durations, key=durations.get)

# Complexity:
# Time: O(n logn) due to sorting
# Space: O(n)

"""
Problem 2 - Find peak concurrent users:
You are given a list of login/logout logs. Each log entry is a string in the format:

"user_id action timestamp"

logs = [
    "u1 login 100",
    "u2 login 200",
    "u3 login 250",
    "u1 logout 300",
    "u2 logout 400",
    "u3 logout 500"
]

Write a function peak_concurrent_users(logs) that returns the maximum number of users logged in at the same time.

Assumption:
- Logs are NOT guaranteed to be sorted
- Every login has a corresponding logout
- A user cannot have overlapping sessions with themselves
- Timestamps are integers
"""

# Solution:
def peak_concurrent_users(logs):
  events = []
  for log in logs:
    uid, action, time = log.split()
    if action == "login":
      priority = 1
    elif action == "logout":
      priority = 0
    events.append((int(time), priority, action, uid))
    
  # events = sorted(events, key=lambda x: (x[0], x[1]) ) # sorted logs by timestamp and action coz if two users one logs in and another logs out at the same time which one to process, logout is preferred to avoid double counting. Now simply events.sort() will also work as for same-timestamp tie breaking logout i.e. 0 will be prioritized.
  events.sort()
  active_users = 0 # current number of logged in users.
  max_active = 0 # highest value active_users ever achieved at any given timestamp
  for time, priority, action, uid in events:
    if action == "login":
      active_users += 1
      max_active = max(max_active, active_users)
    elif action == "logout":
      active_users -= 1
  return max_active

# Complexity:
# Time: O(n logn) (due to sorting)
# Space: O(n)
# If the logs were already sorted by timestamp, the time complexity would be O(n), and we could avoid storing all events if we process them directly.

"""
Problem 3 - Return active user count after each event
You are given a list of login/logout logs. Each log entry is a string in the format:

"user_id action timestamp"

logs = [
    "u1 login 100",
    "u2 login 200",
    "u1 logout 300",
    "u3 login 400",
    "u2 logout 500"
]

Write a function active_users_over_time(logs) that returns a list of tuples showing the number of active users after each event.

[
    (100, 1),
    (200, 2),
    (300, 1),
    (400, 2),
    (500, 1)
]

Assumptions:
- Logs may be unsorted
- Every login has a matching logout
- Timestamps are integers
- If multiple events happen at the same timestamp, process logout before login
"""
# Solution:
def active_users_over_time(logs):
  events = []  
  for log in logs:
    uid, action, time = log.split()
    if action == "login":
      priority = 1
    elif action == "logout":
      priority = 0
    events.append((int(time), priority, action, uid))
  events.sort()
  active_users = 0
  active_users_by_event = []
  for time, priority, action, uid in events:
    if action == "login":
      active_users += 1
    elif action == "logout":
      active_users -= 1
    active_users_by_event.append((time, active_users))
  return active_users_by_event

# Complexity:
# Time: O(n logn) (due to sorting)
# Space: O(n)
# If the logs were already sorted by timestamp, the time complexity would be O(n), and we could avoid storing all events if we process them directly.

# Note: One nuance: if several events happen at the same timestamp, this function will return multiple entries with the same timestamp, one after each processed event. That matches the wording “after each event.” If the interviewer wanted one value per timestamp, you would need to aggregate events by timestamp first.

"""
Problem 4 - Find users with too many failed logins
You are given login attempt logs. Each log entry is a string in the format:

"user_id status timestamp"

where status is either "success" or "fail".

logs = [
    "u1 fail 100",
    "u1 fail 200",
    "u1 fail 500",
    "u2 fail 100",
    "u2 fail 800",
    "u2 fail 1500",
    "u3 success 200"
]

Write a function users_with_failed_logins(logs) that returns users who had at least 3 failed login attempts within a 600-second window.

Assumptions:
- Logs may be unsorted
- Only failed attempts count
- Success attempts do not count as failures
- Timestamps are integers
- Return each user only once
- Output order does not matter
"""

# Solution:
from collections import defaultdict
def users_with_failed_logins(logs):
  failed_logins = defaultdict(list)  
  for log in logs:
    uid, status, time = log.split()
    time = int(time)
    if status == "fail":
      failed_logins[uid].append(time)
  failed_logins = {
uid: sorted(times) 
for uid, times in failed_logins.items() 
if len(times) >= 3
} # this filtering could also be done within the loop below but its fine here as well
  result = []
  for uid, times in failed_logins.items():
    for i in range(len(times) - 2): # stopping initial boundary before -2 so we only iterate on full windows else it will throw out of bound errors.
      if times[i+2] - times[i] <= 600:
        result.append(uid)
        break
  return result

# Complexity:
# Time: O(n logn) (due to sorting per user)
# Space: O(n) as we store failed attempts
# If the logs were already sorted by timestamp, the time complexity would be O(n), and we could avoid storing all events if we process them directly.

"""
Problem 5 - Sessionize user logs
You are given login/logout logs. Each log entry is a string in the format:

logs = [
    "u1 logout 300",
    "u1 login 100",
    "u2 login 200",
    "u2 logout 500",
    "u1 login 600",
    "u1 logout 900"
]
Write a function get_user_sessions(logs) that returns each user’s sessions as intervals. 
{ 
"u1": [(100, 300), (600, 900)], 
"u2": [(200, 500)]
 }

Assumptions:
Logs may be unsorted
Every login has a corresponding logout
A user may have multiple sessions
A user cannot have overlapping sessions with themselves
Timestamps are integers
"""

# Solution:
from collections import defaultdict
def get_user_sessions(logs):
  events = []
  for log in logs:
    uid, action, time = log.split()
    events.append((int(time), action, uid))
  events.sort()
  user_sessions = defaultdict(list)
  login_times = {}
  for time, action, uid in events:
    if action == "login":
      login_times[uid] = time
    elif action == "logout":
      start_time = login_times.pop(uid)
      session = (start_time, time)
      user_sessions[uid].append(session)
  return dict(user_sessions)

# Complexity:
# Time: O(n logn) due to sorting all events globally if sorted already then O(n)
# Space: O(n) as we store all events and output sessions during sorting. Active logins uses O(u)

"""
Problem 6 - Aggregate average API latency per endpoint
You are given API request logs. Each log entry is a string in the format:

"endpoint latency_ms timestamp"

Example:
logs = [
    "/login 120 100",
    "/search 300 110",
    "/login 180 120",
    "/checkout 500 130",
    "/search 200 140"
]

Write a function average_latency_by_endpoint(logs) that returns the average latency for each endpoint.

Expected output:
{
    "/login": 150.0,
    "/search": 250.0,
    "/checkout": 500.0
}

Assumptions:
Each log is complete and space-delimited
latency_ms is an integer
timestamp is an integer
Logs may be unsorted, but ordering does not matter for average latency
Return values should be floats
"""

# Solution:
from collections import defaultdict
def average_latency_by_endpoint(logs):
  latency_sum = defaultdict(int)
  latency_count = defaultdict(int)
  for log in logs:
    endpoint, latency, time = log.split()
    latency = int(latency)
    latency_sum[endpoint] += latency
    latency_count[endpoint] += 1
  average_latency = {endpoint: latency_sum[endpoint] / latency_count[endpoint] for endpoint in latency_sum}
  return average_latency

# Complexity:
# Time: O(n) as we scan all logs and then compute average for each endpoint
# Space: O(e) as we store all endpoint latencies sums and counts where e is number of unique endpoints
# Sorting is not needed as latency does not depend on event order

"""
Problem 7 - Find services with high error rate
You are given service request logs. Each log entry is a string in the format:

"service_name status timestamp"

where status is either "success" or "error".
Example:
logs = [
    "svcA success 100",
    "svcA error 110",
    "svcA success 120",
    "svcB error 130",
    "svcB error 140",
    "svcB success 150"
]

Write a function services_above_error_rate(logs, threshold) that returns services whose error rate is greater than or equal to threshold.

services_above_error_rate(logs, 0.5)

Expected output:
["svcB"]

Because:
svcA: 1 error / 3 total = 0.333...
svcB: 2 errors / 3 total = 0.666...


Assumptions:
Error rate = number of error logs / total logs for that service
Logs may be unsorted
Ordering does not matter
threshold is a float between 0 and 1
Return services in any order
"""

# Solution:
from collections import defaultdict
def services_above_error_rate(logs, threshold):
  error_counts = defaultdict(int)
  log_counts = defaultdict(int)
  for log in logs:
    service, status, time = log.split()
    log_counts[service] += 1
    if status == "error":
      error_counts[service] += 1
  error_rates = {service: error_counts[service] / log_counts[service] for service in log_counts}
  return [service for service, error_rate in error_rates.items() if error_rate >= threshold] # OR
  # return [service for service in log_counts if error_counts[service] / log_counts[service]  >= threshold]
  

# Complexity:
# Time: O(n) as we scan all logs and then scan all unique services
# Space: O(s) where s is number of unique services and we store all error rates and total count for each service 
# Sorting is not needed as error_rate is aggregate metric and does not depend on the event order

"""
Problem 8 - Two sum
You are given a list of integers and a target integer.

nums = [2, 7, 11, 15]
target = 9

Write a function two_sum(nums, target) that returns the indices of the two numbers such that they add up to the target.

Expected output:
[0, 1]

Because:
nums[0] + nums[1] = 2 + 7 = 9

Assumptions:
Exactly one valid answer exists
You may not use the same element twice
Return the indices, not the values
The input list is not sorted
"""

# Solution:
def two_sum(nums, target):
  seen = {}
  for i, num in enumerate(nums):
    needed = target - num
    if needed in seen:
     return [seen[needed], i]
    seen[num] = i
  return None

# Complexity:
# Time: O(n) as we scan all items
# Space: O(n) as we might end up storing all items in worst case.
# I check needed in seen before storing the current number so I do not accidentally use the same element twice.
# Example Use of this pattern - Two log events exactly K seconds apart - see below

"""
Given logs with timestamps, find two events from the same user that are exactly 300 seconds apart.

Example:
logs = [
    "u1 login 100",
    "u1 search 250",
    "u1 logout 400",
    "u2 login 500"
]

For u1, 100 and 400 are 300 seconds apart.
"""

# Solution:
from collections import defaultdict
def find_two_logs(logs, k):
  seen_by_user = defaultdict(dict)
  for log in logs:
    uid, action, time = log.split()
    time = int(time)
    needed = time - k
    if needed in seen_by_user[uid]:
      return (uid, seen_by_user[uid][needed], (action, time)) 
    seen_by_user[uid][time] = (action, time)
  return None



###############################################
#                                             #
#                 LIST / DEQUE                #
#                                             #
###############################################


"""
Problem 1 - Apply backspaces to a string:
You are given a string representing a stream of characters typed by a user. A backspace is represented by the character #. Return the final string after all backspaces have been applied.

For Example:
input:  "ab#c#d"
output: "ad"

input:  "a##b"
output: "b"

input:  "abc#"
output: "ab"
"""

# Solution:
def apply_backspace(seq):
  processed = []
  for ch in seq:
    if ch == '#':
      if processed:
        processed.pop()
    else:
      processed.append(ch)
  return "".join(processed)


# Using Deque:
from collections import deque
def apply_backspace(seq):
  processed = deque()
  for ch in seq:
    if ch == '#':
      if processed:
        processed.pop()
    else:
      processed.append(ch)
  return "".join(processed)

# Complexity:
# Time: O(n)
# Space: O(n)

"""
Problem 1a:
Given two strings s and t, return True if they produce the same final string after all backspaces are applied.

s = "ab#c"
t = "ac"
output: True  # both produce "ac"

s = "ab##"
t = "c#"
output: True  # both produce ""

s = "a#b"
t = "c"
output: False  # "b" != "c"
"""

# Solution:
def seq_validate(s, t):
  if apply_backspace(s) == apply_backspace(t):
    return True
  else:
    return False

#OR

def seq_validate(s, t):
  return apply_backspace(s) == apply_backspace(t)

# Complexity:
# Time: O(n)
# Space: O(n)
# Due to underlying complexity of apply_backspace function.

"""
Problem 2 - Implement a simple rate limiter
Implement a function or class that decides whether a request should be allowed. 

A user is allowed at most 5 requests within any 60-second window.

Example:
limiter = RateLimiter()

limiter.allow_request("u1", 1)    # True
limiter.allow_request("u1", 10)   # True
limiter.allow_request("u1", 20)   # True
limiter.allow_request("u1", 30)   # True
limiter.allow_request("u1", 40)   # True
limiter.allow_request("u1", 50)   # False
limiter.allow_request("u1", 70)   # True

Write:
from collections import defaultdict
class RateLimiter:
  def allow_request(self, user_id, timestamp):
    pass
    
Assumptions:
- Requests arrive in non-decreasing timestamp order
- Timestamps are integers
- Each user has an independent rate limit
- A request at timestamp t is allowed only if the user has made fewer than 5 requests in the previous 60 seconds
"""

# Solution:
from collections import defaultdict, deque
class RateLimiter:
  def __init__(self):
    self.requests = defaultdict(deque)

  def allow_request(self, user_id, timestamp):
    # check if this user has less than equal to 5 requests within 60 seconds
    # so remove timestamps older than 60 second and see if remaining are 5 or less
    history = self.requests[user_id]
    while history and timestamp - history[0] >= 60:
      history.popleft()
    if len(history) < 5:
      history.append(timestamp) # should directly update the requests dictionary
      return True
    else:
      return False


# Complexity:
# Time: O(1)
# Doesnt scale with number of requests from user as the request queue size is fixed
# Space: O(u * k)
# Where u is users and k is timestamps per user
# As k is fixed so its O(u)
# Assumption here is that requests arrive in non-decreasing timestamp order. If they do not, this deque-based approach would not be sufficient without additional sorting or a different data structure.



###############################################
#                                             #
#                     SET                     #
#                                             #
###############################################


"""
Problem 1 - Find services with consecutive failures
You are given service health-check logs. Each log entry is a string in the format:

"service_name status timestamp"

where status is either "success" or "fail".

logs = [
    "serviceA success 100",
    "serviceA fail 110",
    "serviceA fail 120",
    "serviceB success 130",
    "serviceA fail 140",
    "serviceB fail 150"
]
Write a function services_with_consecutive_failures(logs, threshold) that returns services that had at least threshold consecutive failures.
Example:
services_with_consecutive_failures(logs, 3) gives:
["serviceA"]

Assumptions:
Logs may be unsorted
Consecutive failures are counted per service
A success resets the failure count for that service
Timestamps are integers
Return each service only once
Output order does not matter
"""

# Solution:
from collections import defaultdict
def services_with_consecutive_failures(logs, threshold):
  # In order to ensure consequtiveness we need to sort the logs
  events = []  
  for log in logs:
    service, status, time = log.split()
    events.append((int(time), status, service))
  events.sort()
  service_failure_counts = defaultdict(int)
  result = set() # So we dont end up appending the same service multiple times
  for time, status, service in events:
    if status == 'fail':
      service_failure_counts[service] += 1
      if service_failure_counts[service] >= threshold:
        result.add(service)
    elif status == 'success':
      service_failure_counts[service] = 0
  return list(result)

# Complexity:
# Time: O(n logn) due to sorting all events globally if sorted already then O(n)
# Space: O(n) as we store all events during sorting. Failure counts uses O(s) where s is number is services.


###############################################
#                                             #
#                   COUNTER                   #
#                                             #
###############################################


"""
Problem 1 - top k most frequent words:
Given a list of words, return the top K most frequent words. If two words have the same frequency, return them in alphabetical order.

words = ["apple", "banana", "apple", "cherry", "banana", "apple"]
k = 2
output: ["apple", "banana"]
"""

# Solution: dict version first:
def top_k(words, k):
  if not words:
    return words
  if len(words) == 1:
    return words[0]

  topk = {}
  for word in words:
    topk[word] = topk.get(word, 0) + 1

  return [word for word, count in sorted(topk.items(), key=lambda x: (-x[1], x[0]))][:k]

# Counter Version:
from collections import Counter
def top_k(words, k):
  words = Counter(words)
  max_words = words.most_common(k) # returns a list of tuples
  return [word for word, count in sorted(max_words, key=lambda x: (-x[1], x[0]))][:k]

# Note:
# Here sorting by two keys freq x[1] and alphabetically x[0]. When sorting in descending order we simply put a -ve sign in front because negative numbers have higher the lower.
# Sorting complexity is O(n log) coz it requires to visit all elements and some of them (if not all of them) to determine the order

# Complexity:
# Time: O(n logn)
# Space: O(n)
# Due to sorting step dominating the counting step

"""
Problem 2: String Manipulation
Given a sentence string, reverse the order of words but keep each word's characters in the correct order. Remove any extra spaces between words.

input:  "  the sky   is blue  "
output: "blue is sky the"
"""

# Solution:
def reverse_seq(seq):
  seq = seq.split() # this should also remove leading/training whitespaces
  return " ".join(seq[::-1]) #::-1 reverses a list in python

# Complexity:
# Time: O(n)
# Space: O(n)
# As we are visiting all elements twice, once to split and once to join and also storing them.



###############################################
#                                             #
#              OOP CLASS DESIGN               #
#                                             #
###############################################


"""
Problem 5 - Implement a TaskQueue to manage tasks:
Design a class TaskQueue that manages a queue of tasks. Each task has a name, a priority (integer, higher = more important), and a status (pending, complete).

The class should support:
add_task(name, priority) — adds a new task
complete_task(name) — marks a task as complete
get_next_task() — returns the highest priority pending task without removing it
get_all_pending() — returns all pending tasks sorted by priority descending
"""

# Solution:

class Task:
  def __init__(self, name: str, priority: int, status: str='pending'):
    self.name = name
    self.priority = priority
    self.status = status

class TaskQueue:
  def __init__(self):
    self.queue = {}

 def add_task(self, name, priority) -> None:
   self.queue[name] = Task(name, priority)

 def complete_task(self, name):
   self.queue[name].status = 'complete' 

 def get_next_task(self):
   if self.queue:    # max() throws an error on empty iterable
     return max(self.queue, key=lambda x: self.queue[x].priority)
   else:
     return None

 def get_all_pending(self):
   pending_tasks = {name: task for name, task in self.queue.items() if task.status == 'pending'}
   return sorted(pending_tasks.values(), key=lambda x: -x.priority)

# Tip:
# Key argument used inside sorted or max functions is originally what iterable gives as a sequence e.g., list as items or dict as list of key by default. You could modify it originally by modifying the iterator e.g., self.queue.values(). Now key can be further used to modify it before comparison e.g., compare original iterator values based on the returned sequences of lambda function e.g., key=lambda x: -x.priority. So compare names of task based on the task priority list returned by lambda function and return the name associated with max number.

## Heaps:
"""
Concept:
It's always a binary tree. Every node has at most 2 children. And it's always complete and filled left to right at every level.
Smallest always at root. When a small element is added it starts bubbling to get to top or higher and is why insert is O(log n) — tree height is log n.
Heaps in python are implemented through lists with some rules.
While heaps could be min-heap or max-heap Python only implements min-heap. To implement max-heap one could negate the numbers.
Heap can be applied essentially on lists. Its functionality is stored in heapq package in python. 
Useful methods:
heapq.heapify(list)
heapq.heappush(list, element)
heapq.heappop(list)
heapq.nlargest(k, list)  # For stream dont pass the whole stream list else loose the benefit, see implementation below
heapq.nsmallest(k, list)
"""

"""
Problem 1 - K-largest numbers
Given a list of numbers, return the K largest elements in descending order.

nums = [3, 1, 4, 1, 5, 9, 2, 6]
k = 3
output: [9, 6, 5]
"""

# Solution:
def k_largest(nums):
  return sorted(nums, reverse=True)[:k]

# Complexity:
# Time: O(n logn)
# Space: O(n)

# OR

def k_largest(nums, k):
  return heapq.nlargest(k, nums)

# Complexity:
# Time: O(n logk)
# Space: O(k)

# What if the list nums was an infinite stream?

def topk(stream, k):
  h_nums = []
  for num in stream:
    heapq.heappush(h_nums, num)
    if len(h_nums) > k:
      heapq.heappop(h_nums) # Remove small keep only top k
  # heap h_nums now contain top k largest elements
  return sorted(h_nums, reverse=True) # for top_k largest
  # return sorted(h_nums) # for top_k smallest

"""
Problem 2 - Find top K users by number of actions
You are given user activity logs. Each log entry is a string in the format:

"user_id action timestamp"

Example:
logs = [
    "u1 login 100",
    "u2 search 110",
    "u1 search 120",
    "u3 login 130",
    "u1 logout 140",
    "u2 logout 150"
]

Write a function top_k_users(logs, k) that returns the top k users by number of actions.

Example:
top_k_users(logs, 2)

Expected output:
["u1", "u2"]

Assumptions:
Logs may be unsorted
Timestamp order does not matter
If there is a tie, return users in lexicographical order
k is always valid
"""

# Solution:
from collections import defaultdict
def top_k_users(logs, k):
  num_actions = defaultdict(int)
  for log in logs:
    uid, action, time = log.split()
    num_actions[uid] += 1
  sorted_users = sorted(num_actions.items(), key=lambda x: (-x[1], x[0]))
  return [user for user, _ in sorted_users[:k]]
  
# Using Counter: It could be solved but counter can not handle double sorting using most_common(k)

# Using heap:
import heapq
from collections import defaultdict
def top_k_users(logs, k):
  num_actions = defaultdict(int)
  for log in logs:
    uid, action, time = log.split()
    num_actions[uid] += 1
  heap_items = [(-count, user) for user, count in num_actions.items()]
  topk = heapq.nsmallest(k, heap_items) 
  return [user for _, user in topk]

# Complexity:
# Time: O(n + ulogu) where n is number of log and u is number of users. Worst case u = n so O(n logn)
# Space: O(u) where u is number of unique users
# For heap solution space complexity remains but time becomes now O(n ulogk) as we only find k items



###############################################
#                                             #
#               MOCK INTERVIEW                #
#                                             #
###############################################



"""
Problem:
You are given a list of transactions. Each transaction is a string in the format:

"user_id amount category”

For example:

transactions = [
    "u1 100 food",
    "u2 200 travel",
    "u1 50 food",
    "u3 300 travel",
    "u2 100 food",
    "u1 200 travel"
]

Part 1: Write a function that returns the user who spent the most in total across all categories.
"""

# Thoughts:
# So I'd simply iterate over the transactions and basically for each user aggregate the amounts across 
# all categories without caring about the order. # This should give me total amounts spent per user. 
# Now I could use a dictionary to store this information, in fact, a defaultdict of int type would be preferred 
# as Id be accessing the items back and forth to add to existing keys so to avoid missing error a defaultdict is safer. 
# As a find step I could simple take the max of this dict over its values and return the key of that max value.

# Solution:
from collections import defaultdict
def most_spent_user(transactions):
  user_expenditure = defaultdict(int)
  for transaction in transactions:
    uid, amount, category = transaction.split()
    amount = int(amount)
    user_expenditure[uid] += amount
  return max(user_expenditure, key=user_expenditure.get)

# Complexity:
# Time: O(n) where n is number of transactions
# Space: O(u) where u is unique users

"""
Part 1: Now return a dictionary where each key is a user and the value is their single highest spending category.

Expected output for the example above:
{
  "u1": "food",    # u1 spent 150 on food, 200 on travel → travel
  "u2": "travel",  # u2 spent 200 on travel, 100 on food → travel
  "u3": "travel"   # u3 spent 300 on travel → travel
}
"""

# Thoughts:
# We could simply again iterate over transactions and add them to a nested dict which has user a top level 
# and category: expenditure as child level. And we need to aggregate first at the category level for each user 
# and then simply return the max of each nest dict for each user. For next dict Id user defaultdict of defaultdict 
# using lambda.

# Solution:
from collections import defaultdict
def highest_user_spent(transactions):
  user_expenditure = defaultdict(lambda: defaultdict(int))
  for transaction in transactions:
    uid, amount, category = transaction.split()
    amount = int(amount)
    user_expenditure[uid][category] += amount
  return {uid: max(categories, key=categories.get)
  for uid, categories in user_expenditure.items()
  }

# Complexity:
# Time: O(n)
# Space: O(u x c)

"""
Part 3: Now write a function that returns the top 2 users by total spending per category. Output should be a dictionary where each key is a category and the value is a list of the top 2 user IDs ordered by spending descending.


Expected output:
{
  "food":   ["u1", "u2"],   # u1=150, u2=100
  "travel": ["u3", "u2"],   # u3=300, u2=200
}
"""

# Thoughts:
# We store aggregated values per category per user.
# So dict of dict like category -> user -> amount
# We could also use counter or simply [:k] of sorted dict.
# Or use heap to take nlargest but for that we need to store like -x.
# I'd simply stick to defaultdict way.

# Solution:
def category_spending(transactions, k=2):
  cat_expenditure = defaultdict(lambda: defaultdict(int))
  for transaction in transactions:
    uid, amount, category = transaction.split()
    amount = int(amount)
    cat_expenditure[category][uid] += amount
  return {category: sorted(users, key=users.get, reverse=True)[:k]
    for category, users in cat_expenditure.items()
  }

# Complexity:
# Time: O(n logu)
# Space: O(c x u)

# To improve the parsing transaction logic can be factored out and reused across all three functions.



###############################################
#                                             #
#           REAL INTERVIEW - JUNE 2026        #
#                                             #
###############################################


"""
Real Coding Interview — June 2026

Problem: GenAI Experiment Manager
Design a class system to manage GenAI experiments. Each experiment has:

- `exp_id` — unique string identifier
- `metrics` — dict of metric names to float values e.g. `{"accuracy": 0.92, "latency": 120.5}`
- `tags` — set of strings e.g. `{"gemini:exp1", "production", "v2"}`

Implement two classes:

**`Experiment`** — data class representing a single experiment.

**`ExperimentManager`** — manages a collection of experiments with the following methods:

-----

Methods

#### `add_experiment(exp_id, metrics, tags)`

Add a new experiment. If exp_id already exists, raise an error.

#### `get_experiment(exp_id)`

Return the experiment object for the given exp_id.
Raise a KeyError with a meaningful message if not found.

#### `filter_experiments(metrics_filter, tags_filter)`

Return a list of experiments matching ALL conditions:

- `metrics_filter` — dict of metric name to list of (operator, value) tuples
  e.g. `{"accuracy": [(">", 0.8)], "latency": [("<", 200)]}`
- `tags_filter` — list of tags that must ALL be present
  e.g. `["gemini:exp1", "production"]`

#### `aggregate_metrics()`

Return a dict of metric name to average value across all experiments.
e.g. `{"accuracy": 0.89, "latency": 145.2}`
"""

# Solution

VALID_OPERATORS = {">", "<", ">=", "<=", "=="}

class Experiment:
    def __init__(self, exp_id: str, metrics: dict, tags: set):
        self.exp_id = exp_id
        self.metrics = metrics
        self.tags = set(tags)  # set for O(1) lookup

class ExperimentManager:
    def __init__(self):
        self.experiments = {}

    def add_experiment(self, exp_id: str, metrics: dict, tags: set) -> None:
        if exp_id in self.experiments:
            raise ValueError(f"Experiment {exp_id} already exists")
        self.experiments[exp_id] = Experiment(exp_id, metrics, tags)

    def get_experiment(self, exp_id: str) -> Experiment:
        try:
            return self.experiments[exp_id]
        except KeyError:
            raise KeyError(f"Experiment {exp_id} not found")

    def filter_experiments(self, metrics_filter: dict, tags_filter: list) -> list:
        results = []
        for exp in self.experiments.values():
            # Check tags — set lookup O(1)
            if not all(tag in exp.tags for tag in tags_filter):
                continue
            # Check metrics
            match = True
            for metric, conditions in metrics_filter.items():
                if metric not in exp.metrics:
                    match = False
                    break
                for operator, threshold in conditions:
                    # Security: validate operator against whitelist
                    if operator not in VALID_OPERATORS:
                        raise ValueError(f"Invalid operator: {operator}")
                    if operator == ">":
                        if not exp.metrics[metric] > threshold:
                            match = False
                    elif operator == "<":
                        if not exp.metrics[metric] < threshold:
                            match = False
                    elif operator == ">=":
                        if not exp.metrics[metric] >= threshold:
                            match = False
                    elif operator == "<=":
                        if not exp.metrics[metric] <= threshold:
                            match = False
                    elif operator == "==":
                        if not exp.metrics[metric] == threshold:
                            match = False
            if match:
                results.append(exp)
        return results

    def aggregate_metrics(self) -> dict:
        totals = {}
        counts = {}
        for exp in self.experiments.values():
            for metric, value in exp.metrics.items():
                totals[metric] = totals.get(metric, 0) + value
                counts[metric] = counts.get(metric, 0) + 1
        return {metric: totals[metric] / counts[metric]
                for metric in totals}

# Test Cases

# Setup
manager = ExperimentManager()
manager.add_experiment(
    "exp1",
    {"accuracy": 0.92, "latency": 120.0},
    {"gemini:exp1", "production"}
)
manager.add_experiment(
    "exp2",
    {"accuracy": 0.75, "latency": 95.0},
    {"gemini:exp2", "staging"}
)
manager.add_experiment(
    "exp3",
    {"accuracy": 0.88, "latency": 200.0},
    {"gemini:exp1", "production"}
)

# Test 1 — get experiment
exp = manager.get_experiment("exp1")
assert exp.exp_id == "exp1"
assert exp.metrics["accuracy"] == 0.92

# Test 2 — filter by metric and tag
results = manager.filter_experiments(
    metrics_filter={"accuracy": [(">", 0.85)]},
    tags_filter=["production"]
)
assert len(results) == 2  # exp1 and exp3
assert all("production" in r.tags for r in results)

# Test 3 — aggregate metrics
agg = manager.aggregate_metrics()
assert abs(agg["accuracy"] - 0.85) < 0.01  # (0.92+0.75+0.88)/3
assert abs(agg["latency"] - 138.33) < 0.01

# Test 4 — edge case: missing experiment
try:
    manager.get_experiment("exp99")
    assert False, "Should have raised KeyError"
except KeyError:
    pass  # expected

# Test 5 — edge case: duplicate experiment
try:
    manager.add_experiment("exp1", {}, set())
    assert False, "Should have raised ValueError"
except ValueError:
    pass  # expected

# Test 6 — edge case: invalid operator
try:
    manager.filter_experiments(
        metrics_filter={"accuracy": [("DROP TABLE", 0.8)]},
        tags_filter=[]
    )
    assert False, "Should have raised ValueError"
except ValueError:
    pass  # expected

### Complexity

"""
|Method              |Time        |Space|
|--------------------|------------|-----|
|`add_experiment`    |O(1)        |O(1) |
|`get_experiment`    |O(1)        |O(1) |
|`filter_experiments`|O(n x m x c)|O(n) |
|`aggregate_metrics` |O(n x m)    |O(m) |
"""

# Where:
# - n = number of experiments
# - m = number of metrics per experiment
# - c = number of conditions per metric filter

# Key Learnings
# - Use `set` for tags — O(1) lookup vs O(n) for list
# - Validate operators against a whitelist before use — security best practice
# - `try/except KeyError` for missing dict keys — cleaner than `.get()` when you want to raise
# - Single iteration in `filter_experiments` — check both tags and metrics in one pass
# - Always include `return` statements — easy to forget under pressure
# - `defaultdict` not appropriate here — want explicit error on missing experiment


# Interview Notes
# - Asked by a Tech company for FDE role, June 2025
# - Interviewer suggested `set` for tags optimization (to retrieve) instead of list
# - Security question on operator injection eval() usage is unsafe — answer: whitelist validation with if conditions
# - Follow-up: write test cases including edge cases
# - Role context: production-ready GenAI engineering, not data science heavy