### 遍历法

#### 双指针
##### 141.环形链表    (快慢指针)
给定一个链表，判断链表中是否有环。如果链表中有某个节点，可以通过连续跟踪 next 指针再次到达，则链表中存在环。 为了表示给定链表中的环，我们使用整数 `pos` 来表示链表尾连接到链表中的位置（索引从 0 开始）。 如果 `pos` 是 -1，则在该链表中没有环。注意：`pos` 不作为参数进行传递，仅仅是为了标识链表的实际情况。如果链表中存在环，则返回 `true` 。 否则，返回 `false` 。

``````python
class Solution:
    def hasCycle(self, head: ListNode) -> bool:
        if not head or not head.next:
            return False
        fast = slow = head                  # 创建 fast 和 slow 两个指针
        while fast and fast.next:           # 每次循环，fast指针走两步，slow走一步
            fast = fast.next.next
            slow = slow.next
            if fast == slow:                # 如果某次循环中两指针相遇，则说明存在闭环
                return True
        return False
``````

##### 881.救生艇 （对撞指针）

第 `i` 个人的体重为 `people[ i ]`，每艘船可以承载的最大重量为 `limit`。每艘船最多可同时载两人，但条件是这些人的重量之和最多为 `limit`。返回载到每一个人所需的最小船数。(保证每个人都能被船载)。 

```python
class Solution:
    def numRescueBoats(self, people: List[int], limit: int) -> int:
        a=sorted(people)
        #对list中的元素进行排序，也可以用people.sort(),时间复杂度为O(NlogN) 
        nums=0
        i , j=0, len(people)-1   #建立前后两个指针
        while i<j:
            if a[i]+a[j]<= limit:   #两指针同时向中间靠拢
                nums+=1
                i+=1 
                j-=1
            else :
                j-=1
        return nums+(len(people)-2*nums)
   
```

#### 二分查找法

二分查找法需要记好三个遍历关键点，熟练套用公式解题

##### 704.二分查找

给定一个 n 个元素有序的（升序）整型数组 `nums` 和一个目标值 `target`  ，写一个函数搜索 `nums` 中的 `target`，如果目标值存在返回下标，否则返回 -1。

```python
class Solution:           #【写法2】
    def search(self, nums: List[int], target: int) -> int:
        l=0
        r= len(nums)-1    #若左闭右开，改为 r = len(nums) 【写法1】
        while l <= r:         #改为l < r 
            m=(l+r)//2
            if nums[m]>target:
                r=m-1         #改为 r = m 
            elif nums[m] < target:
                l=m+1
            else:
                return m
        return -1
    # 分析：由于//的运算机制问题，左边边界始终为闭，不管右边是开是闭，始终不影响左边界的值，输出时以左边界为准即可
    #  写法1为左闭右开的结构，结束循环时l=r，且nums[l] 的输出总是一个大于目标值的数，nums[l-1]总是一个小于目标值的数。（此方法优先考虑）
    #  写法2为双边闭合结构，结束循环时为 l=r的下一次循环，nums[l]为大于目标值的数，nums[r]为小于目标值的数。

```

##### 162.寻找峰值

峰值元素是指其值大于左右相邻值的元素。给你一个输入数组 nums，找到峰值元素并返回其索引。数组可能包含多个峰值，在这种情况下，返回 任何一个峰值 所在位置即可。你可以假设 nums[-1] = nums[n] = -∞ 。且相邻两个数的值不相等！

```python
class Solution:
    def findPeakElement(self, nums: List[int]) -> int:
        nums.append(-float("inf"))   #注意这里！
        lo, hi = 0, len(nums) - 1       
        #由于nums[n]= - inf ,所以肯定不纳入范围，实际上还是两边都是闭
        while lo < hi:
            mid = (lo + hi) // 2
            if nums[mid] < nums[mid + 1]:
                lo = mid + 1
            else:
                hi = mid
        return lo 
    	#二分查找时间复杂度都是 O(nlogn）
```

##### 74.搜索二维矩阵（二维数组二分法）

编写一个高效的算法来判断 m x n 矩阵中，是否存在一个目标值。该矩阵具有如下特性：

- 每行中的整数从左到右按升序排列。
- 每行的第一个整数大于前一行的最后一个整数。

```python
class Solution:
    def searchMatrix(self, matrix: List[List[int]], target: int) -> bool:
        m = len(matrix)
        n = len(matrix[0])
        l,r=0, m*n
        while l<r :
            mi = (l+r)//2
            if matrix[(mi//n)][mi % n] == target:
                return True   
            elif matrix[(mi//n)][mi % n] > target:
                r = mi
            else:
                l = mi+1
        return False
```

#### 滑动窗口法

处理连续几个数的问题

##### 209.长度最小的子数组（基于双指针实现的滑动窗口）

给定一个含有 n 个正整数的数组和一个正整数 `target` 。

找出该数组中满足其和 ≥ target 的长度最小的 连续子数组` [numsl, numsl+1, ..., numsr-1, numsr]` ，并返回其长度。如果不存在符合条件的子数组，返回 `0` 。

```python
class Solution:
    def minSubArrayLen(self, target: int, nums: List[int]) -> int:
        i,j=0,0
        total=0
        result=len(nums)+1
        while j< len(nums):
            total+= nums[j]
            j+=1
            while total >= target :
                result = min(result,j-i)
                total =total - nums[i]
                i+=1
        return 0 if result == len(nums)+1 else result

# 此题也可以用前缀和+二分查找来做
```

##### 1456.定长字串中的元音最大数目（基于队列实现的滑动窗口）

给你字符串 `s` 和整数 `k` 。

请返回字符串 `s` 中长度为 `k` 的单个子字符串中可能包含的最大元音字母数。

英文中的 元音字母 为`（a, e, i, o, u）`。

```python
class Solution:
    def maxVowels(self, s: str, k: int) -> int:
        yuan=['a','e','i','o','u']
        q= deque()
        re=0
        temp=0
        for i in range(k):
            if s[i] in yuan:
                re+=1
        temp=re
        for j in range(k,len(s)):
            m=j-k
            if s[m] in yuan:
                temp-=1
            if s[j] in yuan:
                temp+=1
            re=max(re,temp)
        return re
```

### 递归法

#### 普通递归

##### 206.反转链表

给你单链表的头节点 `head` ，请你反转链表，并返回反转后的链表。

```python
# Definition for singly-linked list.
# class ListNode:
#     def __init__(self, val=0, next=None):
#         self.val = val
#         self.next = next
class Solution:
    def reverseList(self, head: ListNode) -> ListNode:
        if not head or not head.next:
            return head 				# 找到下行的最后一个元素 
        p=self.reverseList(head.next) 	#下行不断深入
        head.next.next=head 			#碰触到截止条件后开始上行
        head.next = None 				#注意 python中没有null，是None，N大写
        return p

#利用python快速赋值方法
class Solution:
    def reverseList(self, head: ListNode) -> ListNode:
            pre = None
            cur = head
            while cur:
                pre,pre.next,cur=cur,pre,cur.next
            return pre
 
```

##### 169.多数元素

给定一个大小为 n 的数组，找到其中的多数元素。多数元素是指在数组中出现次数 大于 ⌊ n/2 ⌋ 的元素。你可以假设数组是非空的，并且给定的数组总是存在多数元素。

```python
class Solution:
    def majorityElement(self, nums: List[int]) -> int:
        count = 0
        key = 0
        for i in range(len(nums)):
            if count == 0:
                key=nums[i]
                count+=1
            elif key == nums[i]:
                count +=1
            else:
                count-=1
        return key
   
# 经典的议员投票问题，让全部其他元素都来对抗多数元素，若多数元素占三分之一，则任选两个元素，其他元素一个当两个用...以此类推
```

 #### 分治法

##### 53. 最大子序和

给定一个整数数组 `nums` ，找到一个具有最大和的连续子数组（子数组最少包含一个元素），返回其最大和。

```python
#分治法
#当把整个数列分为两半的时候，最大子数列的和就等于max(左数列和，右数列和，中间和)

class Solution:
    def maxSubArray(self, nums: List[int]) -> int:
        l= 0
        r= len(nums)-1
        S=self.Sum(l,r,nums)
        return S

    def Sum(self,l,r,nums): #求左右两边的最大和的方法
        if l==r:
            return nums[l]
        m=(l+r)//2
        LeftSum=self.Sum(l,m,nums)
        RightSum=self.Sum(m+1,r,nums)
        MiddleSum= self.MiddleSum(l,r,nums)
        return max(LeftSum,RightSum,MiddleSum)

    def MiddleSum(self,l,r,nums): #求中间和的方法
        m=(l+r)//2
        LSum=0
        LeftSum=nums[m]
        for i  in range(m,l-1,-1):
            LSum+=nums[i]
            LeftSum = max(LSum,LeftSum)
        RSum=0
        RightSum=nums[m+1]
        for i  in range(m+1,r+1):
            RSum+=nums[i]
            RightSum = max(RSum,RightSum)
        MiddleSum = RightSum+LeftSum
        return MiddleSum

# 动态规划法
# F(n)=max[F(n-1),0]+num[n]  F(n)为数组长度为n时，包含nums[n]的子序列的和的最大值

class Solution(object):
    def maxSubArray(self, nums):
        """
        :type nums: List[int]
        :rtype: int
         """
        for i in range(1, len(nums)):
            nums[i]= nums[i] + max(nums[i-1], 0)
        return max(nums)
```

#### 回溯法

##### 22.括号生成

数字 `n` 代表生成括号的对数，请你设计一个函数，用于能够生成所有可能的并且 有效的 括号组合。有效括号组合需满足：左括号必须以正确的顺序闭合。

 ```python
 class Solution:
     def generateParenthesis(self, n: int) -> List[str]:
         a=[]
         res=[]
         self.bt(n,0,0,'',res)
         return res
 
     def bt(self,n,l,r,a,res):
         if l==r==n:
             res.append(a) #遍历完成终止条件
             return 
         if r>l:           #中途中断,回溯
             return
         if l < n:
             self.bt(n,l+1,r,a+'(',res) #添加左分支
         if l > r: 
             self.bt(n,l,r+1,a+")",res) #添加右分支
 
 ```

##### 78.子集

给你一个整数数组 `nums` ，数组中的元素 互不相同 。返回该数组所有可能的子集（幂集）。

解集 不能 包含重复的子集。你可以按 任意顺序 返回解集。

```python
# 回溯法：根据子集的大小遍历

class Solution:
    def subsets(self, nums: List[int]) -> List[List[int]]:
            res=[[]]
            for i in range(1,len(nums)+1):
                self.Count(res,nums,i,0,[]) # 长度可以从1取到整个数组长度
            return res

    def Count(self,res,nums,length,index,sub):
        if len(sub)==length:
            res.append(sub[:])#注意！sub[:]是不更新的，sub会更新
            return
        
        for i in range(index,len(nums)): 
            #默认不回头，一旦前一个元素取了nums[i],后面的元素只能在i之后取，防止重复
            sub.append(nums[i])
            self.Count(res,nums,length,i+1,sub)
            sub.pop()

# 动态规划: n+1的子集是【n的子集】+【n的子集+第n+1个元素】

class Solution:
    def subsets(self, nums: List[int]) -> List[List[int]]:
        res = [[]]
        for i in range(len(nums)-1, -1, -1):
            for subres in res[:]: res.append(subres+[nums[i]])
        return res
    

# DFS 深度优先算法：[1]->[1,2]->[1,2,3] ···

class Solution:
    def subsets(self, nums: List[int]) -> List[List[int]]:
        ans = []
        def backtrack(temp,idx):
            if idx == len(nums):
                ans.append(temp)
                return 
            backtrack(temp + [nums[idx]],idx+1)
            backtrack(temp,idx+1)
            
        backtrack([],0)
        return ans


```

#### DFS 深度优先算法

##### 938. 二叉搜索树的范围和

给定二叉搜索树的根结点 `root`，返回值位于范围 *`[low, high]`* 之间的所有结点的值的和。

```python
# 深度优先搜索

class Solution:
    sum=0
    def rangeSumBST(self, root: TreeNode, low: int, high: int) -> int:
        def Count(r):
            if r == None:
                return
            
            elif r.val <= high and r.val >= low:
                 self.sum += r.val
                 Count(r.left)
                 Count(r.right)
            else:
                Count(r.left)
                Count(r.right)
        Count(root)

        return self.sum
    
 # 广度优先搜索

class Solution:
    def rangeSumBST(self, root: TreeNode, low: int, high: int) -> int:
        total = 0
        q = collections.deque([root])
        while q:
            node = q.popleft()
            if not node:
                continue
            if node.val > high:
                q.append(node.left)
            elif node.val < low:
                q.append(node.right)
            else:
                total += node.val
                q.append(node.left)
                q.append(node.right)

        return total

```



#### BFS广度优先搜索

##### 102. 二叉树的层序遍历（该题可用作模板）

给你一个二叉树，请你返回其按 **层序遍历** 得到的节点值。 （即逐层地，从左到右访问所有节点）。

```python
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right


# BFS 法
class Solution:
    def levelOrder(self, root: TreeNode) -> List[List[int]]:
        res =[]
        if root is None: 
            return res
        q = deque([])    #用队列，好处是遵循先入先出原则，保证从左到右的顺序
        q.append(root)
        while len(q) > 0:
            size=len(q)
            ls = []
            while size > 0:
                cur = q.popleft()    # 把上一层元素一一出队，同时按顺序将下一层元素入队
                ls.append(cur.val)
                if cur.left is not None:
                    q.append(cur.left)
                if cur.right is not None:
                    q.append(cur.right)
                size = size-1
            res.append(ls[:])
        return res
    
# DFS 法

    class Solution:
    def levelOrder(self, root: TreeNode) -> List[List[int]]:
        res=[]
        
        self.count(0,root,res)
        return res

    def count(self,layer,r,res):      
            if r is None: # 一定注意这一行！！！ 排除[]的情况
                return

            if layer > len(res)-1:
                    res.append([])
            res[layer].append(r.val)

            if r.left is not None:
                self.count(layer+1,r.left,res)
                
            if r.right is not None:
                self.count(layer+1,r.right,res)
           
```

##### 200.岛屿数量

给你一个由 '1'（陆地）和 '0'（水）组成的的二维网格，请你计算网格中岛屿的数量。岛屿总是被水包围，并且每座岛屿只能由水平方向和/或竖直方向上相邻的陆地连接形成。此外，你可以假设该网格的四条边均被水包围。

```python
# 另类DFS：定义传染函数，每碰到一个1就把它周围的1都变成0

class Solution:
    def numIslands(self, grid: List[List[str]]) -> int:
        m,n=len(grid),len(grid[0])
        ans=0

        def dfs(i,j):
            if 0<=i<m and 0<=j<n and grid[i][j]=='1':
                grid[i][j] = '0'

                dfs(i+1,j)
                dfs(i-1,j)
                dfs(i,j+1)
                dfs(i,j-1)
        for i in range(m):
            for j in range(n):
                if grid[i][j]=='1':
                        ans += 1
                        dfs(i,j)
        return ans

# 比较麻烦的并查集
    
class UnionFind:
    def __init__(self, grid):
        m, n = len(grid), len(grid[0])
        self.count = 0
        self.parent = [-1] * (m * n)
        self.rank = [0] * (m * n)
        for i in range(m):
            for j in range(n):
                if grid[i][j] == "1":
                    self.parent[i * n + j] = i * n + j
                    self.count += 1
    
    def find(self, i):
        if self.parent[i] != i:
            self.parent[i] = self.find(self.parent[i])
        return self.parent[i]
    
    def union(self, x, y):
        rootx = self.find(x)
        rooty = self.find(y)
        if rootx != rooty:
            if self.rank[rootx] < self.rank[rooty]:
                rootx, rooty = rooty, rootx
            self.parent[rooty] = rootx
            if self.rank[rootx] == self.rank[rooty]:
                self.rank[rootx] += 1
            self.count -= 1
    
    def getCount(self):
        return self.count

class Solution:
    def numIslands(self, grid: List[List[str]]) -> int:
        nr = len(grid)
        if nr == 0:
            return 0
        nc = len(grid[0])

        uf = UnionFind(grid)
        num_islands = 0
        for r in range(nr):
            for c in range(nc):
                if grid[r][c] == "1":
                    grid[r][c] = "0"
                    for x, y in [(r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)]:
                        if 0 <= x < nr and 0 <= y < nc and grid[x][y] == "1":
                            uf.union(r * nc + c, x * nc + y)
        
        return uf.getCount()


```

##### 322. 零钱兑换

给你一个整数数组 `coins` ，表示不同面额的硬币；以及一个整数 `amount` ，表示总金额。计算并返回可以凑成总金额所需的 最少的硬币个数 。如果没有任何一种硬币组合能组成总金额，返回 -1 。你可以认为每种硬币的数量是无限的。

```python
# bfs 方法
class Solution:
    total=10001
    def coinChange(self, coins: List[int], amount: int) -> int:
      coins.sort()
      def bfs(amount):  
        distance=0
        q = [0]
        visited_list=[1]+[0]*amount
        while q:
            tep = []
            while q: #出栈
                t=q.pop()
                if t==amount:
                    return distance
                for i in coins:
                    next=t+i
                    if next <= amount and visited_list[next]==0:
                        tep.append(next) #入栈
                        visited_list[next]=1
            distance+=1
            q,tep = tep,q
        return -1
      return bfs(amount)
    # bfs的思路为：一枚硬币可以解决吗？两枚呢？三枚呢？
    
# 动态规划
class Solution:

    def coinChange(self, coins: List[int], amount: int) -> int:
        dp=[0]+[10001]*amount 
        #可以中这种方法创建数组,取10001是保证数组中的数总比输入值大
        #dp[i]表示总金额为i的情况下的最小取值
        for coin in coins:
            for i in range(coin,amount+1):
                dp[i] = min(dp[i],dp[i-coin]+1)
        return dp[-1] if dp[-1]!=10001 else -1 

```

#### 贪心算法

##### 1217.玩筹码

数轴上放置了一些筹码，每个筹码的位置存在数组 `chips` 当中。你可以对 任何筹码 执行下面两种操作之一（不限操作次数，0 次也可以）：

- 将第 i 个筹码向左或者右移动 2 个单位，代价为 0。
- 将第 i 个筹码向左或者右移动 1 个单位，代价为 1。

最开始的时候，同一位置上也可能放着两个或者更多的筹码。返回将所有筹码移动到同一位置（任意位置）上所需要的最小代价。

```python
# 贪心算法

class Solution:
    def minCostToMoveChips(self, position: List[int]) -> int:
        odd = 0
        even = 0
        for i in position:
            if i%2==0:
                even+=1
            else:
                odd+=1
        return min(even,odd)

```

##### 55.跳跃游戏

给定一个非负整数数组 `nums` ，你最初位于数组的 第一个下标 。数组中的每个元素代表你在该位置可以跳跃的最大长度。判断你是否能够到达最后一个下标。

```python
# 贪心算法

class Solution:
    def canJump(self, nums: List[int]) -> bool:
        n = 1
        if nums == [0]:
            return True
        for i in range(len(nums)-2,-1,-1): #从后往前，总是检查前面的数是否能跳过后面的数
            if nums[i]>=n:
                n=1
            else:
                n+=1
            if i == 0 and n>1:
                return False
        return True
```

#### 动态规划

##### 509.斐波那契数列

斐波那契数，通常用 F(n) 表示，形成的序列称为 斐波那契数列 。该数列由 0 和 1 开始，后面的每一项数字都是前面两项数字的和。也就是：

F(0) = 0，F(1) = 1 ，F(n) = F(n - 1) + F(n - 2)，其中 n > 1      给你 n ，请计算 F(n) 。

```python
class Solution:
    
    def fib(self, n: int) -> int:
        a=0
        b=1
        if n==0:
            return 0
        if n==1:
            return 1
        for i in range(2,n+1):
            a,b = b,a+b
            if i == n:
                return b
```

##### 62.不同路径

一个机器人位于一个 m x n 网格的左上角 （起始点在下图中标记为 “Start” ）。机器人每次只能向下或者向右移动一步。机器人试图达到网格的右下角（在下图中标记为 “Finish” ）。问总共有多少条不同的路径？

```python
#公式法

class Solution:
    def uniquePaths(self, m: int, n: int) -> int:
            return math.comb(m+n-2,m-1)
#动态规划

class Solution:
    def uniquePaths(self, m: int, n: int) -> int:
        a=[0]*n
        dp=[a]*m
        for i in range(m):
            for j in range(n):
                if i == 0 or j==0:
                    dp[i][j] = 1
                else:
                    dp[i][j]=dp[i-1][j]+dp[i][j-1]
        return dp[m-1][n-1]

```

