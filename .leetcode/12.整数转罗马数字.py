#
# @lc app=leetcode.cn id=12 lang=python3
#
# [12] 整数转罗马数字
#

# @lc code=start
class Solution:
    def intToRoman(self, num: int) -> str:
        hashmap={1000:"M",
                900:"CM",
                500:"D",
                400:"CD",
                100:"C",
                90:"XC",
                50:"L",
                40:"XL",
                10:"X",
                9:"IX",
                5:"V",
                4:"IV",
                1:"I"}
        res = ''
        for key in hashmap:
            if num // key != 0:
                count = num // key  # 比如输入4000，count 为 4
                res += hashmap[key] * count 
                num %= key
        return res

# @lc code=end

