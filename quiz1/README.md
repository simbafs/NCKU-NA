![](./quiz1.jpg)

# 執行結果
```
pi = 0.785319645811445 , n = 3184 , error = 0.7500249929238642
pi = 0.5235514642438139 , n = 6 , error = 0.8333483930051946
```

# 問題
## Which converges faster?
第二個， ${\pi \over 6} = \tan^{-1}{1 \over \sqrt{3}} $

## Why?
因為 ${1 \over \sqrt{3}} < 1$，所以對於所有正整數 $n$，${{1 \over \sqrt{3}}^n \over n}$ 收斂速度比 $1^n \over n$ 快
