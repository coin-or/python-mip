import matplotlib.pyplot as plt
import numpy as np

t = np.arange(0.000001, 150, 0.01)
s = np.log(t)
for i in range(len(s)):
    s[i] = max(0, s[i]*1520)

print('s:')
print(s)

plt.plot(t, s, color='orange')

d = [0, 10, 30, 70, 150]
f = [round(1520*np.log(x), 3) if x > 0 else 0 for x in d]


print(d)
print(f)

for x, y in zip(d, f):
    plt.scatter(x, y, marker='^', color = 'darkgreen')
    print('x {} y {}'.format(x, y))
    plt.plot( [x, x], [0, y], ':', color='gray' )


plt.plot( d, f, color='darkgray' )
#plt.plot( [0, 10], [0, np.log(10)], color='darkgray' )


#plt.ylim(-0.01,5)
plt.show()
