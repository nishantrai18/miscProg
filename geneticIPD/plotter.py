import pickle
import matplotlib.pyplot as plt
import pylab

target = open('multObjPop.p', 'rb')
a, b = pickle.load(target)
a = zip(*a)
b = zip(*b)
# c = zip(*c)

plt.figure()  
ax = plt.subplot(1,1,1)
# ax.spines["top"].set_visible(False)  
# ax.spines["right"].set_visible(False)

# ax.get_xaxis().tick_bottom()
# ax.get_yaxis().tick_left()
# plt.ylim(200, 340)
# plt.xlim(100, 350)
# plt.xticks(range(120, 340, 30), fontsize=14)
# plt.yticks(range(120, 340, 30), fontsize=14)

# gen = range(len(a))
# plt.ylabel("Fitness", fontsize=16)
# plt.xlabel("Generation No.", fontsize=16)
# plt.title("Fitness during Evolution", fontsize=18)
# plt.plot(gen, a, color = 'red')
# # plt.plot(gen,b, color = 'blue')
# plt.plot(gen, c, color = 'green')

plt.ylabel("Opponent Score", fontsize=16)
plt.xlabel("Self Score", fontsize=16)
plt.title("Evolved Strategies", fontsize=18)
s = [40]*len(a)
plt.scatter(a[0],a[1], marker = '+', color = 'red', s=s)
plt.scatter(b[0],b[1], marker = 'x', color = 'blue', s=s)
# plt.show()	
plt.savefig("evolvePlot.eps", bbox_inches="tight", format = 'eps')