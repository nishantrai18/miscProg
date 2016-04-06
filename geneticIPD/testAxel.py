import axelrod
from axelrod import Actions, Player, init_args

ply = axelrod.EvolveAxel()
print ply
p1, p2 = axelrod.Cooperator(), axelrod.Forgiver()
# for turn in range(10):
#     p1.play(p2)
# print p1.history, p2.history
for turn in range(10):
    ply.play(p2)
print ply.history
print p2.history

p2.history = []
print p2.history