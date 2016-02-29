import axelrod
print axelrod.ordinary_strategies
strategies = [s() for s in axelrod.ordinary_strategies]
tournament = axelrod.Tournament(strategies)
results = tournament.play()
plot = axelrod.Plot(results)
p = plot.boxplot()
p.show()
input()