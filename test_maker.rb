redTeam  = "MonteCarloTeam"
blueTeam = "RandomTeam"
numGames = 20

totalWinsRed = 0
totalWinsBlue = 0
totalTies = 0

mediumScore = 0

numGames.times do |x|
  x = %x[python capture.py -r #{redTeam} -b #{blueTeam} -l RANDOM]

  last_line = x.split("\n").last
  match = last_line.match(/The (.*) team wins by (.*) points./)

  if match.nil?
    totalTies += 1
  else
    winnerTeam = match[1]
    points = match[2].to_i
    if winnerTeam == "Blue"
      totalWinsBlue += 1
    else
      totalWinsRed += 1
    end
    mediumScore += points
  end

end

puts "Red won #{totalWinsRed} games."
puts "Blue won #{totalWinsBlue} games."
puts "#{totalTies} ties."
puts "Games medium score was #{mediumScore.to_f/numGames.to_f}"
