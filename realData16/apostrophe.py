origList = ['\'tis','\'twas','ain\'t','aren\'t','can\'t','could\'ve','couldn\'t','didn\'t','doesn\'t','don\'t','hasn\'t','he\'d','he\'ll','he\'s','how\'d','how\'ll','how\'s','i\'d','i\'ll','i\'m','i\'ve','isn\'t','it\'s','might\'ve','mightn\'t','must\'ve','mustn\'t','shan\'t','she\'d','she\'ll','she\'s','should\'ve','shouldn\'t','that\'ll','that\'s','there\'s','they\'d','they\'ll','they\'re','they\'ve','wasn\'t','we\'d','we\'ll','we\'re','weren\'t','what\'d','what\'s','when','when\'d','when\'ll','when\'s','where\'d','where\'ll','where\'s','who\'d','who\'ll','who\'s','why\'d','why\'ll','why\'s','won\'t','would\'ve','wouldn\'t','you\'d','you\'ll','you\'re','you\'ve']

trimList = ['tis','twas','aint','arent','cant','couldve','couldnt','didnt','doesnt','dont','hasnt','hed','hell','hes','howd','howll','hows','id','ill','im','ive','isnt','its','mightve','mightnt','mustve','mustnt','shant','shed','shell','shes','shouldve','shouldnt','thatll','thats','theres','theyd','theyll','theyre','theyve','wasnt','wed','well','were','werent','whatd','whats','when','whend','whenll','whens','whered','wherell','wheres','whod','wholl','whos','whyd','whyll','whys','wont','wouldve','wouldnt','youd','youll','youre','youve']

def getPos(strList, word):
    try:
        ind = strList.index(word)
        return ind
    except ValueError:
        return None

while True:

    try:
        text = raw_input()
    except EOFError:
        break
        
    newText = ''

    for w in text.split():
        valid = 0
        newW = ''
        
        for i in range(len(trimList)):
            var = trimList[i]
            tmp = str(w)
            tmp2 = str(w)

            #print 'OLD TMP IS', tmp
            tmp = tmp.lower().strip(',').strip('\"').strip('.').strip(',').strip('\"').strip('.')
            #print 'NEW TMP IS', tmp
            #print 'NEW TMP 2 IS', tmp2
            
            if (len(tmp) > len(var)):
                continue
            
            #print var, w, w.lower(), w
            
            if var in tmp2.lower():
                #print 'VAR IS', var
                valid = 1
                if ((len(var) > 2) and ('Ive' not in w)):
                    #print 'FIRST', (var[-3:], origList[i][-4:])
                    newW = str(w).replace(var[-3:], origList[i][-4:])
                else:
                    newW = str(w).replace(var[-2:], origList[i][-3:])
                #print 'THE NEW ONE IS', newW
                break
        
        if valid:
            newText += newW + ' '
        elif ('ys' in w) and ('ys' in w[-2:]):
            #print '\n\n\nYES YES\n\n\n'
            #print w
            w = w.replace('ys', 'y\'s')
            #print w
            newText += w + ' '
        else:
            newText += w + ' '

    print list(newText)

print 'At a news conference Thursday at the Russian manned-space facility in Baikonur, Kazakhstan, Kornienko said \"we will be missing nature, we will be missing landscapes, woods.\" He admitted that on his previous trip into space in 2010 \"I even asked our psychological support folks to send me a calendar with photographs of nature, of rivers, of woods, of lakes.\"\nKelly was asked if he\'d miss his twin brother Mark, who also was an astronaut.\n\"We\'re used to this kind of thing,\" he said. \"I\'ve gone longer without seeing him and it was great.\"\nThe mission won\'t be the longest time that a human has spent in space - four Russians spent a year or more aboard the Soviet-built Mir space station in the 1990s.\nSCI Astronaut Twins\nScott Kelly (left) was asked Thursday if he\'d miss his twin brother, Mark, who also was an astronaut. \'We\'re used to this kind of thing,\' he said. \'I\'ve gone longer without seeing him and it was great.\' (NASA/Associated Press)\n\"The last time we had such a long duration flight was almost 20 years and of course all ... scientific techniques are more advanced than 20 years ago and right now we need to test the capability of a human being to perform such long-duration flights. So this is the main objective of our flight, to test ourselves,\" said Kornienko.\"'
