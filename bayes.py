import numpy as np
import re
import mmap
import io
import codecs
import math
from collections import Counter
from collections import defaultdict
import sys
from scipy.spatial import distance
import matplotlib
import matplotlib.pyplot as plt

np.set_printoptions(precision=2,linewidth = 120)

def formatMovieTitle(title):
	reext1 = re.compile('\([^\(]*\d*[a-z][a-z0-9]*\)')
	reext2 = re.compile('\[[^\(]*\d*[a-z][a-z0-9]*\]')
	reext3 = re.compile('\<[^\(]*\d*[0-9]*\>')
	title = reext1.sub('',title)
	title = reext2.sub('',title)
	title = reext3.sub('',title)
	title = title.strip()
	return title

print "ratings...ratings..."
discard = ["(v)","(vg)","(tv)","{"]
imdbMovieRatings = dict()
with open('../ratings.list') as f:
	for line in f:
		line =  line.lower()
		if any(x in line for x in discard): continue
		if len(line)<=37: continue
		rating = line[25:35]
		moviename = formatMovieTitle(line[35:].strip())
		if re.match(r'[0-9]\.[0-9]',rating):
			#print moviename + '\t\t'+rating
			imdbMovieRatings[moviename] = float(rating)/10

matchGenres = {'action':1,'adventure':2,'animation':3,'family':4,'comedy':5,'crime':6,'documentary':7,'drama':8,'fantasy':9,'film-noir':10,'horror':11,'musical':12,'mystery':13,'romance':14,'sci-fi':15,'thriller':16,'war':17,'western':18, 'children':4}
print "ratings!"

print "genres...genres..."
imdbMovieGenres= defaultdict(lambda:np.zeros(19))
with open('../genres.list') as f:
	for line in f:
		line =  line.lower().strip()
		if any(x in line for x in discard): continue
		if '\t' in line:
			strsplt=line.split("\t")
			movie = formatMovieTitle(strsplt[0].strip())
			genre = strsplt[-1].strip()
#			print movie + " " + genre
			if genre in matchGenres:
				vector = np.zeros(19)
				vector[matchGenres[genre]]=1
				imdbMovieGenres[movie]+=vector

print "genres!"

(directors,actors,actresses,writers,composers,cinematographers,producers) =  range(7)
fnames = ['../directors.list','../actors.list','../actresses.list','../writers.list','../composers.list','../cinematographers.list','../producers.list']

print "datasets ready to load..."
filelist = []
filelist.append(open(fnames[directors],'rb'))
filelist.append(open(fnames[actors],'rb'))
filelist.append(open(fnames[actresses],'rb'))
filelist.append(open(fnames[writers],'rb'))
filelist.append(open(fnames[composers],'rb'))
filelist.append(open(fnames[cinematographers],'rb'))
filelist.append(open(fnames[producers],'rb'))

mmaplist = []
mmaplist.append(mmap.mmap(filelist[directors].fileno(),0,prot=mmap.PROT_READ))
mmaplist.append(mmap.mmap(filelist[actors].fileno(),0,prot=mmap.PROT_READ))
mmaplist.append(mmap.mmap(filelist[actresses].fileno(),0,prot=mmap.PROT_READ))
mmaplist.append(mmap.mmap(filelist[writers].fileno(),0,prot=mmap.PROT_READ))
mmaplist.append(mmap.mmap(filelist[composers].fileno(),0,prot=mmap.PROT_READ))
mmaplist.append(mmap.mmap(filelist[cinematographers].fileno(),0,prot=mmap.PROT_READ))
mmaplist.append(mmap.mmap(filelist[producers].fileno(),0,prot=mmap.PROT_READ))
print "datasets ready to load..."

movieToCrew = defaultdict(list)
peopleToExperience = defaultdict(list)
def loadExperience(position):
	found = False
	possibleCrewList = []
	possibleCrewMember = None

	mmaplist[position].seek(0)
	line = mmaplist[position].readline()
	crewMember = ""
	i=0
	while line:
		i+=1
		if (i)%20000 == 0:
			sys.stdout.write("\rfile "+ str(position) +" lines procesed "+str(i))    
			sys.stdout.flush()
		if "\t" not in line:
			line = mmaplist[position].readline()
			continue
		line = line.lower()
		if not line.startswith("\t\t"):
			spltline = line.strip().split("\t")
			crewMember = spltline[0].strip()
			if len(spltline)>1:
				movie = formatMovieTitle(spltline[-1].strip())
				peopleToExperience[crewMember].append(tuple([movie,position]))
				movieToCrew[movie].append(tuple([crewMember,position]))
		elif crewMember is not "":
			movie = formatMovieTitle(line)
			peopleToExperience[crewMember].append(tuple([movie,position]))
			movieToCrew[movie].append(tuple([crewMember,position]))
		line = mmaplist[position].readline()
	sys.stdout.write("\rfile "+ str(position) +" lines procesed "+str(i))    
	sys.stdout.flush()
	print

print "Experience...Experience..."
loadExperience(directors)
loadExperience(writers)
loadExperience(producers)
print "Experience!"



def formatExperienceMovie(title):
	reext1 = re.compile('\([^\(]*\d*[a-z][a-z0-9]*\)')
	reext2 = re.compile('\[[^\(]*\d*[a-z][a-z0-9]*\]')
	reext3 = re.compile('\<[^\(]*\d*[0-9]*\>')

	title = reext1.sub('',title)
	title = reext2.sub('',title)
	title = reext3.sub('',title)
	title = title.strip()
	return title


def movielensFormatToImdbFormat(title):
	title=title.lower()

	if ', the (' in title:
		title=title.replace(', the (', ' (')
		title="the "+title

	if ', a (' in title:
		title=title.replace(', a (', ' (')
		title="a "+title

	if ', an (' in title:
		title=title.replace(', an (', ' (')
		title="an "+title

	removeextras = re.compile('\([^\(]*\d*[a-z][a-z0-9]*\) ')
	title = removeextras.sub('',title)
	return title

def getGenresPerMovie(title):
	genreslist = []
	with open('../genres.list') as f:
		for line in f:
			line=line.lower()
			if line.startswith(title):
				spltline = line.strip().split('\t')
				genreslist.append(spltline[-1].strip())
			if len(genreslist)>10:
				break
	return genreslist

def getRatingsPerMovie(title):
	with open('../ratings.list') as f:
		spltline = ""
		for line in f:
			line = line.lower()
			line = formatExperienceMovie(line)
			if line.endswith(title):
				line=line.replace(title,'').strip()
				if not re.match("\d\.\d$", line):
					next
				numbers=re.findall("\d\.\d",line)
				spltline = numbers[-1]
		return spltline


def getRatedMovieVector(movieName):
	vector = np.zeros(19)
	if movieName in imdbMovieGenres and movieName in imdbMovieRatings:
		vector=imdbMovieGenres[movieName]*imdbMovieRatings[movieName]
	return vector

def getGenreMovieVector(movieName):
	vector = np.zeros(19)
	if movieName in imdbMovieGenres and movieName in imdbMovieRatings:
		vector=imdbMovieGenres[movieName]
	return vector

def getPersonVectorExperience(personName):
	vector = np.zeros(19)
	countVector = np.zeros(19)
	if personName in peopleToExperience:
		movielist = peopleToExperience[personName]
		for movie in movielist:
			vector+=getRatedMovieVector(movie[0])
			countVector+=getGenreMovieVector(movie[0])
	maxGenre=np.amax(countVector)
	for i,x in enumerate(vector):
		if countVector[i]!=0:
			vector[i] = vector[i]/countVector[i]
			vector[i] = vector[i]*countVector[i]/float(maxGenre)
		else:
			vector[i] = 0
	return vector

def getTitleExperienceVector(title):
	vector = np.zeros(19)
	if title in movieToCrew:
		crew = movieToCrew[title]
		for person in crew:
			vector+=(getPersonVectorExperience(person[0])/len(crew))
	return vector


moviesVector = []
moviesTitle = []
users = []
userTotalReview = []
userReviews = defaultdict(list)
vectorsUsers=defaultdict(lambda:np.zeros(19))
vectorCountUsers=defaultdict(lambda:np.zeros(19))
moviesReviews = []

moviesVector.append([0])
moviesTitle.append([0])
users.append([0])
userTotalReview.append(0)
moviesReviews.append([0])

def load_movies_ml-20m():
	with open('../ml-20m/u.item') as f:
		for line in f:
			lines =  line.strip().split('|')
			title = lines[1]
			genres = lines[5:]
			moviesTitle.append(title)
			moviesVector.append(np.array([float(x) for x in genres]))
			moviesReviews.append([])
def load_users_ml-20m():
	with open('../ml-20m/u.user') as f:
		for line in f:
			lines =  line.strip().split('|')
			users.append(np.array([x for x in lines[1:]]))
			userTotalReview.append(0)
def load_ratings_ml-latest():
	with open('../ml-latest/u.data') as f:
		for line in f:
			lines =  line.strip().split('\t')
			userId = int(lines[0])
			movieId = int(lines[1])
			rating = int(lines[2])
			userTotalReview[userId]+=1
			vectorCountUsers[userId]+=moviesVector[movieId]
			vectorsUsers[userId]+=(moviesVector[movieId]*float(rating)/5.0)
			moviesReviews[movieId].append(tuple([userId,rating]))
			userReviews[userId].append(tuple([movieId,rating]))

def load_movies_ml-20M():
	with open('../ml-20M/movies.dat') as f:
		for line in f:
			lines = line.strip().split("::")
			movieid = int(lines[0])
			moviesTitle[movieid] = lines[1]
			listgenres = lines[2].split("|")
			for gr in listgenres:
				if gr.lower() in matchGenres:
					vector = np.zeros(19)
					vector[matchGenres[gr.lower()]]=1
					moviesVector[movieid]+=vector
					moviesReviews[movieid] = []

def load_ratings_ml-20M():
	with open('../ml-20M/ratings.dat') as f:
		i=0
		for line in f:
			i+=1
			if (i)%20000 == 0:
				sys.stdout.write("\rRatings, lines procesed "+str(i))    
				sys.stdout.flush()
			lines = line.strip().split("::")
			userid=int(lines[0])
			movieid=int(lines[1])
			rating=float(lines[2])
			userTotalReview[userid]+=1
			vectorsUsers[userid]+=(moviesVector[movieid]*float(rating)/5.0)
			vectorCountUsers[userid]+=moviesVector[movieid]
			moviesReviews[movieid].append(tuple([userid,rating]))
			userReviews[userid].append(tuple([movieid,rating]))
		print

print "loading movielens"
load_movies_ml-20m()
load_users_ml-latest()
load_ratings_ml-20m()

print "movilens loaded"


for userid in vectorsUsers:
	maxGenreUser = max(vectorCountUsers[userid])
	for i,x in enumerate(vectorsUsers[userid]):
		if vectorCountUsers[userid][i]!=0:
			vectorsUsers[userid][i]=vectorsUsers[userid][i]/vectorCountUsers[userid][i]
			vectorsUsers[userid][i]=vectorsUsers[userid][i]*vectorCountUsers[userid][i]/maxGenreUser

total = len(moviesTitle)
print "total movies in movielens dataset "+ str(total)
i = 0
countNotFound = 0
countFound = 0
moviesdictgenres = dict()
moviesdictratings = dict()

np.set_printoptions(precision=3,linewidth = 180)
found=0
noFnd=0
x=list()
y=list()
r=list()

histelements=defaultdict(Counter)
movieAvgRating=list()
counter=0
ratingsVScosine = defaultdict(list)
statistcsUser = defaultdict(lambda:defaultdict(int))
ratingcounter = 0
ratingmisses = 0
ratingOK = 0
print "bayes structure:"
for userid in userReviews:
	for review in userReviews[userid]:
		tempRatingsMovies = defaultdict(list)
		tempRatingsCrew = defaultdict(list)
		tempReviewsBaseModel = [movie for movie in userReviews[userid] if movie != review]
		tempReviewsBaseModel = [movie for movie in tempReviewsBaseModel if movielensFormatToImdbFormat(moviesTitle[movie[0]]) in movieToCrew]
		if movielensFormatToImdbFormat(moviesTitle[review[0]]) not in movieToCrew: continue

		for movie in tempReviewsBaseModel:
			rating =movie[1]
			movieId = movie[0]
			tempRatingsMovies[round(rating,0)].append(movieId)
		for rt in tempRatingsMovies:
			for mv in tempRatingsMovies[rt]:
				imdbTitle = movielensFormatToImdbFormat(moviesTitle[mv])
				if imdbTitle in movieToCrew:
					tempRatingsCrew[rt].extend([x[0] for x in movieToCrew[imdbTitle]])
		movietotest = review[0]
		crewtotest = []
		imdbtTitleToTest = movielensFormatToImdbFormat(moviesTitle[mv])
		if imdbtTitleToTest in movieToCrew:
			crewtotest = [x[0] for x in movieToCrew[imdbtTitleToTest]]

		results = defaultdict(float)
		for rr in tempRatingsMovies.iterkeys():
			probabilityList = []
			for testmember in crewtotest:
				NumberMoviesWithMemberandRating = len([x for x in tempRatingsCrew[rr] if x == testmember])+1
				NumberMovieswithRating = len(tempRatingsMovies[rr])
				totalmoviesBaseModel = sum([len(x) for x in tempRatingsMovies.itervalues()])+5
				MoviesWithMemberDifferentRating = 4
				for rev in [x for x in tempRatingsMovies.iterkeys() if x != rr]:
					MoviesWithMemberDifferentRating+=len([x for x in tempRatingsCrew[rev] if x == testmember])
				probXgivenRating = float(NumberMoviesWithMemberandRating)/NumberMovieswithRating
				probRating = float(NumberMovieswithRating)/totalmoviesBaseModel
				probNoRating = float(totalmoviesBaseModel-NumberMovieswithRating)/totalmoviesBaseModel
				probXgivenNoRating = float(MoviesWithMemberDifferentRating)/(totalmoviesBaseModel-NumberMovieswithRating)
				pp = probXgivenRating*probRating
				pp = pp/(pp+probNoRating*probXgivenNoRating)
				probabilityList.append(pp)
			prob = 0
			n =  sum([math.log(1-x)+math.log(x) for x in probabilityList])
			prob = 1.0/(1.0+math.exp(n))
			results[rr]=prob
		bestrating = 1
		bestratingvalue = 0
		for ww in results:
			if results[ww]>=bestratingvalue:
				bestrating=ww
				bestratingvalue = results[ww]
		if float(float(review[1]))-float(bestrating)==0:
			ratingOK+=1
			statistcsUser[userid]["ratingok"]+=1
		else:
			ratingmisses+=1
			statistcsUser[userid]["ratnotok"]+=1
		ratingcounter+=1
	print "\r"+ str(userid),
	print "\tOK "+str(statistcsUser[userid]['ratingok']),
	print "\tNO "+str(statistcsUser[userid]['ratnotok'])
print "Bayes OK "+str(ratingOK)
print "Bayes NoOK "+str(ratingmisses)
print "Total "+str(ratingcounter)

avgOk = 0.0
for ww in statistcsUser:
	avgOk +=float(statistcsUser[ww]['ratingok'])/(statistcsUser[ww]['ratingok'] + statistcsUser[ww]['ratnotok'])

avgOk = avgOk/len(statistcsUser)

print "Number of Users "+str(len(statistcsUser))
print "Average success per user "+str(avgOk)


