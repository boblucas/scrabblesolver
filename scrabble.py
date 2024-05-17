from collections import *
from itertools import *
from functools import *
import numpy as np
from ortools.sat.python import cp_model
import math
LANG = 'EN'

if LANG == 'NL':
	word_file = 'words2024'
	# Dutch letter counts and scores
	letters = [
		('a',1,6),('b',3,2),('c', 5,2),('d',2,5),('e',1,18),('f',4,2),('g',3,3),
		('h',4,2),('i',1,4),('j', 4,2),('k',3,3),('l',3, 3),('m',3,3),('n',1,10),
		('o',1,6),('p',3,2),('q',10,1),('r',2,5),('s',2, 5),('t',2,5),('u',4,3),
		('v',4,2),('w',5,2),('x', 8,1),('y',8,1),('z',4, 2),('*',0,2)]
elif LANG.startswith('EN'):
	#word_file = 'CSW21.txt'
	word_file = 'NWL2023'
	#word_file = 'NWL2020'

	letters = [
		('a',1,9),('b',3,2),('c', 3,2),('d',2,4),('e',1,12),('f',4,2),('g',2,3),
		('h',4,2),('i',1,9),('j', 8,1),('k',5,1),('l',1, 4),('m',3,2),('n',1,6),
		('o',1,8),('p',3,2),('q',10,1),('r',1,6),('s',1, 4),('t',1,6),('u',1,4),
		('v',4,2),('w',4,2),('x', 8,1),('y',4,2),('z',10, 1),('*',0,2)]

abc = ''.join([c for c,p,n in letters if c != '*'])
counts = Counter({c:n for c,p,n in letters})
scores = {c:p for c,p,n in letters}
score_arr = np.array([scores[chr(c).lower()] if chr(c).lower() in scores else 0 for c in range(0,128)])

hand_size = 7
emptyhand_bonus = 50

# scrabble board properties
N = 15
_ = 1
word_multiplier = np.array([
	[3,_,_,_,_,_,_,3,_,_,_,_,_,_,3],
	[_,2,_,_,_,_,_,_,_,_,_,_,_,2,_],
	[_,_,2,_,_,_,_,_,_,_,_,_,2,_,_],
	[_,_,_,2,_,_,_,_,_,_,_,2,_,_,_],
	[_,_,_,_,2,_,_,_,_,_,2,_,_,_,_],
	[_,_,_,_,_,_,_,_,_,_,_,_,_,_,_],
	[_,_,_,_,_,_,_,_,_,_,_,_,_,_,_],
	[3,_,_,_,_,_,_,2,_,_,_,_,_,_,3],
	[_,_,_,_,_,_,_,_,_,_,_,_,_,_,_],
	[_,_,_,_,_,_,_,_,_,_,_,_,_,_,_],
	[_,_,_,_,2,_,_,_,_,_,2,_,_,_,_],
	[_,_,_,2,_,_,_,_,_,_,_,2,_,_,_],
	[_,_,2,_,_,_,_,_,_,_,_,_,2,_,_],
	[_,2,_,_,_,_,_,_,_,_,_,_,_,2,_],
	[3,_,_,_,_,_,_,3,_,_,_,_,_,_,3],
])
letter_multiplier = np.array([
	[_,_,_,2,_,_,_,_,_,_,_,2,_,_,_],
	[_,_,_,_,_,3,_,_,_,3,_,_,_,_,_],
	[_,_,_,_,_,_,2,_,2,_,_,_,_,_,_],
	[2,_,_,_,_,_,_,2,_,_,_,_,_,_,2],
	[_,_,_,_,_,_,_,_,_,_,_,_,_,_,_],
	[_,3,_,_,_,3,_,_,_,3,_,_,_,3,_],
	[_,_,2,_,_,_,2,_,2,_,_,_,2,_,_],
	[_,_,_,2,_,_,_,_,_,_,_,2,_,_,_],
	[_,_,2,_,_,_,2,_,2,_,_,_,2,_,_],
	[_,3,_,_,_,3,_,_,_,3,_,_,_,3,_],
	[_,_,_,_,_,_,_,_,_,_,_,_,_,_,_],
	[2,_,_,_,_,_,_,2,_,_,_,_,_,_,2],
	[_,_,_,_,_,_,2,_,2,_,_,_,_,_,_],
	[_,_,_,_,_,3,_,_,_,3,_,_,_,_,_],
	[_,_,_,2,_,_,_,_,_,_,_,2,_,_,_],
])

# valid words, also ordered by some special properties for performance
words = set(open(word_file).read().lower().split('\n')) - {''}
words |= set(abc)

#words -= black_list
prefixable = defaultdict(set, {k:set(g) for k,g in groupby(sorted(x for x in words if x[1:] in words), key=lambda x: x[0])})
postfixable = defaultdict(set, {k:set(g) for k,g in groupby(sorted(x for x in words if x[:-1] in words), key=lambda x: x[-1])})
#pillars = defaultdict(set, {k:set(g) for k,g in groupby(sorted([w for w in words if len(w) == N and w[1:-1] in words and (w[1:] or w[:-1] in words)], key=lambda w: w[0]+w[-1]), key=lambda w: w[0]+w[-1])})

words_blanks = words
def cache_blanks():
	global words_blanks
	if len(words_blanks) == len(words):
		words_blanks |= {w[:i] + '*' + w[i+1:] for w in words for i in range(0, len(w))}


def word_to_npcount(w): return np.array([w.count(chr(97+i)) for i in range(26)], dtype=np.int8)
def word_to_np(w): return np.array([ord(c)-97 for c in w], dtype=np.int8)

def all_positions():
	positions = []
	for x,y in product(range(N),range(N)):
		positions += [(x,y,True, x2-x+1) for x2 in range(x,N)]
		positions += [(x,y,False,y2-y+1) for y2 in range(y,N)]
	return positions

def valid_positions(board):
	'''
	all valid position on an given board with static tiles
	'''
	return [(x,y,h,n) for x,y,h,n in all_positions() if not (
		(h and x > 0 and not board[y*N+x-1] in ' -') or 
		(h and x+n < N and not board[y*N+x+n] in ' -') or 
		(not h and y > 0 and not board[(y-1)*N+x] in ' -') or 
		(not h and y+n < N and not board[(y+n)*N+x] in ' -') or
		('-' in [board[(y+i*(1-h))*N+(x+i*h)] for i in range(n)] ))]

def has_sufficient_tiles(text, allowed_blanks = 2, _counts = counts):
	required_blanks = 0
	for k,v in Counter(text).items():
		if k in '* ': continue
		required_blanks += max(0, v - _counts[k])
		if required_blanks > allowed_blanks:
			return False
	return True

# returns text with minimum amount of letters replaced by blanks
def sufficient_tiles(text, allowed_blanks = 2):
	for k,v in Counter(text).items():
		if k in '* ': continue
		if (blanks := max(0, v - counts[k])):
			if blanks > allowed_blanks:
				break
			
			for pos in combinations([i for i, x in enumerate(text) if x == k], blanks):
				w = text
				for i in pos:
					w = w[:i] + '*' + w[i+1:]
				yield from sufficient_tiles(w, allowed_blanks - blanks)
			break
	else:
		yield text

def valid_subwords(w):
	for i in range(len(w)):
		for j in range(i+1, len(w)+1):
			if w[i:j] in words:
				yield (i, w[i:j])

def get_word_multiplier(x, y, h, n):
	return int(np.prod(word_multiplier[y,x:x+n] if h else word_multiplier[y:y+n,x]))

# calculates score of isolated word at specific location
# uppercase letters are assumed to be the added letters
def word_score(w, x, y, h):
	ws, score, tiles_layed = 1, 0, 0
	for i,c in enumerate(w):
		ls = 1
		if c.isupper():
			tiles_layed += 1
			ws *= word_multiplier[y+i*(1-h)][x+i*h]
			ls *= letter_multiplier[y+i*(1-h)][x+i*h]
		score += scores[c.lower()]*ls
	return score*ws + (tiles_layed==hand_size)*emptyhand_bonus

def word_score_arr(w, x, y, h):
	_w = np.frombuffer(w.encode('ascii'), dtype='uint8')
	n = _w.shape[-1]
	letter_score = score_arr[_w]
	pos = np.arange(x*h+y*(h-1), (x*h+y*(h-1))+n)[np.where(_w < 0x60)]
	if h:
		ws = word_multiplier[y, pos].prod()
		letter_bonus = (letter_score[_w < 0x60] * (letter_multiplier[y, pos]-1))
	else:
		ws = word_multiplier[pos, x].prod()
		letter_bonus = (letter_score[_w < 0x60] * (letter_multiplier[pos, x]-1))
	
	return (letter_score.sum() + letter_bonus.sum()) * ws + ((_w < 0x60).sum()==7)*emptyhand_bonus

# given many turns, how many points can a word generate WITHOUT assistence?
def all_word_scores(score, w, x, y, h, backtrack = []):
	cache_blanks()
	if w.isupper():
		yield (score, backtrack)
	else:
		uppers = [i for i,c in enumerate(w) if c.isupper()]
		for i in range(0, (uppers[0]+1 if uppers else len(w))):
			start = (uppers[-1] if uppers else i)+1
			end = len(w)+1
			j = start+1+(len(backtrack)==0)
			#for j in range(start, end):
			if j <= len(w):
				if 1 <= sum(c.islower() for c in w[i:j]) <= hand_size and w[i:j].lower() in words_blanks:
					part_score = word_score(w[i:j].swapcase(), x+i*h, y+i*(1-h), h)
					yield from all_word_scores(score + part_score, w[:i] + w[i:j].upper() + w[j:], x, y, h, backtrack + [(part_score, w[i:j])])

@cache
def max_word_score(w, x, y, h):
	if len(w) == 1:
		return (word_score(w, x, y, h), [w])
	scores = list(all_word_scores(0, w, x, y, h))
	if scores:
		return max(scores)
	return (0, [])

# find all ways to have 1..hand_size open gaps in w (true word)
def possible_partial_placements(w, x, y, h, keep_multi=True):
	# all valid substrings
	sub = sorted(list(valid_subwords(w)) + [(i, w[i]) for i in range(len(w))])

	# exclude words that cover multipliers, which are always optimal to save for last
	if keep_multi:
		sub = [(i,_w) for i,_w in sub if all(word_multiplier[y+j*(1-h)][x+j*h] == 1 for j in range(i, i+len(_w)))]

	# all combinations of those that leave 1..hand_size tiles
	def non_overlapping_comb(options, state = [], start = 0):
		yield state
		for i,(start, w) in enumerate(options[start:]):
			if not state or (state[-1][0] + len(state[-1][1])) < start:
				yield from non_overlapping_comb(options, state + [(start, w)], i+1)

	return [parts for parts in non_overlapping_comb(sub) if 1 <= len(w)-sum(len(q) for _,q in parts) <= hand_size]

# get the maximum score given you are allowed to put down
# any tiles to start
@cache
def max_word_score_ideal(w, x, y, h, keep_multi=True):
	placed = possible_partial_placements(w, x, y, h, keep_multi)

	# get scores of each part
	max_score = (0, [])
	for parts in placed:
		score = sum(max_word_score(p, x+i*h, y+i*(1-h), h)[0] for i,p in parts)
		w_marked = w.upper()
		for i,p in parts:
			w_marked = w_marked[:i] + p + w_marked[i+len(p):]

		score += word_score(w_marked, x, y, h)
		max_score = max((score, parts), max_score)
		#print(score, parts, w_marked)

	return max_score
