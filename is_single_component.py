import re, sys, time

from collections import *
from itertools import *
from functools import *
from math import prod

from lexpy import DAWG
from ortools.sat.python import cp_model

from scrabble import *
from multiprocessing import Pool

DEBUG = False

'''
An implementation of scrabble rules in OR-tools
you can provide board states of high scoring potential as such:
OXyPhenButAZonE opacifying xenophile prequalified brainwashed ameliorative zoogametes ejaculating

Interpreted as a board state like this, which must be made 'valid'.
-pacifying
-enophile
y
-requalified
h
e
n
-rainwashed
u
t
-meliorative
-oogametes
o
n
-jaculating

By ordering such options on score, the first verified valid result is the proven maximum solution.

And this will return a new board states for which:
- all words are valid given the dictionary
- all letter counts are within available tiles (blanks are used if needed)
- there is at least one unused tile
- all words are connected in a single component, which is connected to the start tile

This program does NOT take into account the following. Which means you should still check the result.
- There is an actual series of turns that works, which is not always true
	for example:
	- The word on the start tile is 9 letters and has a single hook
	- None of the substrings of the 9 letter word crossing the start tile are valid
	- Then we can conclude that this game can never be played.

And the program generating the options has the following assumptions
- The optimal solution uses a 15 letter word at one of the edges of the board
- The optimal solution has no blanks in the 15 letter word itself
'''

Cell = namedtuple('Cell', ['active', 'letter', 'x', 'y', 'blank', 'depth'])
LONGEST_PATH = 64

def create_row(max_depth, fixed_words, row_str):
	'''
	Creates an constant depth automaton that describes a valid `max_depth` sized row or column in a
	scrabble game. Each word (seperated by whitespace) must be in the dictionary.
	Visualise the nodes as the edges of the board spaces. node[0] means the left edge.
	node[max_depth] means the rightmost edge
	'''
	nodes, edges, terminal, done = defaultdict(lambda: len(nodes)), set(), set(), set()
	def get_dawg(fixed_words):
		dawg = DAWG()
		dawg.add_all(fixed_words)
		dawg.reduce()
		return dawg

	def encode_dawg(dawg_id, node, depth):
		if node.eow:
			if depth < max_depth:
				if row_str[depth] in '- ':
					edges.add((nodes[(dawg_id, node.id, depth)], 26, nodes[depth+1]))
			else:
				terminal.add(nodes[(dawg_id, node.id, depth)])

		for c,child in node.children.items():
			edges.add((nodes[(dawg_id, node.id, depth)], ord(c)-ord('a'), nodes[(dawg_id, child.id, depth+1)]))
			encode_dawg(dawg_id, child, depth+1)


	start = nodes[0]
	for depth in range(max_depth):
		if row_str[depth] in '- ' and (depth == 0 or row_str[depth-1] in '- '):
			edges.add((nodes[depth], 26, nodes[depth+1]))
	
	for depth in range(max_depth):
		if depth in nodes:
			for c,start_node in get_dawg(fixed_words[depth]).root.children.items():
				edges.add((nodes[depth], ord(c)-ord('a'), nodes[(depth, start_node.id, depth+1)]))
				encode_dawg(depth, start_node, depth+1)

	terminal.add(nodes[max_depth])
	names = {v:k for k,v in nodes.items()}
	return (start, terminal, edges, names)

def single_component(model, cells):
	def neighbours(x,y):
		if x > 0: yield (x-1, y)
		if x < N-1: yield(x+1, y)
		if y > 0: yield (x, y-1)
		if y < N-1: yield(x, y+1)

	for (x,y), cell in cells.items():
		if (x,y) == (N//2,N//2):
			continue
		# all active cells must have a valid depth
		model.add(cell.depth == LONGEST_PATH).only_enforce_if(~cell.active)
		model.add(cell.depth < LONGEST_PATH).only_enforce_if(cell.active)
		# and must be connected
		connected = []
		for p in neighbours(x,y):
			v = model.new_bool_var(f'isconnected_{x}_{y}->{p}')
			model.add(cells[p].depth == cell.depth-1).only_enforce_if([v, cell.active])
			connected.append(v)

		model.add(sum(connected) > 0).only_enforce_if(cell.active)

	center = cells[(N//2,N//2)]
	model.add(center.active == 1)
	model.add(center.depth == 0)

def limit_letter_count_blankfree(model, cells, letter_count):
	for c, limit in letter_count.items():
		if c == '*': continue
		lettercount = 0
		for cell in cells.values():
			v = model.new_bool_var(f'letter_is_active_{cell.x}_{cell.y}_{c}')
			model.add(v == 0).only_enforce_if(cell.blank)
			model.add(v == cell.letter[c]).only_enforce_if(~cell.blank)
			lettercount += v
		model.add(lettercount <= limit)

def automaton_to_dot(automaton, names):
	print('start at', automaton[0])
	print('ends at', *automaton[1])
	for k,v in names.items():
		print(f'{k} [label="{k},{v}"]')

	for f,v,t in automaton[2]:
		print(f'{f} -> {t} [label="{chr(ord("a")+v)}"]')

def all_paths(automaton, file):
	connected = defaultdict(set)
	for f,v,t in automaton[2]:
		connected[f].add((v,t))
	
	total = 0
	def enum_paths(node, state=''):
		if total == 10_000_000:
			return
		if len(state) == 15:
			total += 1
			print(state, file=file)

		for v,t in connected[node]:
			enum_paths(t, state + chr(ord('a')+v))

	enum_paths(automaton[0])

def create_board(model, counts, fixed_words, board, size=N):
	# create 2N automatons
	lines = []
	row_cache = {}
	for d in range(2*size):
		fixed_words_row = defaultdict(set)
		for (x,y,h,n),words in fixed_words.items():
			if h == (d < size) and (y if (d < size) else x) == d%size:
				fixed_words_row[(x if (d < size) else y)] |= set(words)

		#print(f'{d}/{2*size}')
		#for k,v in fixed_words_row.items():
		#	print('   ', k, len(v), list(v)[:10])
		row_str = board[size*d:size*(d+1)] if d < size else ''.join(board[i*size+d%size] for i in range(size))
		automaton = create_row(size, fixed_words_row, row_str)
		
		names = automaton[3]
		automaton = automaton[:3]
		if DEBUG:
			for i in range(size):
				print(board[i*size:(i+1)*size])
			with open(f'{d}.txt', 'w') as f:
				all_paths(automaton, f)
				#automaton_to_dot(a,utomaton, names)

		lines.append([model.new_int_var(0, len(abc), f'{d}_letter_{i}') for i in range(size)])
		model.add_automaton(lines[-1], *automaton)

	# they must be equivalent at their intersections
	rows, columns = lines[:size], lines[size:]
	for y,row in enumerate(rows):
		for x,v in enumerate(row):
			model.add(columns[x][y] == v)

	# channel resulting integers to bitsets
	cells = {}
	for i,row in enumerate(rows):
		for j,v in enumerate(row):
			c = [model.new_bool_var(f'{i}_{j}_{c}') for c in range(len(abc)+1)]
			model.Add(sum(c) == 1)
			for k,active in enumerate(c):
				model.add(v == k).only_enforce_if(active)

			cells[(j,i)] = Cell(
				~c[-1], 
				{abc[i]:v for i,v in enumerate(c[:len(abc)])}, 
				j, i, 
				model.new_bool_var(f'{i}_{j}_blank'),
				model.new_int_var(0, LONGEST_PATH, f'{i}_{j}_depth'))

	limit_letter_count_blankfree(model, cells, counts)
	model.add(sum(c.blank for c in cells.values()) <= 2)
	return cells, list(chain(rows))

def estimate_score(cells, N):
	'''
	TODO:
	1: De bingo tiles toeschrijven aan een richting om zo netjes groepjes van 7 te maken
	'''
	score = 0
	print('Creating word bindings')
	# there are "only" 2730 word positions on a 15^2 board
	# any can be active, or not. Easy to test by looking at cell activity
	pos = [(x,y,h,n) for x,y,h,n in product(range(N), range(N), [0,1], range(2,N+1)) if x*h + y*(1-h) + n <= N]
	pos_active = {p: model.new_bool_var(f'{p}_active') for p in pos}
	
	for j,((x,y,h,n),v) in enumerate(pos_active.items()):
		if j%50 == 0:
			print(f'{j}/{len(pos_active)}')
		w_cells = [cells[(x+i*h, y+i*(1-h))] for i in range(0, n)]
		lb = (~cells[(x-1,y)].active if x > 0 and h else h)
		rb = (~cells[(x+n,y)].active if x+n < N and h else h)
		tb = (~cells[(x,y-1)].active if y > 0 and not h else 1-h)
		bb = (~cells[(x,y+n)].active if y+n < N and not h else 1-h)
		model.add(sum(c.active for c in w_cells) + lb + rb + tb + bb == n+2).only_enforce_if(v)
		model.add(sum(c.active for c in w_cells) + lb + rb + tb + bb  < n+2).only_enforce_if(~v)

	# to find out the word-multiplication factor of each word position
	# we need to assign the multipliers. We start by assigning each multiplier
	# to either the vertical or horizontal direction.
	mult_dir = {(x,y):model.new_bool_var(f'mult_dir_{x}_{y}') for x,y in product(range(N), range(N))}

	print('Binding words to score')
	# and we can sum score as a sum over those active word locations
	for j, ((x,y,h,n),v) in enumerate(pos_active.items()):
		if j%50 == 0:
			print(f'{j}/{len(pos_active)}')

		if n >= 8:
			score += v*50

		positions = [(x+i*h, y+i*(1-h)) for i in range(0, n)]

		# get active multiplication
		options = {(x,y): word_multiplier[y,x] for (x,y) in positions if (x,y) in mult_dir}
		subsets = list(chain(*[combinations(options.keys(), i) for i in range(len(options)+1)]))
		multis  = sorted({prod(options[p] for p in s) for s in subsets})
		subset_vars = {s:model.new_bool_var(f'p{x}_{y}_{h}_{n}_s{s}') for s in subsets}
		multi_vars = {m:model.new_bool_var(f'p{x}_{y}_{h}_{n}_m{m}') for m in multis}
		model.add(sum(multi_vars.values()) == 1)
		for m in multi_vars:
			subsets_m = [subset_vars[s] for s in subsets if prod(options[p] for p in s) == m]
			model.add(multi_vars[m] <= sum(subsets_m))

		for m, multi_active in multi_vars.items():
			for cell in [cells[p] for p in positions]:
				binding = {c:model.new_bool_var(f'p{x}_{y}_{h}_{n}_l{c}_m{m}') for c in abc}
				for c,letter_active in binding.items():
					model.add(letter_active == 1).only_enforce_if([cell.letter[c], v, ~cell.blank, multi_active])
					for w in [~cell.letter[c], ~v, cell.blank, ~multi_active]:
						model.add(letter_active == 0).only_enforce_if(w)
					score += letter_active*scores[c]*letter_multiplier[y,x]*m
		
	return score, pos_active

#####################
## Preprocessing
#####################

# group by length, then create matrices that (for each word) give the word, and their letter counts
def build_index(words):
	by_length = {i:[] for i in range(1,16)}
	[by_length[len(w)].append(w) for w in words]
	by_length_arr = {k:np.array([word_to_np(w) for w in v], dtype=np.int8) if len(v) else np.zeros((0,k),dtype=np.int8) for k,v in by_length.items()}
	by_length_cnt = {k:np.array([word_to_npcount(w) for w in v], dtype=np.int8) if len(v) else np.zeros((0,26),dtype=np.int8) for k,v in by_length.items()}
	return by_length, by_length_arr, by_length_cnt

# a 26 sized array that specifies the amount of non-blank tiles remaining, combined with a simply count of blanks
def remaining_tiles_array(main_word, board):
	tilecount_arr = np.array([counts[chr(i+97)] for i in range(26)], dtype=np.int8)
	tilecount_arr -= word_to_npcount(''.join(c.lower() for c in main_word if c.isupper()) + board.replace('-', '').replace(' ', ''))
	blanks_left = 2+sum(tilecount_arr[tilecount_arr < 0])
	tilecount_arr[tilecount_arr < 0] = 0
	return tilecount_arr, blanks_left

# given a constrained board state, what words can be placed anywhere
def find_fitting_words(main_word, board, _board, _words, positions):
	tilecount_arr, blanks_left = remaining_tiles_array(main_word, board)
	by_length, by_length_arr, by_length_cnt = WORD_CACHE if _words == None else build_index(_words)

	words_per_location = {}
	for x,y,h,n in positions:
		boardword = [_board[(y+i*(1-h))*N+(x+i*h)] for i in range(n)]
		boardword = [(i,c) for i,c in enumerate(boardword) if c != ' ']
		if boardword:
			ind = np.ones(len(by_length_cnt[n]), dtype=np.uint8)
			for i,c in boardword:
				sub_ind = np.zeros(len(by_length_cnt[n]), dtype=np.uint8) 
				for d in c:
					sub_ind |= (by_length_arr[n][:,i] == ord(d)-97)
				
				ind &= sub_ind
			ind = np.where(ind)[0]
		else:
			ind = np.arange(len(by_length_cnt[n]), dtype=np.int32) 

		fixed = ''.join([board[(y+i*(1-h))*N+(x+i*h)] for i in range(n)]).replace(' ', '')
		letters_placed = word_to_npcount(fixed)
		ind = ind[np.sum(np.clip((tilecount_arr - by_length_cnt[n][ind] + letters_placed),-3,0), axis=1) >= -2]

		words_per_location[(x,y,h,n)] = [by_length[n][i] for i in ind]
	return words_per_location

# given all possible placeable words. What letters are possible per cell.
def letters_per_cell(board, words_per_location, positions):
	cell_positions = defaultdict(list)
	[cell_positions[(x+i*h, y+i*(1-h))].append((x,y,h,n)) for x,y,h,n in positions for i in range(n)]

	board2 = defaultdict(set)
	for (x,y), pos in cell_positions.items():
		if board[y*N+x] == '-':
			board2[y*N+x] = {'-'}
		elif board[y*N+x] != ' ':
			board2[y*N+x] = {board[y*N+x]}
		else:
			h = {w[(x-a[0])+(y-a[1])] for a in pos for w in words_per_location[a] if a[2]}
			v = {w[(x-a[0])+(y-a[1])] for a in pos for w in words_per_location[a] if not a[2]}
			board2[y*N+x] = h&v
	return board2

def words_for_board(main_word, board):
	_board = board
	_words = None
	positions = valid_positions(board)
	# generally more than 1 iteration only filters a handful of additional words
	for i in range(1):
		words_per_location = find_fitting_words(main_word, board, _board, _words, positions)
		_board = letters_per_cell(board, words_per_location, positions)
		_words = {w for v in words_per_location.values() for w in v}

	return _board, words_per_location

def still_possible(main_word, board):
	groups = [(g.start(), len(g.group())) for g in re.finditer(r'[a-z]+', main_word)]
	return all(any(board[N+i] for i in range(s,s+n)) for s,n in groups)

def quick_validity_check(main_word, board):
	positions = valid_positions(board)
	positions = [v for v in positions if v[1] <= 1]
	words_per_location = find_fitting_words(main_word, board, board, None, positions)
	_board = letters_per_cell(board, words_per_location, positions)
	return still_possible(main_word, _board)

def get_board_str(main_word, verticals):
	board = list(' '*(N*N))
	for i,c in enumerate(main_word):
		board[i] = c if c.islower() else '-'
	
	for i,v in zip([i for i,c in enumerate(main_word) if c.isupper()], verticals):
		for j,c in enumerate((v+'-')[1:N]):
			board[(j+1)*N+i] = c

	# any empty rows beyond are longest vertical we don't use
	if verticals:
		for i in range(max(map(len, verticals)), N):
			for j in range(N):
				board[i*N+j] = '-'

	board = ''.join(board)
	return board

def limit_valid_board_states(model, letter_grid, boards):
	assert len(boards) > 0, "Restricting to 0 possible boards is impossible"	
	# we make a new letter-grid where each letter has 28 states
	# |abc| + <empty> + <open>
	# the letter and empty states just map 1:1
	# but the open state simply means anything is allowed
	options_per_cell = [{board[i] for board in boards} for i in range(len(boards[0]))]
	active = [(i,x) for i,x in enumerate(options_per_cell) if x != {' '}]
	subvars = {i:model.new_int_var(0, len(abc)+2, f'{i}_binding') for i,x in active}

	for i, v in subvars.items():
		is_active = model.new_bool_var(f'{i}_binding_active')
		model.add(v != len(abc)+1).only_enforce_if(is_active)
		model.add(v == len(abc)+1).only_enforce_if(~is_active)
		model.add(letter_grid[i] == v).only_enforce_if(is_active)

	model.add_allowed_assignments(
		subvars.values(), 
		[[(abc+"- ").index(board[i]) for i,_ in active] for board in boards])

###########
### SOLVING
###########

def solve_instance(line):
	score, main_word, verticals, time_limit = line
	print('solving', score, main_word, *verticals, time_limit)

	board = get_board_str(main_word, verticals)
	board_limits, words_by_row = words_for_board(main_word, board)
	
	model = cp_model.CpModel()
	_counts = {k:v-main_word.count(k.upper()) for k,v in counts.items()}
	cells, _ = create_board(model, _counts, words_by_row, board, N)
	single_component(model, cells)

	model.add(sum(cell.active for cell in cells.values()) <= sum(_counts.values())-1)
	#model.Minimize(sum(cell.active for cell in cells.values()))

	for (x,y), cell in cells.items():
		if board[y*N+x] == ' ':
			pass
		elif board[y*N+x] == '-':
			model.add(cell.active == 0)
		else:
			model.add(cell.active == 1)
			model.add(cell.letter[board[y*N+x]] == 1)

	solver = cp_model.CpSolver()
	solver.parameters.log_search_progress = False
	solver.parameters.max_presolve_iterations = 1
	solver.parameters.num_search_workers = 1 if time_limit else 8
	solver.parameters.optimize_with_core = False
	if time_limit:
		solver.parameters.max_time_in_seconds = time_limit

	t1 = time.time()
	status = solver.Solve(model)
	t2 = time.time()
	#print(int(t1-t0),'s total, solving', int(t2-t1), 's')
	if status == cp_model.UNKNOWN:
		return (False, '', line)

	if status in [cp_model.FEASIBLE, cp_model.OPTIMAL]:
		lines = []
		for j in range(N):
			lines.append(''.join([(''.join(c for c,v in cells[(i,j)].letter.items() if (b if isinstance(v, bool) else solver.Value(v))) + ' ')[0] for i in range(N)]))
		#print('\n'.join(lines))
		return (True, '\n'.join(lines), line)

	return (True, '', line)

def initial_filter(x):
	score, main_word, verticals, _ = x
	board = get_board_str(main_word, verticals)
	if not quick_validity_check(main_word, board):
		return None

	return x

def main_connectivity_solver():
	print(f'using {word_file}')
	remaining = []
	for i,line in enumerate(sys.stdin):
		if not line.strip(): continue
		if i % 10000 == 0:
			print(i)
		t0 = time.time()
		line = line.strip()
		(score, blanks, main_word, verticals) = line.split(' ', 3)
		verticals = verticals.split(' ')
		verticals = sorted([(int(x.split('_')[1]), x.split('_')[0]) for x in verticals])
		verticals = [x[1] for x in verticals]
		if len(verticals) != 7:
			continue

		remaining.append((score, main_word, verticals, 100))

	difficult = []
	with Pool(8) as p:
		_remaining = []
		i = 0
		for x in p.imap(initial_filter, remaining):
			if i % 1000 == 0:
				print(f'{i}/{len(remaining)} ({len(_remaining)} remaining)')
			i += 1
			if x:
				_remaining.append(x)

		print(len(_remaining), 'left after quick analysis, filtering problems that solve in pre-solve')
		for (is_solved, solution, x) in p.imap(solve_instance, _remaining):
			if is_solved and solution:
				print(*x)
				print(solution)
				print('found a solution, filtering worse scoring open problems')
				difficult = [y for y in difficult if y[0] >= x[0]]
				break

			if not is_solved:
				print(f'STORE {x[1]} {" ".join(x[2])}')
				difficult.append((*x[:2], 0))

	print(len(difficult), 'left for deep analysis')
	for v in difficult:
		is_solved, solution, x = solve_instance(v)
		if is_solved and solution:
			print(solution)

WORD_CACHE = build_index(set(words)|set(abc))
if __name__ == '__main__':
	main_connectivity_solver()
