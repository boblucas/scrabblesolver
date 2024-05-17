from collections import *
from itertools import *
from ortools.sat.python import cp_model
import re, sys
from scrabble import *
import time
import numpy as np

WordSlot = namedtuple('WordSlot', ['name', 'active', 'word', 'x', 'y', 'h', 'n', 'static'])
Cell = namedtuple('Cell', ['active', 'letter', 'x', 'y', 'bindings', 'h', 'v'])

def prepare_words(model, words, var_prefix):
	n = max(len(w) for w in words)
	# if the word is static short circuit to using constants
	if len(words) == 1:
		w = [ord(c)-ord('a') for c in list(words)[0]]
		return None, [model.NewIntVar(w[i], w[i], f'{var_prefix}_letter_{i}') for i in range(n)] + [n]

	# length encoding (eg: <word><padding><word length>)
	words = [w + 'a'*(n-len(w)) + chr(ord('a')+len(w)) for w in words]
	words = sorted(words)
	word = [model.NewIntVar(0, 26, f'{var_prefix}_letter_{i}') for i in range(n)] + [model.NewIntVar(0, n, f'{var_prefix}_wordlength')]
	return words, word

# solves slower, but uses much less memory
def get_dict_automaton(model, words, var_prefix):
	words, out = prepare_words(model, words, var_prefix)
	if not words:
		return out

	# create a DAWG with lexpy and extract graph
	from lexpy import DAWG
	dawg = DAWG()
	dawg.add_all(sorted(words))
	dawg.reduce()
	edges = set()
	all_nodes_index = {}
	def get_nodes(node):
		all_nodes_index[node] = len(all_nodes_index)
		for letter in node.children:
			edges.add( (node, letter, node.children[letter]) )
			get_nodes(node.children[letter])

	get_nodes(dawg.root)

	# convert to CP-model automaton
	automaton = [(all_nodes_index[f], ord(v)-ord('a'), all_nodes_index[t]) for f,v,t in edges]
	model.AddAutomaton(out, all_nodes_index[dawg.root], [all_nodes_index[v] for v in all_nodes_index if v.eow] , automaton)
	return out

def get_dict_assignment(model, words, var_prefix):
	words, out = prepare_words(model, words, var_prefix)
	if not words: return out
	model.AddAllowedAssignments(out, [[ord(c)-ord('a') for c in w] for w in words])
	return out

def make_letter(model, prefix):
	letter = {c:model.NewBoolVar(f'{prefix}_{c}') for c in abc}
	model.Add(sum(letter.values()) <= 1)
	return letter

def make_slot(model, prefix, valid_words, fix_x = -1, fix_y = -1, fix_h = -1, must_exist = False):
	valid_words = set(valid_words)
	min_length = min(len(w) for w in valid_words) if must_exist else 0
	max_length = max(len(w) for w in valid_words)
	
	#word_vars = get_dict_automaton(model, valid_words, f'{prefix}_auto')
	word_vars = get_dict_assignment(model, valid_words, f'{prefix}_assign')

	slot = WordSlot(
		prefix,
		model.NewBoolVar(f'{prefix}_active') if not must_exist else True,
		word_vars[:-1],
		model.NewIntVar(0, N-1, f'{prefix}_x') if fix_x == -1 else fix_x, 
		model.NewIntVar(0, N-1, f'{prefix}_y') if fix_y == -1 else fix_y,
		model.NewBoolVar(f'{prefix}_h') if fix_x == -1 and fix_y == -1 and fix_h == -1 else (fix_x==-1 if fix_h == -1 else fix_h),
		word_vars[-1],
		list(valid_words)[0] if len(valid_words) == 1 and must_exist else None
	)

	# coordinates must be within grid
	OnlyEnforceIf(model, slot.x + slot.n <= N, slot.h)
	OnlyEnforceIf(model, slot.y + slot.n <= N, neg(slot.h))

	# the active letters must form a word, or all letters must be 0
	ws = [[ord(c)-ord('a') for c in w] for w in valid_words]
	if not must_exist:
		ws.append([])

	return slot

# make a bool x for which: x == (v1 == v2)
def bind(model, name, v1, v2):
	v3 = model.NewBoolVar(name)
	if isinstance(v1 == v2, bool):
		return v1 == v2

	model.Add(v1 == v2).OnlyEnforceIf(v3)
	model.Add(v1 != v2).OnlyEnforceIf(v3.Not())
	return v3

# does .Not() on a variable and not x on a boolean
def neg(v):
	return not v if isinstance(v, bool) else v.Not()

def make_cell(model, x,y, slots):
	# if we are at the -1 or +1 position of any slot we must be empty, so no cell is made
	'''
	for slot in slots:
		if isinstance(slot.x, int) and isinstance(slot.y, int) and isinstance(slot.n, int):
			if  (slot.y == y and slot.x == x+1 and slot.h and slot.active) or \
				(slot.y == y and slot.x+slot.n == x and slot.h and slot.active ) or \
				(slot.x == x and slot.y == y+1 and not slot.h and slot.active) or \
				(slot.x == x and slot.y+slot.n == y and not slot.h and slot.active):
				return Cell(False, {c:0 for c in abc}, x, y, {}, False, False)

	# if we are within a static slot we have a static letter value, so return a static cell
	# TODO: support static crossing words
	for slot in slots:
		if isinstance(slot.x, int) and isinstance(slot.y, int) and isinstance(slot.n, int):
			if (slot.h and x == slot.x and y >= slot.y and y < slot.y+slot.n) or (not slot.h and y == slot.y and x >= slot.x and x < slot.x+slot.n):
				i = y - slot.y if slot.h else x - slot.x
				return Cell(True, {c:c==slot.static[i] for c in abc}, x, y, {(slot.name, i, slot.h): True}, slot.h, not slot.h)
	'''
	
	cell = Cell(model.NewBoolVar(f'cell_{x}_{y}_active'), make_letter(model, f'cell_{x}_{y}_letter'), x, y, {},  model.NewBoolVar(f'cell_{x}_{y}_h'), model.NewBoolVar(f'cell_{x}_{y}_v'))
	model.Add(sum(cell.letter.values()) == cell.active)

	# if we are at the -1 or +1 position of any slot we must be empty
	# TODO: slots with static positions can either be ignored or force an empty/word state
	for slot in slots:
		x_overlap = bind(model, f'bind_{x}_{y}_to_{slot.name}_0_1_TMP9', slot.x, x)
		x_left = bind(model, f'bind_{x}_{y}_to_{slot.name}_0_1_TMP5', slot.x, x+1)
		x_right = bind(model, f'bind_{x}_{y}_to_{slot.name}_0_1_TMP6', slot.x+slot.n, x)
		
		y_overlap = bind(model, f'bind_{x}_{y}_to_{slot.name}_0_0_TMP9', slot.y, y)
		y_top = bind(model, f'bind_{x}_{y}_to_{slot.name}_0_0_TMP5', slot.y, y+1)
		y_bottom = bind(model, f'bind_{x}_{y}_to_{slot.name}_0_0_TMP6', slot.y+slot.n, y)

		model.Add(y_overlap + x_left + slot.h + slot.active + cell.active < 5)
		model.Add(y_overlap + x_right + slot.h + slot.active + cell.active < 5)
		model.Add(x_overlap + y_top + neg(slot.h) + slot.active + cell.active < 5)
		model.Add(x_overlap + y_bottom + neg(slot.h) + slot.active + cell.active < 5)

	# each slot can horizontally or vertically use this cell with up to maximum of N positions.
	for slot,i,h in product(slots, list(range(N)), [1,0]):
		# beyond maximum length of word, or out of grid
		if i >= len(slot.word): continue
		if (h and x-i < 0) or (not h and y-i < 0): continue

		if isinstance(slot.h, bool) and h != slot.h:
			continue

		bind_active = model.NewBoolVar(f'bind_{x}_{y}_to_{slot.name}_{i}_{h}')
		model.Add(bind_active <= slot.active)

		OnlyEnforceIf(model, bind_active == 0, (neg(slot.h) if h else slot.h))

		x_overlap = bind(model, f'bind_{x}_{y}_to_{slot.name}_{i}_{h}_TMP1', slot.x, x-i*h)
		y_overlap = bind(model, f'bind_{x}_{y}_to_{slot.name}_{i}_{h}_TMP2', slot.y, y-i*(1-h))

		length_cover = model.NewBoolVar(f'bind_{x}_{y}_to_{slot.name}_{i}_{h}_TMP3')
		model.Add(slot.n > i).OnlyEnforceIf(length_cover)
		model.Add(slot.n <= i).OnlyEnforceIf(length_cover.Not())

		intersects = model.NewBoolVar(f'bind_{x}_{y}_to_{slot.name}_{i}_{h}_TMP4')
		model.Add(x_overlap + y_overlap + length_cover == 3).OnlyEnforceIf(intersects)
		model.Add(x_overlap + y_overlap + length_cover < 3).OnlyEnforceIf(intersects.Not())

		OnlyEnforceIf(model, bind_active == 1, [intersects, slot.active, (slot.h if h else neg(slot.h))])
		
		for v in [intersects, slot.active]:
			OnlyEnforceIf(model, bind_active == 0, neg(v))

		cell.bindings[(slot.name, i, h)] = bind_active

		for c,v in cell.letter.items():
			model.Add(slot.word[i] == ord(c)-ord('a')).OnlyEnforceIf(bind_active, v)
			model.Add(slot.word[i] != ord(c)-ord('a')).OnlyEnforceIf(bind_active, v.Not())
			#model.Add(slot.word[i][c] == v).OnlyEnforceIf(bind_active)

	# only one slot is allowed per direction
	model.Add(cell.h == sum([b for k,b in cell.bindings.items() if k[2]]))
	model.Add(cell.v == sum([b for k,b in cell.bindings.items() if not k[2]]))
	
	# dissallow lonely letters eg: floating single letter words
	model.Add(cell.h + cell.v >= 1).OnlyEnforceIf(cell.active)
	return cell



# this is a bit of optimization so we don't need slots for single letters
def allow_single_words(model, cells):
	_cells = defaultdict(lambda: None, cells)
	# the only reason a letter is allowed is for single-letter "words"
	for x,y in product(list(range(N)), list(range(N))):
		c = cells[(x,y)]
		if isinstance(c.active, bool) and not c.active:
			continue

		l,r,u,d = _cells[(x-1,y)],_cells[(x+1,y)],_cells[(x,y-1)],_cells[(x,y+1)]

		if x > 0: OnlyEnforceIf(model, neg(l.active) == 1, [neg(c.h), c.active])
		if x < N-1: OnlyEnforceIf(model, neg(r.active) == 1, [neg(c.h), c.active])
		if y > 0: OnlyEnforceIf(model, neg(u.active) == 1, [neg(c.v), c.active])
		if y < N-1: OnlyEnforceIf(model, neg(d.active) == 1, [neg(c.v), c.active])

def limit_letter_count(model, cells, letter_count, blanks):
	blank_vars = {}
	for c in set(letter_count.keys()) - {'*'}:
		blank_vars[c] = [model.NewBoolVar(f'blank_{c}{i}') for i in range(blanks)]
	model.Add(sum(chain(*list(blank_vars.values()))) <= blanks)

	for c, limit in letter_count.items():
		if c != '*':
			model.Add(sum(x.letter[c] for x in cells.values()) - sum(blank_vars[c]) <= limit)

	return blank_vars

# like a normal OnlyEnforceIf, but allows a mix of static and variable booleans
# useful when some variables are sometimes static.
def OnlyEnforceIf(model, statement, variables):
	if not isinstance(statement, list): statement = [statement]
	if not isinstance(variables, list): variables = [variables]

	static = [v for v in variables if isinstance(v, bool)]
	dynami = [v for v in variables if not isinstance(v, bool)]
	if not all(static): 
		return

	for s in statement:
		if isinstance(s, bool):
			if not s:
				model.AddBoolOr([x.Not() for x in dynami])
		else:
			model.Add(s).OnlyEnforceIf(dynami)

def force_single_component(model, slots, start):
	# we give each slot a number, only valid when active, maximum otherwise
	# each active slot must be connected to a slot with a number one lower.
	# the exception is the slot with number 0.
	# There is only allowed to be one number 0.

	MAX_DEPTH = 12

	ordering = {}
	for i,slot in enumerate(slots):
		ordering[slot.name] = model.NewIntVar(0, MAX_DEPTH, f'order_{i}')

	known_active = start
	model.Add(ordering[known_active] == 0)
	for i,slot in enumerate(slots):
		if slot.name == known_active:
			continue
		OnlyEnforceIf(model, ordering[slot.name] > 0, slot.active)
		OnlyEnforceIf(model, ordering[slot.name] == MAX_DEPTH, neg(slot.active))

	# First make a 'is_connected' bool for each pair of slots
	interacts = {}
	for a,b in combinations(slots, 2):
		if isinstance(a.h, bool) and isinstance(b.h, bool) and a.h == b.h:
			continue

		if isinstance(a.x, int) and isinstance(b.x, int):
			if (a.h and all([b.x < a.x+a.n, b.x >= a.x, b.y <= a.y, b.y+b.n > a.y])) or (b.h and all([a.x < b.x+b.n, a.x >= b.x, a.y <= b.y, a.y+a.n > b.y])):
				interacts[(a.name, b.name)] = True
			continue

		is_connected = model.NewBoolVar(f'connected_{a.name}_{b.name}')
		OnlyEnforceIf(model, [a.h+b.h == 1, a.active == 1, b.active == 1], is_connected)
		OnlyEnforceIf(model, [b.x < a.x+a.n, b.x >= a.x, b.y <= a.y, b.y+b.n > a.y], [a.h, neg(b.h), a.active, b.active, is_connected])
		OnlyEnforceIf(model, [a.x < b.x+b.n, a.x >= b.x, a.y <= b.y, a.y+a.n > b.y], [neg(a.h), b.h, a.active, b.active, is_connected])
		interacts[(a.name, b.name)] = is_connected
		interacts[(b.name, a.name)] = is_connected

	#print('total edges:', len(interacts)//2)

	adjacency_matrix = []
	for a in slots:
		if a.name == known_active: continue
		partners = [b for b in slots if (a.name,b.name) in interacts]
		is_adjacent = [model.NewBoolVar(f'is_turn_adjacent_{a.name}_{b.name}') for b in partners]
		adjacency_matrix.append(is_adjacent)
		for b, v in zip(partners, is_adjacent):
			model.Add(ordering[a.name] == ordering[b.name]+1).OnlyEnforceIf(v)
			if not isinstance(interacts[(a.name, b.name)], bool):
				model.Add(interacts[(a.name, b.name)] == 1).OnlyEnforceIf(v)

		OnlyEnforceIf(model,sum(is_adjacent) > 0, a.active)

	return interacts, adjacency_matrix

def create_single_component(main_word, verticals, h_words, v_words):
	model = cp_model.CpModel()
	fixed_letters_main_word = list(re.sub(r'[A-Z]', ' ', main_word) + '?'*N*(N-1))
	vertical_indices = [i for i,c in enumerate(main_word) if c.isupper()]

	for i,w in enumerate(verticals):
		for j,c in list(enumerate(w))[1:]:
			fixed_letters_main_word[vertical_indices[i]+j*N] = c
	fixed_letters_main_word = ''.join(fixed_letters_main_word)
	#print(fixed_letters_main_word)

	# this allows specific dictionaries for specific board positions
	special_words = []
	j = 0
	for i,c in enumerate(main_word):
		if c.islower(): continue
		if len(verticals[j]) > 1:
			special_words.append((i,1,False,False,[verticals[j][1:]]))
		j += 1

	for p,i in [(m.group(0), m.start()) for m in re.finditer(r'[a-z]+', main_word)]:
		if len(p) <= 0: continue
		special_words.append((i, 0, True, False, [p]))

	#for x in special_words:
	#	print(*x[:-1], x[-1][:10])

	modified_counts = Counter(counts)
	#for c in abc:
	#	modified_counts[c] *= 2

	for c in main_word:
		if c.isupper():
			modified_counts[c.lower()] -= 1

	basic_slots = []
	special_slots = []
	extra_slots = defaultdict(list)

	for i,(x,y,h, optional,d) in enumerate(special_words):
		slot = make_slot(model, f'specialslot_{i}', d, x, y, h, not optional)
		for i in range(max(len(x) for x in d)):
			extra_slots[(x+i*h, y+i*(1-h))].append(slot)
		special_slots.append(slot)

	# some slots per direction, turns out to be most efficient
	# 10 horizontal words (excluding static at row 0)
	#for repeat,n in enumerate([5,4, 3,3,3 ,2,2,2,2,2]):
	for repeat,n in enumerate([15,10,7,5,4,4, 3,3,3 ,2,2,2,2,2]):
		_words = [w for w in h_words if 2 <= len(w) <= n]
		slot = make_slot(model, f'basicslot_h_{repeat}', _words, -1, -1, True)
		basic_slots.append(slot)

	# 6 vertical words
	#for repeat,n in enumerate([4,3,3,2,2,2,2,2]):
	for repeat,n in enumerate([15,9,5,4,3,3,2,2,2,2,2]):
		_words = [w for w in v_words if 2 <= len(w) <= n]
		if _words:
			slot = make_slot(model, f'basicslot_v_{repeat}', _words, -1, -1, False)
			model.Add(slot.y != 1)
			basic_slots.append(slot)

	for i,j in product(range(N), range(N)):
		extra_slots[(i,j)] += basic_slots

	#print('total slots: ', len(basic_slots)+len(special_slots))
	cells = {(x,y):make_cell(model, x, y, extra_slots[(x,y)]) for x,y in product(list(range(N)), list(range(N)))}
	allow_single_words(model, cells)
	#model.Add(sum(cell.active for cell in cells.values()) <= 102-1-7)

	blank_vars = limit_letter_count(model, cells, modified_counts, modified_counts['*'])
	graph = force_single_component(model, basic_slots+special_slots, special_slots[2].name)
	
	solver = cp_model.CpSolver()
	#solver.parameters.log_search_progress = True
	solver.parameters.max_presolve_iterations = 1
	solver.parameters.num_search_workers = 8
	solver.parameters.optimize_with_core = True

	status = solver.Solve(model)
	if status in [cp_model.FEASIBLE, cp_model.OPTIMAL]:
		print('\nBOARD OVERVIEW')
		lines = []
		for j in range(N):
			lines.append(''.join([(''.join(c for c,v in cells[(i,j)].letter.items() if (b if isinstance(v, bool) else solver.Value(v))) + ' ')[0] for i in range(N)]))
		print('\n'.join(lines))

		for i,k in enumerate(special_words):
			if k[1] == 1:
				if k[0] in (0,7,14): print('x', k[0], main_word[k[0]])
				elif k[0] in (3,11): print('x', k[0], main_word[k[0]])
				else: print('x', k[0], main_word[k[0]])

		if special_slots:
			print('\nSPECIAL SLOTS')
			for slot in special_slots:
				word = ''.join([(chr(solver.Value(letter) + ord('a'))) for letter in slot.word])
				print(slot.name, word, f'active: {solver.Value(slot.active)}, x: {solver.Value(slot.x)}, y: {solver.Value(slot.y)}, h: {solver.Value(slot.h)}, n: {solver.Value(slot.n)}')

		print('\nBASIC SLOTS')
		for slot in basic_slots:
			word = ''.join([(chr(solver.Value(letter) + ord('a'))) for letter in slot.word])
			print(slot.name, word, f'active: {solver.Value(slot.active)}, x: {solver.Value(slot.x)}, y: {solver.Value(slot.y)}, h: {solver.Value(slot.h)}, n: {solver.Value(slot.n)}')

		print('\nCELL DETAILS')
		for cell in cells.values():
			if not isinstance(cell.active, bool):
				letter = ''.join(c for c,v in cell.letter.items() if (b if isinstance(v, bool) else solver.Value(v)))
				active_bindings = [f'{name} n{i} h{h}: {solver.Value(binding_var)}' for (name, i, h), binding_var in cell.bindings.items() if solver.Value(binding_var)]
				print(cell.x, cell.y, letter, f'active: {solver.Value(cell.active)}, h: {solver.Value(cell.h)}, v: {solver.Value(cell.v)}', ' '.join(active_bindings))


## Preprocessing
#####################
def valid_positions(board):
	# all valid position on an empty board
	positions = []
	for x,y in product(range(N),range(N)):
		positions += [(x,y,True, x2-x+1) for x2 in range(x,N)]
		positions += [(x,y,False,y2-y+1) for y2 in range(y,N)]

	# don't modify existing static words
	positions = [(x,y,h,n) for x,y,h,n in positions if not (
		(h and x > 0 and not board[y*N+x-1] in ' -') or 
		(h and x+n < N and not board[y*N+x+n] in ' -') or 
		(not h and y > 0 and not board[(y-1)*N+x] in ' -') or 
		(not h and y+n < N and not board[(y+n)*N+x] in ' -') or
		('-' in [board[(y+i*(1-h))*N+(x+i*h)] for i in range(n)] ))]

	return positions

def word_to_count(w): return np.array([w.count(chr(97+i)) for i in range(26)], dtype=np.int8)
def word_to_arr(w): return np.array([ord(c)-97 for c in w], dtype=np.int8)

# group by length, then create matrices that (for each word) give the word, and their letter counts
def build_index(words):
	by_length = {i:[] for i in range(1,16)}
	[by_length[len(w)].append(w) for w in words]
	by_length_arr = {k:np.array([word_to_arr(w) for w in v], dtype=np.int8) if len(v) else np.zeros((0,k),dtype=np.int8) for k,v in by_length.items()}
	by_length_cnt = {k:np.array([word_to_count(w) for w in v], dtype=np.int8) if len(v) else np.zeros((0,26),dtype=np.int8) for k,v in by_length.items()}
	return by_length, by_length_arr, by_length_cnt

# a 26 sized array that specifies the amount of non-blank tiles remaining, combined with a simply count of blanks
def remaining_tiles_array(main_word, board):
	tilecount_arr = np.array([counts[chr(i+97)] for i in range(26)], dtype=np.int8)
	tilecount_arr -= word_to_count(''.join(c.lower() for c in main_word if c.isupper()) + board.replace('-', '').replace(' ', ''))
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
		letters_placed = word_to_count(fixed)
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

def split_direction(words_per_location):
	horizontals = set()
	verticals = set()
	for (x,y,h,n), fits in words_per_location.items():
		if h: horizontals |= set(fits)
		if not h: verticals |= set(fits)
	return horizontals, verticals

def words_for_board(main_word, board):
	_board = board
	_words = None
	positions = valid_positions(board)
	# generally more than 1 iteration only filters a handful of additional words
	for i in range(1):
		words_per_location = find_fitting_words(main_word, board, _board, _words, positions)
		_board = letters_per_cell(board, words_per_location, positions)
		_words = {w for v in words_per_location.values() for w in v}
		horizontals, verticals = split_direction(words_per_location)

	return _board, horizontals, verticals

def still_possible(main_word, board):
	groups = [(g.start(), len(g.group())) for g in re.finditer(r'[a-z]+', main_word)]
	return all(any(board[N+i] for i in range(s,s+n)) for s,n in groups)

def quick_validity_check(main_word, board):
	positions = valid_positions(board)
	positions = [v for v in positions if v[1] <= 1]
	words_per_location = find_fitting_words(main_word, board, board, None, positions)
	_board = letters_per_cell(board, words_per_location, positions)
	return still_possible(main_word, _board)

if __name__ == '__main__':
	WORD_CACHE = build_index(set(words)|set(abc))
	for line in sys.stdin:
		line = line.strip()
		(score, blanks, main_word, verticals) = line.split(' ', 3)

		# for example, for dutch this might be an example input that solves
		#main_word = 'GesChenKCheQUeS'  
		#verticals = ['glansvogeltjes', 'crayonerenden', 'klastemperatuur', 'capex', 'qat', 'uzelf', 'strekbeweginkje']

		verticals = verticals.split(' ')
		verticals = [verticals[i] for i in [0, 5, 1, 6, 3, 4, 2]]

		print(score, blanks, main_word, verticals)

		board = list(' '*(N*N))
		for i,c in enumerate(main_word):
			board[i] = c if c.islower() else '-'

		score_check = [word_score(main_word, 0,0,1)]
		for i,v in zip([i for i,c in enumerate(main_word) if c.isupper()], verticals):
			if v:
				score_check.append(word_score(v.capitalize(), i, 0, 0))
				for j,c in enumerate((v+'-')[1:N]):
					board[(j+1)*N+i] = c


		print(sum(score_check), score_check)

		board = ''.join(board)

		if not quick_validity_check(main_word, board):
			print('impossible by static quick-analysis')
			continue
		

		t0 = time.time()
		board_limits, fit_h, fit_v = words_for_board(main_word, board)
		possible = still_possible(main_word, board_limits)
		print(f'{int(time.time()-t0)}s', len(fit_h), len(fit_v))

		if not possible:
			print('impossible by static analysis, NEVER HAPPENS')
			continue

		t0 = time.time()
		create_single_component(main_word, verticals, fit_h, fit_v)
		print(f'{int(time.time()-t0)}s')
