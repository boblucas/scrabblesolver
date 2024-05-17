from scrabble import *
import numpy as np
from itertools import *
import multiprocessing

THRESHOLD = 8

# word15 is the word as tiled, as thus may include blanks.
# word15_og is the word originally, so only a-z
# will yield lists of fitting words within tile limits in order of score
# find_minimum: give minimum aggregate score for given x positions
# all_at_position: give all unique words that fit at x position
ALL_VALID_CACHE = {}
def find_vertical_fixwords(word15, word15_og, positions = [0, 7, 14], prefix = True, find_minimum = [], all_at_position = -1, minimum_points = 0):
	fixable = prefixable if prefix else postfixable
	word_positions = {p:fixable[word15_og[p]] for p in positions}

	model = cp_model.CpModel()
	letter_counts_x3 = defaultdict(int)
	letter_counts_x1 = defaultdict(int)

	def _word_score(w, x):
		if prefix:
			return word_score(w.capitalize(), k, 0, False)
		return word_score(w[:-1] + w[-1].upper(), k, 15-len(w), False)

	word_positions_vars = {}
	for k,v in word_positions.items():
		word_positions_vars[k] = {len(v):model.NewBoolVar(f'WORD__{len(v)}')}
		for i,w in enumerate(v):
			word_positions_vars[k][i] = model.NewBoolVar(f'WORD_{w}_{i}')
			for c,v in Counter(w[1:] if prefix else w[:-1]).items():
				if k in [0,7,14]:
					letter_counts_x3[c] += word_positions_vars[k][i] * v
				else:
					letter_counts_x1[c] += word_positions_vars[k][i] * v

		model.Add(sum(word_positions_vars[k].values()) == 1)

	''' This code rejects adjacent sets of vertical words if their horizonal
	substrings are trivially impossible to satisfy. This turns out to be not
	a strong enough filter to be worth the compute right now. It will be picked
	up by the final solver
	'''
	if False:
		substrings = defaultdict(lambda: (16,16))
		for w in words:
			for i in range(len(w)-1):
				for n in [2,3]:
					l,r = substrings[w[i:i+n]]
					substrings[w[i:i+n]] = (min(l,i), min(r,len(w)-(i+n)))

		def adjacent_valid(start, rows):
			lspace = start
			rspace = 14-(start+len(rows))
			rows = [w.Name().split('_')[1] for w in rows]
			columns = [''.join(w).replace('-','') for w in zip_longest(*rows, fillvalue='-')]
			#return all(len(w) == 1 or w in words for w in columns)
			return all(len(w) == 1 or (substrings[w][0] <= lspace and substrings[w][1] <= rspace) for w in columns)

		# proper way to do is is to look at all valid additional letters at each vertical position, mask
		# out the confliction words, repeat until you are at length
		for b,e in zip([i for i in positions if not i-1 in positions], [i+1 for i in positions if not i+1 in positions]):
			total = math.prod(len(word_positions_vars[i]) for i in range(b,e))
			if e-b == 1: continue
			#if total > 500_000_000: 
			#	print(word15_og[b:e], 'is too expensive to encode because it has', total, 'options (TODO)')
			#	continue

			k = (b, word15_og[b:e])
			if k in ALL_VALID_CACHE:
				all_valid = ALL_VALID_CACHE[k]
			else:
				tuplets = [words for words in product(word_positions_vars[b].values(), word_positions_vars[b+1].values()) if adjacent_valid(b, words)]
				print(len(tuplets))
				for i in range(b+2, e):
					tuplets = [(*words, w) for (words, w) in product(tuplets, word_positions_vars[i].values()) if adjacent_valid(b, (*words, w))]
					print(len(tuplets))
				all_valid = ALL_VALID_CACHE[k] = tuplets

			if len(all_valid) == 0:
				print(word15_og[b:e], 'has no valid verticals, allowing no active adjacent pairs')
				for i in range(b, e-1):
					model.Add(sum(sum(word_positions_vars[j].values()) for j in range(i, i+2)) <= 1)
			else:
				print(word15_og[b:e], f'has {len(all_valid)} possible verticals of {total}')
				if (len(all_valid)/total > 0.9 and len(all_valid) > 1000) or len(all_valid) > 3_000_000:
					print('Not encoding, too expensive')
					continue

				for w, group in groupby(all_valid, key=lambda x: x[0]):
					model.Add(sum([x[1] for x in group]) == 1).OnlyEnforceIf(w)
	
	blanks_used = 0
	lost_by_blanks = 0
	letters_left = counts - Counter(word15.lower())
	for c in abc:
		count_x1 = letter_counts_x1[c]
		count_x3 = letter_counts_x3[c]

		total_count = model.NewIntVar(0, 32, f'sum_{c}')
		model.Add(total_count == count_x1+count_x3)

		blank1 = model.NewBoolVar(f'blank_1_{c}')
		blank2 = model.NewBoolVar(f'blank_2_{c}')
		model.Add(blank2 <= blank1)
		model.Add(total_count-0 <= letters_left[c]).OnlyEnforceIf([blank1.Not(), blank2.Not()])
		model.Add(total_count-1 == letters_left[c]).OnlyEnforceIf(blank1, blank2.Not())
		model.Add(total_count-2 == letters_left[c]).OnlyEnforceIf(blank1, blank2)

		blank1_x3 = model.NewBoolVar(f'blank_1x3_{c}')
		blank2_x3 = model.NewBoolVar(f'blank_2x3_{c}')
		model.Add(blank2_x3 == 1).OnlyEnforceIf(blank1_x3)
		model.Add(count_x1 >= blank1).OnlyEnforceIf(blank1_x3.Not())
		model.Add(count_x1 >= blank1+blank2).OnlyEnforceIf(blank2_x3.Not())

		lost_by_blanks += blank1_x3*2*scores[c] + blank1*scores[c]
		lost_by_blanks += blank2_x3*2*scores[c] + blank2*scores[c]
		blanks_used += blank1 + blank2

	total_score = -lost_by_blanks
	for k,v in word_positions.items():
		for i,w in enumerate(v):
			total_score += word_positions_vars[k][i]*_word_score(w, k)

	if minimum_points:
		total_score_var = model.NewIntVar(minimum_points, minimum_points+60, 'total_score')
		model.Add(total_score == total_score_var)
		model.Add(total_score_var > minimum_points)

	solver = cp_model.CpSolver()

	if find_minimum is not None:
		sub_word_score = 0
		for k,v in word_positions.items():
			if k in find_minimum:
				for i,w in enumerate(v):
					sub_word_score += word_positions_vars[k][i]*_word_score(w, k)
		model.Minimize(sub_word_score)
	else:
		model.Maximize(total_score)
		solver.parameters.max_time_in_seconds = 120.0

	# the problem really just needs a normal search
	solver.parameters.num_search_workers = 2
	solver.parameters.max_presolve_iterations = 1
	#solver.parameters.log_search_progress = True
	solver.parameters.optimize_with_core = False

	while (status := solver.Solve(model)) != 3:
		words_selected = [v.Name().split('_')[1] for pos in word_positions_vars.values() for v in pos.values() if solver.Value(v)]
		blanks_required = solver.Value(blanks_used)
		#if find_minimum:
		print(int(solver.ObjectiveValue()), *words_selected)
		yield solver.ObjectiveValue()
		return

		if all_at_position >= 0:
			word = [v.Name().split('_')[1] for v in word_positions_vars[all_at_position].values() if solver.Value(v)][0]
			print(word)
			yield word
			model.Add(sum(v for v in word_positions_vars[all_at_position].values() if solver.Value(v)) == 0)
		# score = solver.Value(total_score_var)
		#return score, blanks_required, [v.Name().split('_')[1] for pos in word_positions_vars.values() for v in pos.values() if solver.Value(v)]
		#model.Add(sum(v for pos in word_positions_vars.values() for v in pos.values() if solver.Value(v)) < len(positions))

def find_minimum(x):
	main, prefix, order, minimum_points = x
	try:
		return list(find_vertical_fixwords(main, main.lower(), [i for i in range(len(main)) if main[i].isupper()], prefix, order, -1, minimum_points))[0]
	except IndexError:
		return None

def word_to_np(w):
	w = [ord(c)-ord('a') for c in w]
	return np.array([w.count(i) for i in range(26)], dtype=np.int8)

def all_placements(words):
	# collect all words, possibly requiring blanks, possible with multiple options
	w15 = [w for w in words if len(w) == N]
	possible_15_words = [_w for w in set(w15) for _w in sufficient_tiles(w)]
	
	# filter within a range  to only top ones with cheap scoring check
	possible_15_words = sorted([(word_score(w.upper(),0,0,1), w) for w in possible_15_words])[::-1]
	
	# filter with more correct scoring chek	
	possible_15_words = sorted([(max_word_score_ideal(w,0,0,1)[0], w) for score,w in possible_15_words if score > possible_15_words[0][0]-100])[::-1]
	possible_15_words = [w for score,w in possible_15_words if score > possible_15_words[0][0]-120]

	# for the remaining words get all specific placements
	# and filter again on upperbound of score
	def convert_parts(w, parts):
		w_marked = w.upper()
		for i,p in parts:
			w_marked = w_marked[:i] + p + w_marked[i+len(p):]
		return w_marked

	def score_estimate(w):
		return word_score(w,0,0,1) + sum(word_score(c,i,0,0) for i,c in enumerate(w) if c.isupper())

	possible_15_words = sorted([(score_estimate(convert_parts(w, p)), convert_parts(w,p)) for w in possible_15_words for p in possible_partial_placements(w,0,0,1,True)])[::-1]
	possible_15_words = sorted(set(possible_15_words))[::-1]
	possible_15_words = [w for score,w in possible_15_words if score > possible_15_words[0][0]-80]

	print(len(possible_15_words), 'possible 15-letter words (including all variations in blanks and placement)')
	for w in possible_15_words:
		print(score_estimate(w), w)

	return possible_15_words

if __name__ == '__main__':
	print('using', word_file)
	placements_filtered = all_placements(words)

	print('Computing approximate max score (with the assumption we can connect anything)')
	main = placements_filtered[0]
	approx_max_possible = word_score(main,0,0,1) + int(list(find_vertical_fixwords(main, main.lower(), [i for i in range(len(main)) if main[i].isupper()], True, None))[0])
	print(f'using {main} we can probably find no more than {approx_max_possible} points, will list everything down to {approx_max_possible-THRESHOLD} points exhaustively')

	for prefix in [True, False]:
		fixable = prefixable if prefix else postfixable
		# this runs for a couple of hours and generates all possible >threshold scoring layouts 
		for main_i, main in enumerate(placements_filtered):
			if main_i == 0: continue
			order = [0,7,14]
			order += sorted([i for i,c in enumerate(main) if not i in order and c.isupper()], key=lambda i: len(fixable[main[i].lower()]))
			
			print('Computing minimum points required for each additional word, for quick traversal later')
			base_score = word_score(main,0,0,1)
			# within 5 of the maximum is chosen here as a well beyond reasonable value
			# if all those results turn out to be unconnectable, then you will have to increase this value
			minimum_points = approx_max_possible-THRESHOLD-base_score
			with multiprocessing.Pool(len(order)) as p:
				minimum_scores = list(p.imap(find_minimum, [(main, prefix, order[:i+1], minimum_points) for i in range(len(order))]))
				if None in minimum_scores:
					print(f'cannot get {minimum_points} points using {main} at {"A" if prefix else "O"}1')
					continue

			#try:
			#	minimum_scores = [int(list(find_vertical_fixwords(main, main.lower(), [i for i in range(len(main)) if main[i].isupper()], prefix, order[:i+1], -1, minimum_points))[0]) for i in range(len(order))]
			#except IndexError:
			#	continue

			score_arr = np.array([scores[c] for c in 'abcdefghijklmnopqrstuvwxyz'], dtype=np.int16)

			def word_index_to_x(i):
				for j,(a,b) in enumerate(zip(start_indices, start_indices[1:])):
					if a <= i < b:
						return order[j] 

			verticals = list(chain(*[fixable[main[i].lower()] for i in order]))
			start_indices = [len(fixable[main[i].lower()]) for i in order]
			start_indices = [sum(start_indices[:i]) for i in range(len(start_indices)+1)]
			wordscores = np.array([word_score(w.capitalize(), word_index_to_x(i) ,0,0) for i,w in enumerate(verticals)], dtype=np.int16)
			all_tiles_used = np.array([word_to_np(w[1:]) for w in verticals], dtype=np.int8)
			tilecount_arr = np.array([counts[chr(i+ord('a'))] for i in range(26)], dtype=np.int8)
			tilecount_arr -= word_to_np(main.lower())

			# backtrack is a n x m array of indices to words
			# tiles_used is a n x 26 array
			# score is a n sized array
			# next_words is a list of indices to words
			backtrack = np.array([[x] for x in range(start_indices[0], start_indices[1])])
			tiles_used = all_tiles_used[start_indices[0]:start_indices[1]]

			blank_score = np.sum(score_arr * np.clip((tilecount_arr - tiles_used),-3,0), axis=1) * 3
			score_used = wordscores[start_indices[0]:start_indices[1]] + blank_score

			for vertical in range(len(order)-1):
				total = 0
				_backtrack = []#np.zeros((0,vertical+2), dtype=np.uint16)
				_tiles_used = np.zeros((0,26), dtype=np.int8)
				_score_used = np.zeros((0,), dtype=np.int16)
				for i,w in enumerate(np.array(range(start_indices[vertical+1], start_indices[vertical+2]))):
					blanks_prev = np.clip((tilecount_arr - tiles_used),-3,0)
					blanks = np.clip((tilecount_arr - all_tiles_used[w] - tiles_used),-3,0)
					subset = np.sum(blanks, axis=1) >= -2

					blank_score = np.sum(score_arr * (blanks-blanks_prev), axis=1) * ([1,3][vertical < 2])
					subset &= (score_used + wordscores[w] + blank_score) >= minimum_scores[vertical+1]
					subset = np.where(subset)[0]

					if len(subset) == 0:
						continue

					new_word = np.expand_dims(np.repeat(w, subset.shape[-1]), axis=1)
					new_sets = np.concatenate((backtrack[subset], new_word), axis=1)

					# debug, prints/takes single example and verifies scoring (running total) is still correct
					if False:
						vertical_words = list(zip([verticals[i] for i in new_sets[0]], order))
						v_with_blanks = next(sufficient_tiles(' '.join([w[1:] for w,j in vertical_words[::-1]]) + ' ' + main.lower())).split(' ')[:-1][::-1]
						expected_score = sum([word_score(main[j] + w, j, 0, False) for w,j in zip(v_with_blanks, order)])
						print(i, start_indices[vertical+2]-start_indices[vertical+1], total, 
							[verticals[i] for i in new_sets[0]], 
							minimum_scores[vertical+1], 
							score_used[subset[0]] + wordscores[w] + blank_score[subset[0]], 'should be', expected_score)
						assert (score_used[subset[0]] + wordscores[w] + blank_score[subset[0]] == expected_score), "invalid score calculation"

					total += len(new_sets)
					_backtrack.append(new_sets)
					_tiles_used = np.concatenate((_tiles_used, tiles_used[subset] + all_tiles_used[w]), axis=0)
					_score_used = np.concatenate((_score_used, score_used[subset] + wordscores[w] + blank_score[subset]), axis=0)

				if len(_backtrack) == 0:
					print('out of options within threshold')
					break

				_backtrack = np.concatenate(_backtrack, axis=0)
				backtrack, tiles_used, score_used = _backtrack, _tiles_used, _score_used
				print(f'{len(backtrack)} options, best scoring:', *[verticals[i] for i in backtrack[0]])
			else:
				print(f'saving to {main}_verticals.{main_i}.{int(prefix)}.{word_file}3')
				f = open(f'{main}_verticals.{main_i}.{word_file}3', 'w')
				for line, score, tiles in zip(backtrack, score_used, tiles_used):
					blanks = np.clip((tilecount_arr - tiles),-3,0)
					solution = f"{main} {' '.join([verticals[i] + '_' + str(p) for i,p in zip(line, order)])}"
					f.write(f'{base_score+score} {-np.sum(blanks)} {solution}\n')

