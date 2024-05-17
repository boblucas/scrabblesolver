Solves some artifical scrabble problems using constraint solving.

### Dependencies

```
pip install ortools lexpy numpy
```

### Usage

Basic usage right now is to first generate all candidate solutions as such:
```
python3 generate_candidates.py
```

Which will generate files with lines as such
```
1785 2 OXyPhenButaZoNE opacifications_0 bladderlike_7 establishments_14 xerotic_1 zoogamete_11 novation_13 prequalifying_3
```

And then you can judge can see which ones are solveable as such (in order).
```
cat ./*.CWS23 | sort -k1,1nr -k2,2n | python3 is_single_component.py
```
