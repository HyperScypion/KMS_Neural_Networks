#!/usr/bin/python3
# coding: utf-8


def yield_generate():
    a, b = [], []
    for i in range(1, 12345):
        a.append(i)
	    for j in range(1, 12345):
		    b.append(j)
	    a += b
	    yield a


g = yield_generate()
for i in g:
    print(i)
